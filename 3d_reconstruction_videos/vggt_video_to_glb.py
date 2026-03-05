#!/usr/bin/env python3
"""Extract frames from video and run VGGT to produce a point cloud GLB."""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import trimesh
import open3d as o3d
import onnxruntime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "vggt"))
from visual_util import run_skyseg, download_file_from_url

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def extract_frames(video_path, output_dir, max_frames=50):
    """Sample up to max_frames evenly from the video."""
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // max_frames)
    indices = list(range(0, total, step))[:max_frames]
    print(f"Video: {total} frames, sampling every {step} → {len(indices)} frames")
    saved = []
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(path, frame)
        saved.append(path)
    cap.release()
    print(f"Saved {len(saved)} frames to {output_dir}")
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output GLB path")
    parser.add_argument("--checkpoint", required=True, help="Path to vggt_1B_commercial.pt")
    parser.add_argument("--max_frames", type=int, default=50, help="Max frames to sample")
    parser.add_argument("--conf_percentile", type=float, default=25.0,
                        help="Filter out lowest N%% confidence points (0=keep all, 50=keep top half)")
    parser.add_argument("--mask_sky", action="store_true", default=False,
                        help="Remove sky points using ONNX sky segmentation model")
    parser.add_argument("--skyseg_model", type=str, default="skyseg.onnx",
                        help="Path to skyseg.onnx model")
    parser.add_argument("--voxel_size", type=float, default=0.005,
                        help="Voxel size for downsampling (merges redundant overlapping points). 0=disabled")
    parser.add_argument("--sor_neighbors", type=int, default=20,
                        help="Statistical outlier removal: number of neighbors to check. 0=disabled")
    parser.add_argument("--sor_std_ratio", type=float, default=2.0,
                        help="SOR: remove points beyond this many std devs from mean neighbor distance")
    args = parser.parse_args()

    # ── 1. Extract frames ────────────────────────────────────────────────────
    frames_dir = Path(args.output).parent / (Path(args.video).stem + "_frames")
    image_paths = extract_frames(args.video, str(frames_dir), args.max_frames)
    if not image_paths:
        sys.exit("No frames extracted.")

    # ── 2. Load model ────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"Device: {device}  dtype: {dtype}")

    print(f"Loading checkpoint from {args.checkpoint} ...")
    model = VGGT()
    state = torch.load(args.checkpoint, map_location="cpu")
    # Handle wrapped state dicts
    if "model" in state:
        state = state["model"]
    elif "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    # Disable heads we don't need to save VRAM
    model.point_head = None
    model.track_head = None
    # fp16 halves model size (~2.35GB vs ~4.7GB)
    model = model.half().eval().to(device)

    # Monkey-patch forward to remove autocast(enabled=False) which causes
    # fp32/fp16 dtype mismatch when model weights are in fp16
    def _forward_fp16(self, images, query_points=None):
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        predictions = {}
        if self.camera_head is not None:
            pose_enc_list = self.camera_head(aggregated_tokens_list)
            predictions["pose_enc"] = pose_enc_list[-1]
        if self.depth_head is not None:
            depth, depth_conf = self.depth_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            )
            predictions["depth"] = depth
            predictions["depth_conf"] = depth_conf
        predictions["images"] = images
        return predictions

    import types
    model.forward = types.MethodType(_forward_fp16, model)

    # Fix: vggt/heads/utils.py make_sincos_pos_embed hardcodes `return emb.float()`
    # which upcasts pos_embed to fp32, then x + pos_embed upcasts x to fp32 too,
    # causing dtype mismatch with fp16 conv weights in resize_layers.
    import vggt.heads.utils as _vggt_utils
    _orig_make_sincos = _vggt_utils.make_sincos_pos_embed
    def _make_sincos_dtype_aware(embed_dim, pos, omega_0=100):
        return _orig_make_sincos(embed_dim, pos, omega_0=omega_0).to(pos.dtype)
    _vggt_utils.make_sincos_pos_embed = _make_sincos_dtype_aware
    # Also patch in the dpt_head module namespace (it imports the function directly)
    import vggt.heads.dpt_head as _dpt_head_mod
    _dpt_head_mod.make_sincos_pos_embed = _make_sincos_dtype_aware
    print("Model loaded.")

    # ── 3. Preprocess images ─────────────────────────────────────────────────
    print("Preprocessing images...")
    images = load_and_preprocess_images(image_paths, mode="pad").half().to(device)
    print(f"Images tensor: {images.shape}")

    # ── 4. Inference ─────────────────────────────────────────────────────────
    print("Running VGGT inference...")
    with torch.no_grad():
        predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )

    # Move to CPU numpy
    extrinsic_np = extrinsic.squeeze(0).cpu().float().numpy()
    intrinsic_np = intrinsic.squeeze(0).cpu().float().numpy()
    depth_map = predictions["depth"].squeeze(0).cpu().float().numpy()        # (S, H, W, 1)
    depth_conf = predictions["depth_conf"].squeeze(0).cpu().float().numpy()  # (S, H, W)

    # ── 5. Unproject to 3D ───────────────────────────────────────────────────
    print("Unprojecting depth to 3D points...")
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic_np, intrinsic_np)  # (S, H, W, 3)

    # RGB colors from input images (predictions["images"] has batch dim)
    imgs_np = predictions["images"].squeeze(0).cpu().float().numpy()  # (S, 3, H, W)
    colors = imgs_np.transpose(0, 2, 3, 1)  # (S, H, W, 3)
    colors = (colors * 255).clip(0, 255).astype(np.uint8)

    # ── 6. Confidence filter ─────────────────────────────────────────────────
    conf_flat = depth_conf.reshape(-1)
    thresh = np.percentile(conf_flat, args.conf_percentile)
    mask = (depth_conf >= thresh) & (depth_conf > 0.1)  # (S, H, W)

    pts = points_3d[mask]       # (N, 3)
    clrs = colors[mask]         # (N, 3)
    print(f"Point cloud: {len(pts):,} points after {args.conf_percentile}% confidence filter")

    # ── 7. Sky masking (applied after confidence filter) ─────────────────────
    if args.mask_sky:
        skyseg_path = args.skyseg_model
        if not os.path.exists(skyseg_path):
            print("Downloading sky segmentation model...")
            download_file_from_url(
                "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx",
                skyseg_path,
            )
        print("Running sky segmentation on frames...")
        skyseg_session = onnxruntime.InferenceSession(skyseg_path)
        S, H, W = depth_conf.shape
        sky_mask = np.zeros((S, H, W), dtype=bool)  # True = sky
        for i, img_path in enumerate(image_paths):
            img_bgr = cv2.imread(img_path)
            result = run_skyseg(skyseg_session, [320, 320], img_bgr)
            raw = cv2.resize(result, (W, H)).astype(np.float32)
            raw = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
            sky_mask[i] = raw < 0.125
        # Apply sky mask to the already-filtered points
        not_sky = ~sky_mask[mask]
        pts = pts[not_sky]
        clrs = clrs[not_sky]
        pct = 100 * not_sky.size - not_sky.sum()
        print(f"After sky removal: {len(pts):,} points (removed {not_sky.size - not_sky.sum():,} sky points)")

    # ── 8. Voxel downsample + SOR ────────────────────────────────────────────
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(clrs[:, :3].astype(np.float64) / 255.0)

    if args.voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)
        print(f"After voxel downsample (size={args.voxel_size}): {len(pcd.points):,} points")

    if args.sor_neighbors > 0:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=args.sor_neighbors, std_ratio=args.sor_std_ratio
        )
        print(f"After SOR (neighbors={args.sor_neighbors}, std={args.sor_std_ratio}): {len(pcd.points):,} points")

    pts_clean = np.asarray(pcd.points).astype(np.float32)
    pts_clean[:, 1] *= -1  # flip 180° upside down
    clrs_clean = (np.asarray(pcd.colors) * 255).clip(0, 255).astype(np.uint8)

    # ── 9. Export GLB ────────────────────────────────────────────────────────
    print(f"Exporting GLB to {args.output} ...")
    pc = trimesh.PointCloud(vertices=pts_clean, colors=clrs_clean)
    pc.export(args.output)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"Done! {args.output}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
