#!/usr/bin/env python3
"""Convert a 360° equirectangular panorama to a point cloud GLB using VGGT depth estimation.
Uses the same pipeline as vggt_video_to_glb.py — perspective crops are treated as video frames."""

import os
import sys
import argparse
import numpy as np
import torch
import cv2
import trimesh
import open3d as o3d
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "vggt"))

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import types
import vggt.heads.utils as _vggt_utils
import vggt.heads.dpt_head as _dpt_head_mod


def equirect_to_perspective(equirect, yaw_deg, pitch_deg, fov_deg=90, out_size=518):
    """Extract a perspective crop from an equirectangular image."""
    H_eq, W_eq = equirect.shape[:2]
    fov = np.radians(fov_deg)
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)

    xs = np.linspace(-1, 1, out_size) * np.tan(fov / 2)
    ys = np.linspace(-1, 1, out_size) * np.tan(fov / 2)
    xv, yv = np.meshgrid(xs, ys)
    zv = np.ones_like(xv)

    norm = np.sqrt(xv**2 + yv**2 + zv**2)
    xv, yv, zv = xv / norm, yv / norm, zv / norm

    # Pitch rotation (around X)
    xr = xv
    yr = yv * np.cos(pitch) - zv * np.sin(pitch)
    zr = yv * np.sin(pitch) + zv * np.cos(pitch)
    # Yaw rotation (around Y)
    xf = xr * np.cos(yaw) + zr * np.sin(yaw)
    yf = yr
    zf = -xr * np.sin(yaw) + zr * np.cos(yaw)

    lon = np.arctan2(xf, zf)
    lat = np.arcsin(np.clip(yf, -1, 1))

    map_x = ((lon / np.pi + 1) / 2 * W_eq).astype(np.float32)
    map_y = ((-lat / (np.pi / 2) + 1) / 2 * H_eq).astype(np.float32)

    crop = cv2.remap(equirect, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    return crop


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panorama", required=True, help="Input equirectangular panorama (jpg/png)")
    parser.add_argument("--output", required=True, help="Output GLB path")
    parser.add_argument("--checkpoint", required=True, help="Path to vggt_1B_commercial.pt")
    parser.add_argument("--fov", type=float, default=90.0, help="FOV of each perspective crop (degrees)")
    parser.add_argument("--crop_size", type=int, default=518, help="Crop resolution (px)")
    parser.add_argument("--num_yaw", type=int, default=12, help="Number of horizontal viewpoints")
    parser.add_argument("--conf_percentile", type=float, default=25.0, help="Drop lowest N%% confidence points")
    parser.add_argument("--voxel_size", type=float, default=0.005, help="Voxel downsample size (0=off)")
    parser.add_argument("--sor_neighbors", type=int, default=20, help="SOR neighbors (0=off)")
    parser.add_argument("--sor_std_ratio", type=float, default=2.0)
    args = parser.parse_args()

    # ── 1. Extract perspective crops ─────────────────────────────────────────
    print(f"Loading panorama: {args.panorama}")
    equirect = cv2.imread(args.panorama)
    equirect = cv2.cvtColor(equirect, cv2.COLOR_BGR2RGB)
    print(f"Panorama size: {equirect.shape[1]}x{equirect.shape[0]}")

    views = []
    for yaw in np.linspace(0, 360, args.num_yaw, endpoint=False):
        views.append((yaw, 0.0))
    views.append((0.0,  70.0))  # top
    views.append((0.0, -70.0))  # bottom

    print(f"Extracting {len(views)} perspective crops (FOV={args.fov}°, {args.crop_size}px)...")
    tmp_dir = Path(args.output).parent / "pano_crops"
    tmp_dir.mkdir(exist_ok=True)
    image_paths = []
    for i, (yaw, pitch) in enumerate(views):
        crop = equirect_to_perspective(equirect, yaw, pitch, args.fov, args.crop_size)
        path = str(tmp_dir / f"crop_{i:03d}_y{int(yaw):03d}_p{int(pitch):+03d}.jpg")
        crop = cv2.flip(crop, 0)  # flip vertically to match video frame orientation
        cv2.imwrite(path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        image_paths.append(path)
    print(f"Saved {len(image_paths)} crops to {tmp_dir}")

    # ── 2. Load model ────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"Device: {device}  dtype: {dtype}")

    print(f"Loading checkpoint from {args.checkpoint} ...")
    model = VGGT()
    state = torch.load(args.checkpoint, map_location="cpu")
    if "model" in state:
        state = state["model"]
    elif "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.point_head = None
    model.track_head = None
    model = model.to(dtype).eval().to(device)

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

    model.forward = types.MethodType(_forward_fp16, model)

    _orig = _vggt_utils.make_sincos_pos_embed
    def _patched(embed_dim, pos, omega_0=100):
        return _orig(embed_dim, pos, omega_0=omega_0).to(pos.dtype)
    _vggt_utils.make_sincos_pos_embed = _patched
    _dpt_head_mod.make_sincos_pos_embed = _patched
    print("Model loaded.")

    # ── 3. Inference ─────────────────────────────────────────────────────────
    print("Preprocessing crops...")
    images = load_and_preprocess_images(image_paths, mode="pad").to(dtype).to(device)
    print(f"Input tensor: {images.shape}")

    print("Running VGGT inference...")
    with torch.no_grad():
        predictions = model(images)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )

    extrinsic_np = extrinsic.squeeze(0).cpu().float().numpy()
    intrinsic_np = intrinsic.squeeze(0).cpu().float().numpy()
    depth_map = predictions["depth"].squeeze(0).cpu().float().numpy()       # (S, H, W, 1)
    depth_conf = predictions["depth_conf"].squeeze(0).cpu().float().numpy() # (S, H, W)

    # ── 4. Unproject to 3D (identical to vggt_video_to_glb.py) ──────────────
    print("Unprojecting depth to 3D points...")
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic_np, intrinsic_np)  # (S, H, W, 3)

    imgs_np = predictions["images"].squeeze(0).cpu().float().numpy()  # (S, 3, H, W)
    colors = (imgs_np.transpose(0, 2, 3, 1) * 255).clip(0, 255).astype(np.uint8)  # (S, H, W, 3)

    # ── 5. Confidence filter ─────────────────────────────────────────────────
    conf_flat = depth_conf.reshape(-1)
    thresh = np.percentile(conf_flat, args.conf_percentile)
    mask = (depth_conf >= thresh) & (depth_conf > 0.1)

    pts = points_3d[mask]
    clrs = colors[mask]
    print(f"Point cloud: {len(pts):,} points after {args.conf_percentile}% confidence filter")

    # ── 6. Voxel downsample + SOR ────────────────────────────────────────────
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
    clrs_clean = (np.asarray(pcd.colors) * 255).clip(0, 255).astype(np.uint8)

    # ── 7. Export GLB ────────────────────────────────────────────────────────
    print(f"Exporting GLB to {args.output} ...")
    pc = trimesh.PointCloud(vertices=pts_clean, colors=clrs_clean)
    pc.export(args.output)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"Done! {args.output}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
