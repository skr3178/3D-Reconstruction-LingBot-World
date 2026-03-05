#!/usr/bin/env python3
"""Remove sky points from an existing point cloud GLB using geometric + color filtering."""

import argparse
import numpy as np
import trimesh
import open3d as o3d
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input GLB file")
    parser.add_argument("--output", required=True, help="Output GLB file")
    parser.add_argument("--height_pct", type=float, default=85.0,
                        help="Remove points above this height percentile (sky is high up)")
    parser.add_argument("--sor_neighbors", type=int, default=20,
                        help="SOR after sky removal to clean up edges")
    parser.add_argument("--sor_std_ratio", type=float, default=2.0)
    args = parser.parse_args()

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"Loading {args.input} ...")
    mesh = trimesh.load(args.input)
    if isinstance(mesh, trimesh.Scene):
        geoms = list(mesh.geometry.values())
        pts = np.vstack([g.vertices for g in geoms])
        clrs = np.vstack([g.visual.vertex_colors for g in geoms])
    else:
        pts = np.array(mesh.vertices)
        clrs = np.array(mesh.visual.vertex_colors)
    print(f"Loaded {len(pts):,} points")

    # ── Geometric sky filter (Y height) ──────────────────────────────────────
    # Sky points sit near the top of the scene's Y range
    y = pts[:, 1]
    y_thresh = np.percentile(y, args.height_pct)
    not_high = y <= y_thresh
    pts = pts[not_high]
    clrs = clrs[not_high]
    print(f"After height filter (top {100-args.height_pct:.0f}% removed, Y>{y_thresh:.3f}): {len(pts):,} points")

    # ── Color-based sky filter ────────────────────────────────────────────────
    # Sky pixels tend to be: high brightness AND blue-dominant AND low saturation
    r = clrs[:, 0].astype(float)
    g = clrs[:, 1].astype(float)
    b = clrs[:, 2].astype(float)
    brightness = (r + g + b) / 3.0
    # HSV-style saturation: (max - min) / max
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    saturation = np.where(cmax > 0, (cmax - cmin) / cmax, 0.0)
    # Sky: very bright + very low saturation + blue-dominant
    # Use strict thresholds to avoid removing roads/buildings/concrete
    is_sky_color = (brightness > 185) & (saturation < 0.12) & (b >= r - 10) & (b >= g - 10)
    not_sky_color = ~is_sky_color
    pts = pts[not_sky_color]
    clrs = clrs[not_sky_color]
    print(f"After color sky filter: {len(pts):,} points (removed {not_sky_color.size - not_sky_color.sum():,})")

    # ── SOR cleanup ──────────────────────────────────────────────────────────
    if args.sor_neighbors > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(clrs[:, :3].astype(np.float64) / 255.0)
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=args.sor_neighbors, std_ratio=args.sor_std_ratio
        )
        pts = np.asarray(pcd.points).astype(np.float32)
        clrs = (np.asarray(pcd.colors) * 255).clip(0, 255).astype(np.uint8)
        print(f"After SOR: {len(pts):,} points")

    # ── Export ────────────────────────────────────────────────────────────────
    print(f"Exporting to {args.output} ...")
    pc = trimesh.PointCloud(vertices=pts, colors=clrs)
    pc.export(args.output)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"Done! {args.output}  ({size_mb:.1f} MB, {len(pts):,} points)")


if __name__ == "__main__":
    main()
