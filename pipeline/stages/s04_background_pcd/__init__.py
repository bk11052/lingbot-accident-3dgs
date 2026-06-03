"""Stage s04: 배경 포인트클라우드 융합.

s03의 메트릭 depth + 정렬된 c2w 포즈로 픽셀을 world에 역투영해 누적하되,
**동적 마스크(255) 픽셀은 제외**해 움직이는 차량이 배경에 번지는 것을 막는다.
이후 voxel downsample + Open3D 통계적 이상치 제거(SOR)로 정리한다.

Reads:  scaled_depth_maps, poses, intrinsics, registered_frames,
        segmentation_masks, images_colmap
Writes: background_ply (s04_background_pcd/background.ply), background_topdown
"""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MAX_DEPTH = 80.0   # m, 메트릭 가정. 먼 하늘/노이즈 컷.


def run(context):
    try:
        return _run_impl(context)
    except Exception:
        logger.error("Stage s04 failed:\n%s", traceback.format_exc())
        raise


def _backproject_frame(depth, K, c2w, image_rgb, dyn_mask, *, pixel_stride, max_depth):
    """단일 프레임 → (world_pts (N,3), colors (N,3) uint8). 동적/무효 픽셀 제외."""
    H, W = depth.shape
    fx, fy = K["fx"], K["fy"]
    cx, cy = K["cx"], K["cy"]

    d = depth[::pixel_stride, ::pixel_stride]
    img = image_rgb[::pixel_stride, ::pixel_stride]
    dyn = dyn_mask[::pixel_stride, ::pixel_stride] if dyn_mask is not None else None
    vs, us = np.meshgrid(np.arange(0, H, pixel_stride, dtype=np.float32),
                         np.arange(0, W, pixel_stride, dtype=np.float32), indexing="ij")

    valid = np.isfinite(d) & (d > 0) & (d < max_depth)
    if dyn is not None:
        valid &= (dyn[:d.shape[0], :d.shape[1]] != 255)
    if not valid.any():
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)

    d, us, vs, img = d[valid], us[valid], vs[valid], img[valid]
    cam = np.stack([(us - cx) / fx * d, (vs - cy) / fy * d, d], axis=-1)  # (N,3)
    world = cam @ c2w[:3, :3].T + c2w[:3, 3]
    return world.astype(np.float32), img.astype(np.uint8)


def _render_topdown(points, out_path, *, canvas=1024, margin=60):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((canvas, canvas, 3), 255, np.uint8)
    if len(points) == 0:
        cv2.imwrite(str(out_path), img)
        return
    xz = points[:, [0, 2]]
    lo, hi = np.percentile(xz, 1, axis=0), np.percentile(xz, 99, axis=0)
    ext = np.maximum(hi - lo, 1e-3)
    s = (canvas - 2 * margin) / ext.max()
    uv = ((xz - lo) * s + margin).astype(np.int32)
    uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < canvas) & (uv[:, 1] >= 0) & (uv[:, 1] < canvas)]
    img[canvas - 1 - uv[:, 1], uv[:, 0]] = (60, 60, 60)
    cv2.imwrite(str(out_path), img)


def _run_impl(context):
    artifacts = context["artifacts"]
    cfg = context.get("config", {}).get("background_pcd", {})
    out_root = Path(context["out_root"])
    workspace = out_root / "s04_background_pcd"
    workspace.mkdir(parents=True, exist_ok=True)

    import open3d as o3d

    depth_dir = Path(artifacts["scaled_depth_maps"])
    images_dir = Path(artifacts["images_colmap"])
    mask_dir = Path(artifacts.get("segmentation_masks", ""))
    poses = np.load(artifacts["poses"])
    with open(artifacts["intrinsics"]) as f:
        K = json.load(f)
    with open(artifacts["registered_frames"]) as f:
        reg_frames = json.load(f)["frames"]

    pixel_stride = cfg.get("pixel_stride", 4)
    max_depth = cfg.get("max_depth", DEFAULT_MAX_DEPTH)
    max_points = cfg.get("max_points", 3_000_000)

    all_pts, all_cols = [], []
    for i, fname in enumerate(reg_frames):
        if i >= len(poses):
            break
        stem = Path(fname).stem
        depth_path = depth_dir / f"{stem}.npy"
        img_path = images_dir / fname
        if not depth_path.exists() or not img_path.exists():
            continue
        depth = np.load(depth_path)
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != depth.shape[:2]:
            rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]))
        dyn = None
        mpath = mask_dir / f"{stem}.png"
        if mask_dir and mpath.exists():
            dyn = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
            if dyn is not None and dyn.shape[:2] != depth.shape[:2]:
                dyn = cv2.resize(dyn, (depth.shape[1], depth.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
        pts, cols = _backproject_frame(depth, K, poses[i], rgb, dyn,
                                       pixel_stride=pixel_stride, max_depth=max_depth)
        if len(pts):
            all_pts.append(pts)
            all_cols.append(cols)
        if (i + 1) % 50 == 0:
            logger.info("  fused %d / %d frames", i + 1, len(reg_frames))

    if not all_pts:
        raise RuntimeError("배경 융합 결과 0 points — depth/mask/poses 확인 필요.")
    pts = np.concatenate(all_pts, axis=0)
    cols = np.concatenate(all_cols, axis=0)
    logger.info("Fused raw points: %d", len(pts))

    if len(pts) > max_points:
        idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts, cols = pts[idx], cols[idx]
        logger.info("Capped to %d points", max_points)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)

    voxel = cfg.get("voxel_size", 0.05)
    if voxel and voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
        logger.info("After voxel(%.3f): %d points", voxel, len(pcd.points))
    nb = cfg.get("sor_nb_neighbors", 20)
    std = cfg.get("sor_std_ratio", 2.0)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std)
    logger.info("After SOR(nb=%d, std=%.1f): %d points", nb, std, len(pcd.points))

    out_ply = workspace / "background.ply"
    o3d.io.write_point_cloud(str(out_ply), pcd)
    artifacts["background_ply"] = str(out_ply)

    vis_path = workspace / "background_topdown.png"
    _render_topdown(np.asarray(pcd.points), vis_path)
    artifacts["background_topdown"] = str(vis_path)

    logger.info("Stage s04 complete: %d background points -> %s", len(pcd.points), out_ply)
    return context
