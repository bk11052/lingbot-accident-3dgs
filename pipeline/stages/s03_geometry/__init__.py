"""Stage s03: lingbot-map 피드포워드 지오메트리 + 중력/메트릭 정렬. ★핵심

기존 3DGS 파이프라인의 COLMAP(포즈) → DepthAnything(depth) → scale align 체인을
lingbot-map 한 번의 추론으로 대체한다.

Reads:
    context["artifacts"]["images_colmap"]   — s01 프레임 디렉토리(.jpg)
    context["config"]["geometry"]            — 추론/정렬 설정
    env LINGBOT_MODEL_PATH                    — model_path 미설정 시 fallback

Writes (out_root/s03_geometry/):
    poses.npy (S,4,4 aligned c2w), intrinsics.json, registered_frames.json,
    depth_maps/ (proc res), scaled_depth_maps/ (orig res, 메트릭), scaled_depth_vis/,
    sparse.ply, sparse/0/{cameras,images,points3D}.txt, images_3dgs/, align_info.json
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import traceback
from pathlib import Path

import cv2
import numpy as np

from .colmap_writer import (write_cameras_txt, write_images_txt,
                            write_points3D_txt, write_sparse_ply)
from .ground_align import (apply_sim3_to_extrinsics, apply_sim3_to_points,
                           compute_align_transform)
from .inference import run_inference

logger = logging.getLogger(__name__)

MAX_INIT_POINTS = 500_000   # 3DGS 초기화 sparse PC 상한
MAX_VIS_DEPTH = 50.0


def run(context: dict) -> dict:
    try:
        return _run_impl(context)
    except Exception:
        logger.error("Stage s03 failed:\n%s", traceback.format_exc())
        raise


def _build_sparse_pointcloud(world_points, points_conf, images_proc,
                             conf_threshold, downsample, max_points=MAX_INIT_POINTS):
    if downsample > 1:
        wp = world_points[:, ::downsample, ::downsample, :]
        pc = points_conf[:, ::downsample, ::downsample]
        imgs = images_proc[:, :, ::downsample, ::downsample]
    else:
        wp, pc, imgs = world_points, points_conf, images_proc
    mask = pc > conf_threshold
    xyz = wp[mask]
    rgb = np.clip(np.transpose(imgs, (0, 2, 3, 1))[mask] * 255.0, 0, 255).astype(np.uint8)
    n = len(xyz)
    if n > max_points:
        idx = np.random.default_rng(0).choice(n, size=max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
        logger.info("Sparse PC capped: %d → %d", n, max_points)
    return xyz.astype(np.float32), rgb


def _rescale_intrinsic(K_proc, proc_hw, orig_hw):
    H_p, W_p = proc_hw
    H_o, W_o = orig_hw
    sx, sy = W_o / W_p, H_o / H_p
    K = K_proc.copy().astype(np.float32)
    K[0, 0] *= sx; K[1, 1] *= sy; K[0, 2] *= sx; K[1, 2] *= sy
    return K


def _run_impl(context: dict) -> dict:
    cfg = context.get("config", {}).get("geometry", {})
    images_dir = Path(context["artifacts"]["images_colmap"])
    out_root = Path(context["out_root"])
    workspace = out_root / "s03_geometry"
    workspace.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(images_dir.glob("*.jpg"))
    if not image_paths:
        raise RuntimeError(f"No .jpg frames in {images_dir}")
    total = len(image_paths)
    frame_names = [p.name for p in image_paths]

    sample = cv2.imread(str(image_paths[0]))
    if sample is None:
        raise RuntimeError(f"Failed to read {image_paths[0]}")
    H_orig, W_orig = sample.shape[:2]
    logger.info("Stage s03: %d frames from %s (orig %dx%d)", total, images_dir, W_orig, H_orig)

    model_path = cfg.get("model_path") or os.environ.get("LINGBOT_MODEL_PATH")
    if not model_path:
        raise RuntimeError(
            "model_path 미지정: configs/default.yaml 의 geometry.model_path 또는 "
            "환경변수 LINGBOT_MODEL_PATH 에 .pt 경로/HF repo id 를 설정하세요.")

    # ── lingbot 추론 ────────────────────────────────────────────────────
    pred = run_inference(
        image_paths, model_path,
        image_size=cfg.get("image_size", 518),
        patch_size=cfg.get("patch_size", 14),
        num_scale_frames=cfg.get("num_scale_frames", 8),
        mode=cfg.get("mode", "auto"),
        window_size=cfg.get("window_size", 64),
        overlap_size=cfg.get("overlap_size", 16),
        keyframe_interval=cfg.get("keyframe_interval"),
        camera_num_iterations=cfg.get("camera_num_iterations", 4),
        use_sdpa=cfg.get("use_sdpa"),
    )
    H_p, W_p = pred["processed_hw"]
    ext_c2w = pred["extrinsic_c2w"]      # (S,3,4)
    K_proc = pred["intrinsic"]            # (S,3,3)
    depth_proc = pred["depth"]            # (S,H_p,W_p)
    wp = pred["world_points"]             # (S,H_p,W_p,3)
    wp_conf = pred["world_points_conf"]   # (S,H_p,W_p)
    imgs_proc = pred["images_proc"]       # (S,3,H_p,W_p)
    S = ext_c2w.shape[0]

    if depth_proc is None or wp is None or wp_conf is None:
        raise RuntimeError("lingbot 출력 불완전 (depth/world_points 없음).")
    if S != total:
        raise RuntimeError(f"Frame count mismatch: {total} in vs {S} out.")

    poses_4x4 = np.zeros((S, 4, 4), dtype=np.float64)
    poses_4x4[:, :3, :4] = ext_c2w
    poses_4x4[:, 3, 3] = 1.0

    # ── 중력 정렬 + 메트릭 스케일 (Sim3) ──────────────────────────────────
    height_prior = cfg.get("camera_height_prior", {})
    target_h = height_prior.get(context.get("input_type", "video"), 1.5)
    scale = 1.0
    align_info = {}
    try:
        R, t, scale, align_info = compute_align_transform(
            wp, wp_conf, poses_4x4,
            target_height=target_h,
            metric_rescale=cfg.get("metric_rescale", True),
            conf_percentile=cfg.get("ground_conf_percentile", 75),
        )
        poses_4x4 = apply_sim3_to_extrinsics(R, t, scale, poses_4x4)
        wp = apply_sim3_to_points(R, t, scale, wp)
        logger.info("Ground align applied: %s", align_info)
    except Exception as e:   # 정렬 실패해도 스테이지는 진행
        logger.warning("Ground alignment failed (%s); raw lingbot world 사용.", e)
        align_info = {"error": str(e), "scale": 1.0}

    with open(workspace / "align_info.json", "w") as f:
        json.dump(align_info, f, indent=2)

    poses_4x4 = poses_4x4.astype(np.float32)
    poses_path = workspace / "poses.npy"
    np.save(poses_path, poses_4x4)

    # ── intrinsics (원본 해상도, 프레임 median) ──────────────────────────
    K_orig = np.stack([_rescale_intrinsic(K_proc[i], (H_p, W_p), (H_orig, W_orig))
                       for i in range(S)], axis=0)
    K_med = np.median(K_orig, axis=0)
    intrinsics_dict = {
        "model": "PINHOLE", "width": int(W_orig), "height": int(H_orig),
        "fx": float(K_med[0, 0]), "fy": float(K_med[1, 1]),
        "cx": float(K_med[0, 2]), "cy": float(K_med[1, 2]),
    }
    intrinsics_path = workspace / "intrinsics.json"
    with open(intrinsics_path, "w") as f:
        json.dump(intrinsics_dict, f, indent=2)

    # ── registered_frames (전 프레임) ───────────────────────────────────
    reg_path = workspace / "registered_frames.json"
    with open(reg_path, "w") as f:
        json.dump({"total_input": total, "registered": total, "rate": 1.0,
                   "frames": frame_names}, f, indent=2)

    # ── depth: proc + 원본해상도×scale(메트릭) ───────────────────────────
    depth_dir = workspace / "depth_maps"
    scaled_dir = workspace / "scaled_depth_maps"
    vis_dir = workspace / "scaled_depth_vis"
    for d in (depth_dir, scaled_dir, vis_dir):
        d.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(frame_names):
        stem = Path(name).stem
        d_proc = depth_proc[i].astype(np.float32)
        d_orig = cv2.resize(d_proc, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
        d_metric = (d_orig * scale).astype(np.float32)   # 월드 스케일과 일치
        np.save(depth_dir / f"{stem}.npy", d_proc)
        np.save(scaled_dir / f"{stem}.npy", d_metric)
        vis_u8 = np.clip(d_metric / MAX_VIS_DEPTH * 255, 0, 255).astype(np.uint8)
        cv2.imwrite(str(vis_dir / f"{stem}.png"), cv2.applyColorMap(vis_u8, cv2.COLORMAP_TURBO))
    logger.info("Saved %d depth maps (scale=%.4f → metric)", S, scale)

    # ── sparse PC (3DGS 초기화) ──────────────────────────────────────────
    xyz, rgb = _build_sparse_pointcloud(
        wp, wp_conf, imgs_proc,
        conf_threshold=cfg.get("init_conf_threshold", 1.5),
        downsample=cfg.get("init_downsample", 10))
    write_sparse_ply(workspace / "sparse.ply", xyz, rgb)

    # ── COLMAP sparse 모델 (3DGS 학습 subset) ────────────────────────────
    every_n = max(1, cfg.get("images_3dgs_every_n", 2))
    subset_idx = list(range(0, S, every_n))
    subset_names = [frame_names[i] for i in subset_idx]
    subset_poses = poses_4x4[subset_idx]
    sparse_model_dir = workspace / "sparse" / "0"
    write_cameras_txt(sparse_model_dir / "cameras.txt", width=W_orig, height=H_orig,
                      fx=intrinsics_dict["fx"], fy=intrinsics_dict["fy"],
                      cx=intrinsics_dict["cx"], cy=intrinsics_dict["cy"])
    write_images_txt(sparse_model_dir / "images.txt", subset_names, subset_poses)
    write_points3D_txt(sparse_model_dir / "points3D.txt", xyz, rgb)

    # ── images_3dgs (colmap 모델과 동일 subset 복사) ─────────────────────
    images_3dgs_dir = workspace / "images_3dgs"
    images_3dgs_dir.mkdir(parents=True, exist_ok=True)
    for name in subset_names:
        src = images_dir / name
        if src.exists():
            shutil.copy(src, images_3dgs_dir / name)
    logger.info("Subsampled %d frames into %s (every_n=%d)", len(subset_names), images_3dgs_dir, every_n)

    context["artifacts"].update({
        "poses": str(poses_path),
        "intrinsics": str(intrinsics_path),
        "registered_frames": str(reg_path),
        "sparse_ply": str(workspace / "sparse.ply"),
        "colmap_model_dir": str(sparse_model_dir),
        "depth_maps": str(depth_dir),
        "scaled_depth_maps": str(scaled_dir),
        "scaled_depth_vis": str(vis_dir),
        "images_3dgs": str(images_3dgs_dir),
        "align_info": str(workspace / "align_info.json"),
    })
    logger.info("Stage s03 complete: %d-pt sparse init, %d depth maps, scale=%.4f",
                len(xyz), S, scale)
    return context
