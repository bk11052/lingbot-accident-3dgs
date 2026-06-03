"""Stage s06: 3D 차량 궤적 추출 (AB3DMOT 하이브리드).

각 동적 track마다 마스크 하위 40% ROI(뒷범퍼)에서 robust depth(P20–P30)를 샘플,
(u,v,depth)를 world로 역투영한다. track별 위치를 상수속도 칼만필터(AB3DMOT KF)로
평활화하고, 짧은 가림 구간은 predict-only로 보간한다. 출력: trajectories.json.

좌표계는 s03의 c2w(gravity-aligned, 메트릭)와 동일. 속도는 config의 fps로 m/s 환산.
"""

import json
import logging
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from .extractor import (build_frame_index, build_per_frame_track_bboxes,
                        collect_other_bboxes, sample_track_frame)
from .tracker import Track3DKalman
from .unproject import pixel_to_world
from .visualizer import render_topdown
from .writer import write_trajectories

logger = logging.getLogger(__name__)


def run(context):
    try:
        return _run_impl(context)
    except Exception:
        logger.error("Stage s06 failed:\n%s", traceback.format_exc())
        raise


def _resolve_target_ids(target_ids_artifact, bbox_sequence):
    tracks = bbox_sequence.get("tracks", {})
    if target_ids_artifact in ("all_dynamic", None):
        return {tid for tid, ti in tracks.items()
                if ti.get("state") == "dynamic" and tid != "-1"}
    if isinstance(target_ids_artifact, str):
        return {t.strip() for t in target_ids_artifact.split(",") if t.strip()}
    if isinstance(target_ids_artifact, (list, tuple, set)):
        return {str(t) for t in target_ids_artifact}
    raise ValueError(f"Unsupported target_ids type: {type(target_ids_artifact)}")


def _make_point(frame_idx, kf, obs, *, interpolated, fps, frame_name=None):
    x, y, z = kf.xyz()
    vx, vy, vz = kf.velocity_per_frame()
    fname = obs["frame_name"] if obs is not None else frame_name
    depth = obs["depth_m"] if obs is not None else None
    return {
        "frame_idx": frame_idx,
        "frame_name": fname,
        "xyz": [x, y, z],
        "velocity_mps": [vx * fps, vy * fps, vz * fps],
        "depth_m": depth,
        "interpolated": interpolated,
    }


def _kf_track_points(tid, obs_by_frame, sorted_filenames, *, max_predict_gap, obs_noise, fps):
    """track별 KF + 가림 구간 predict. gap > max_predict_gap이면 KF 폐기·재초기화."""
    frame_indices = sorted(obs_by_frame.keys())
    first_f, last_f = frame_indices[0], frame_indices[-1]
    kf = None
    points = []
    for f in range(first_f, last_f + 1):
        obs = obs_by_frame.get(f)
        if kf is None:
            if obs is None:
                continue
            kf = Track3DKalman(obs["xyz"], tid, observation_noise=obs_noise)
            points.append(_make_point(f, kf, obs, interpolated=False, fps=fps))
            continue
        kf.predict()
        if obs is not None:
            kf.update(obs["xyz"])
            points.append(_make_point(f, kf, obs, interpolated=False, fps=fps))
        elif kf.frames_since_update <= max_predict_gap:
            fname = sorted_filenames[f] if 0 <= f < len(sorted_filenames) else None
            points.append(_make_point(f, kf, None, interpolated=True, fps=fps, frame_name=fname))
        else:
            kf = None
    return points


def _run_impl(context):
    artifacts = context["artifacts"]
    cfg = context.get("config", {}).get("trajectory", {})

    lower_frac = cfg.get("lower_frac", 0.40)
    pct_low = cfg.get("pct_low", 20.0)
    pct_high = cfg.get("pct_high", 30.0)
    min_roi_pixels = cfg.get("min_roi_pixels", 20)
    min_pct_pixels = cfg.get("min_pct_pixels", 5)
    max_predict_gap = cfg.get("max_predict_gap", 5)
    obs_noise = cfg.get("kf_observation_noise", 1.0)
    fps = cfg.get("fps", 10)

    required = ["bbox_sequence", "segmentation_masks", "images_colmap",
                "scaled_depth_maps", "poses", "intrinsics", "registered_frames"]
    for k in required:
        if k not in artifacts:
            raise FileNotFoundError(f"Stage s06 missing required artifact: {k}")

    bbox_path = Path(artifacts["bbox_sequence"])
    mask_dir = Path(artifacts["segmentation_masks"])
    images_dir = Path(artifacts["images_colmap"])
    depth_dir = Path(artifacts["scaled_depth_maps"])
    poses_path = Path(artifacts["poses"])
    intrinsics_path = Path(artifacts["intrinsics"])
    reg_path = Path(artifacts["registered_frames"])

    out_root = Path(context["out_root"])
    workspace = out_root / "s06_trajectory"
    workspace.mkdir(parents=True, exist_ok=True)
    out_json = workspace / "trajectories.json"

    logger.info("Step 1/4: Loading bbox_sequence, poses, intrinsics, registered_frames...")
    with open(bbox_path) as f:
        bbox_sequence = json.load(f)
    poses = np.load(poses_path)
    with open(intrinsics_path) as f:
        intrinsics = json.load(f)
    with open(reg_path) as f:
        reg_frames = json.load(f)["frames"]
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]

    target_set = _resolve_target_ids(artifacts.get("target_ids"), bbox_sequence)
    if not target_set:
        logger.warning("Stage s06: no target tracks — writing empty trajectories.")

    logger.info("Step 2/4: Indexing %d target tracks across %d frames...",
                len(target_set), len(reg_frames))
    sorted_filenames, fname_to_pose_idx = build_frame_index(images_dir, reg_frames)
    work_by_frame = build_per_frame_track_bboxes(bbox_sequence, target_set)

    logger.info("Step 3/4: Sampling depth percentiles per (track, frame) — %d frames...",
                len(work_by_frame))
    raw_obs = defaultdict(dict)
    skipped = Counter()
    processed_frames = 0
    for frame_idx in sorted(work_by_frame):
        frame_tracks = work_by_frame[frame_idx]
        n_in_frame = len(frame_tracks)
        if frame_idx >= len(sorted_filenames):
            skipped["frame_idx_out_of_range"] += n_in_frame
            continue
        fname = sorted_filenames[frame_idx]
        if fname not in fname_to_pose_idx:
            skipped["unregistered_frame"] += n_in_frame
            continue
        c2w = poses[fname_to_pose_idx[fname]]
        stem = Path(fname).stem
        mask_path = mask_dir / f"{stem}.png"
        depth_path = depth_dir / f"{stem}.npy"
        if not mask_path.exists() or not depth_path.exists():
            skipped["mask_or_depth_missing"] += n_in_frame
            continue
        mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            skipped["mask_or_depth_missing"] += n_in_frame
            continue
        depth_map = np.load(depth_path)

        for tid, bbox in frame_tracks:
            others = collect_other_bboxes(frame_tracks, tid, bbox)
            sample = sample_track_frame(
                mask_img, depth_map, bbox, others,
                lower_frac=lower_frac, pct_low=pct_low, pct_high=pct_high,
                min_roi_pixels=min_roi_pixels, min_pct_pixels=min_pct_pixels)
            if sample is None:
                skipped["empty_or_low_roi"] += 1
                continue
            u, v, d = sample
            x, y, z = pixel_to_world(u, v, d, c2w, fx, fy, cx, cy)
            raw_obs[tid][frame_idx] = {"frame_name": fname, "xyz": (x, y, z), "depth_m": d}

        processed_frames += 1
        if processed_frames % 50 == 0:
            logger.info("  processed %d / %d frames", processed_frames, len(work_by_frame))

    logger.info("Step 4/4: Kalman filter (AB3DMOT KF) on %d tracks (max_predict_gap=%d)...",
                len(raw_obs), max_predict_gap)
    tracks_out = {}
    interpolated_count = 0
    for tid in sorted(raw_obs.keys(), key=lambda s: int(s)):
        points = _kf_track_points(tid, raw_obs[tid], sorted_filenames,
                                  max_predict_gap=max_predict_gap, obs_noise=obs_noise, fps=fps)
        if not points:
            continue
        interpolated_count += sum(1 for p in points if p["interpolated"])
        tracks_out[tid] = {
            "class_name": bbox_sequence["tracks"][tid].get("class_name", "unknown"),
            "points": points,
        }

    params = {
        "tracker": "AB3DMOT-KF (hybrid: ByteTrack id + lingbot depth)",
        "lower_frac": lower_frac, "pct_low": pct_low, "pct_high": pct_high,
        "min_roi_pixels": min_roi_pixels, "min_pct_pixels": min_pct_pixels,
        "max_predict_gap": max_predict_gap, "kf_observation_noise": obs_noise, "fps": fps,
    }
    write_trajectories(tracks_out, frame_count_total=len(sorted_filenames),
                       skipped=dict(skipped), out_path=out_json, params=params)
    artifacts["trajectories"] = str(out_json)

    vis_path = workspace / "trajectories_topdown.png"
    render_topdown(poses[:, :3, 3], tracks_out, vis_path)
    artifacts["trajectories_vis"] = str(vis_path)

    total_points = sum(len(t["points"]) for t in tracks_out.values())
    logger.info("Stage s06 complete: %d tracks, %d points (%d interp), skipped=%s -> %s",
                len(tracks_out), total_points, interpolated_count, dict(skipped), out_json)
    return context
