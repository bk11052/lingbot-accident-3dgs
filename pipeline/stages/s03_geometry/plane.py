"""지면 평면 RANSAC 피팅 (순수 numpy).

3DGS 레포 06_scale/scale/align.py 의 fit_ground_plane / orient_normal_toward_cameras
를 open3d 의존 없이 추출한 것. lingbot world points 에서 지배적 평면(≈도로)을 찾는다.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class ScaleAlignmentError(Exception):
    """지면 정렬/스케일 보정 단계의 복구 불가 실패."""


def fit_ground_plane(
    points: np.ndarray,
    ransac_iters: int = 1000,
    inlier_thresh: float | None = None,
    max_points: int = 50000,
) -> tuple[np.ndarray, float]:
    """RANSAC 평면 피팅. (normal, d) 반환 — normal·x + d = 0.

    inlier_thresh 가 None 이면 장면 크기(90퍼센타일 지름)의 1%로 자동 설정.
    """
    n = len(points)
    if n < 10:
        raise ScaleAlignmentError(f"Too few road points ({n}) for plane fit.")

    if inlier_thresh is None:
        centroid = np.median(points, axis=0)
        dists_to_centroid = np.linalg.norm(points - centroid, axis=1)
        scene_extent = float(np.percentile(dists_to_centroid, 90)) * 2
        inlier_thresh = 0.01 * scene_extent
        logger.info("  ground plane RANSAC: scene_extent=%.4f, inlier_thresh=%.6f",
                    scene_extent, inlier_thresh)

    if n > max_points:
        idx_sub = np.random.default_rng(0).choice(n, size=max_points, replace=False)
        points = points[idx_sub]
        n = max_points
        logger.info("  subsampled road points to %d for RANSAC", n)

    rng = np.random.default_rng(42)
    best_count = 0
    best_normal = np.array([0.0, 1.0, 0.0])
    best_d = 0.0

    for _ in range(ransac_iters):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        d = -normal @ p0
        dists = np.abs(points @ normal + d)
        count = int(np.sum(dists < inlier_thresh))
        if count > best_count:
            best_count = count
            best_normal = normal
            best_d = d

    # SVD 재피팅 (inlier 기준).
    dists = np.abs(points @ best_normal + best_d)
    inlier_mask = dists < inlier_thresh
    inlier_pts = points[inlier_mask]
    if len(inlier_pts) >= 3:
        centroid = inlier_pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(inlier_pts - centroid)
        best_normal = Vt[-1]
        best_d = -best_normal @ centroid

    logger.info("Ground plane: normal=[%.3f,%.3f,%.3f] d=%.3f (%d/%d inliers, thresh=%.6f)",
                *best_normal, best_d, int(inlier_mask.sum()), n, inlier_thresh)
    return best_normal, best_d


def orient_normal_toward_cameras(
    normal: np.ndarray,
    plane_d: float,
    poses_3x4: np.ndarray,
) -> tuple[np.ndarray, float]:
    """평면 법선이 지면→카메라 방향을 향하도록 부호 정렬."""
    cam_mean = poses_3x4[:, :3, 3].mean(axis=0)
    if normal @ cam_mean + plane_d < 0:
        normal = -normal
        plane_d = -plane_d
    return normal, plane_d
