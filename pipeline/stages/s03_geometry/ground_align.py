"""lingbot world 프레임을 중력 정렬 + (선택) 메트릭 스케일 보정.

lingbot-map 의 예측 월드는 첫 카메라 프레임이며 대략적으로만 중력 정렬돼 있다.
배경 PC / 3DGS / 차량 목업 배치를 위해 **지면을 z=0, +z를 위로** 정렬한다.
추가로 단안 스케일 모호성을 줄이기 위해 카메라 높이 prior(예: 대시캠 1.5m)에 맞춰
월드 전체를 메트릭으로 리스케일할 수 있다(궤적 속도가 m/s 로 의미를 가짐).

변환은 Sim(3): p_new = s · R · p_old + t  (s=스케일, R=회전, t=평행이동).
- world points : p_new = s·R·p + t           (apply_sim3_to_points)
- c2w 포즈     : 회전 = R·R_c2w (직교 유지), 카메라중심 = s·R·c + t  (apply_sim3_to_extrinsics)
- depth 맵     : ×s 로 별도 스케일 (카메라 프레임 양 — 회전엔 영향 없음)
주의: 포즈에 단순히 (sR)을 곱하면 회전이 비직교가 되어 COLMAP 쿼터니언 변환이
깨진다. 그래서 회전과 스케일을 분리해 적용한다.
"""

from __future__ import annotations

import logging

import numpy as np

from .plane import ScaleAlignmentError, fit_ground_plane, orient_normal_toward_cameras

logger = logging.getLogger(__name__)

GROUND_CONF_PERCENTILE = 75   # world_points_conf 상위 사분위만 지면 후보로
MAX_GROUND_POINTS = 100_000


def _rotation_to_align_vec(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """R @ src ≈ dst 가 되는 회전 R(3x3). Rodrigues 공식, 반평행 케이스 처리."""
    a = src / max(np.linalg.norm(src), 1e-12)
    b = dst / max(np.linalg.norm(dst), 1e-12)
    v = np.cross(a, b)
    c = float(a @ b)
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = axis - (axis @ a) * a
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return np.eye(3) + 2 * (K @ K)
    K = np.array([[0, -v[2], v[1]],
                  [v[2], 0, -v[0]],
                  [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def compute_align_transform(
    world_points: np.ndarray,     # (S, H, W, 3)
    points_conf: np.ndarray,      # (S, H, W)
    ext_c2w_4x4: np.ndarray,      # (S, 4, 4)
    *,
    target_height: float,
    metric_rescale: bool,
    conf_percentile: float = GROUND_CONF_PERCENTILE,
    up_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """중력 정렬(+메트릭) Sim(3) 반환: (R(3x3), t(3,), s, info).

    적용: p_new = s·R·p + t. 지면이 z=0, 카메라가 +z 위쪽에 오도록 구성.
    metric_rescale=True 면 카메라 중앙 높이가 target_height 가 되도록 s 설정.
    """
    up = np.asarray(up_axis, dtype=np.float64)

    flat_pts = world_points.reshape(-1, 3).astype(np.float64)
    flat_conf = points_conf.reshape(-1)
    if flat_conf.size == 0:
        raise ScaleAlignmentError("Empty world_points / points_conf.")

    conf_thresh = np.percentile(flat_conf, conf_percentile)
    candidates = flat_pts[flat_conf > conf_thresh]
    if len(candidates) < 100:
        raise ScaleAlignmentError(
            f"Too few high-confidence points for ground fit ({len(candidates)}).")
    if len(candidates) > MAX_GROUND_POINTS:
        idx = np.random.default_rng(0).choice(len(candidates), MAX_GROUND_POINTS, replace=False)
        candidates = candidates[idx]
    logger.info("Ground RANSAC candidates: %d (conf > %.3f)", len(candidates), conf_thresh)

    # 1) 지면 평면 + 법선 방향(카메라 쪽).
    poses_3x4 = ext_c2w_4x4[:, :3, :4]
    normal, d = fit_ground_plane(candidates)
    normal, d = orient_normal_toward_cameras(normal, d, poses_3x4)

    # 2) 법선 → +up 회전.
    R = _rotation_to_align_vec(normal, up)

    # 3) 회전 후 경험적 지면 높이로 t 결정(부호 혼동 방지).
    rotated_candidates = candidates @ R.T
    ground_level = float(np.median(rotated_candidates @ up))   # 지면의 up 좌표
    t0 = -ground_level * up                                     # 지면을 up=0 으로

    cam_centers = poses_3x4[:, :3, 3]
    cam_up = (cam_centers @ R.T) @ up - ground_level
    h0 = float(np.median(cam_up))                              # 정렬 후 카메라 높이(스케일 전)
    if h0 < 0:
        logger.warning("Median camera height < 0 (%.3f) after align — 법선 방향 점검 필요.", h0)

    # 4) 메트릭 스케일.
    s = 1.0
    if metric_rescale and abs(h0) > 1e-6:
        s = target_height / abs(h0)
    t = s * t0

    info = {
        "plane_normal": normal.tolist(),
        "ground_level_pre": ground_level,
        "median_cam_height_pre_scale": h0,
        "median_cam_height_post_scale": float(h0 * s),
        "target_cam_height": float(target_height),
        "scale": float(s),
        "metric_rescale": bool(metric_rescale),
        "num_candidates": int(len(candidates)),
    }
    logger.info("Align: cam height %.3f → %.3f m (target %.2f, scale=%.4f)",
                h0, h0 * s, target_height, s)
    return R, t, s, info


def apply_sim3_to_points(R: np.ndarray, t: np.ndarray, s: float, pts: np.ndarray) -> np.ndarray:
    """(..., 3) world points 에 p_new = s·R·p + t 적용."""
    return s * (pts @ R.T) + t


def apply_sim3_to_extrinsics(R: np.ndarray, t: np.ndarray, s: float,
                             ext_c2w_4x4: np.ndarray) -> np.ndarray:
    """(S,4,4) c2w 에 Sim(3) 적용: 회전=R·R_c2w(직교), 중심=s·R·c+t."""
    out = np.tile(np.eye(4), (len(ext_c2w_4x4), 1, 1)).astype(np.float64)
    rots = ext_c2w_4x4[:, :3, :3]
    centers = ext_c2w_4x4[:, :3, 3]
    out[:, :3, :3] = np.einsum("ij,sjk->sik", R, rots)
    out[:, :3, 3] = s * (centers @ R.T) + t
    return out
