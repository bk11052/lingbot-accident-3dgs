"""Top-down (X–Z plane) trajectory visualisation using OpenCV only."""

import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _hsv_color(idx: int, total: int) -> tuple[int, int, int]:
    hue = int(180 * (idx / max(1, total)))
    hsv = np.uint8([[[hue, 220, 240]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def render_topdown(camera_xyz, tracks, out_path, *, canvas_size=1024, margin=80):
    """카메라 경로 + track별 X–Z 궤적 top-down PNG. (Y=높이 성분은 버림)"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    track_xyz = {}
    for tid, t in tracks.items():
        pts = t.get("points", [])
        if pts:
            track_xyz[tid] = np.asarray([p["xyz"] for p in pts], dtype=np.float64)

    pts_for_extent = [camera_xyz[:, [0, 2]]] if len(camera_xyz) else []
    for arr in track_xyz.values():
        pts_for_extent.append(arr[:, [0, 2]])

    if not pts_for_extent:
        canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
        cv2.putText(canvas, "no trajectories", (40, canvas_size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        cv2.imwrite(str(out_path), canvas)
        return str(out_path)

    all_xz = np.concatenate(pts_for_extent, axis=0)
    x_min, z_min = all_xz.min(axis=0)
    x_max, z_max = all_xz.max(axis=0)
    extent_x = max(x_max - x_min, 1.0)
    extent_z = max(z_max - z_min, 1.0)
    extent = max(extent_x, extent_z)
    drawable = canvas_size - 2 * margin
    scale = drawable / extent
    cx_offset = margin + (drawable - extent_x * scale) * 0.5
    cz_offset = margin + (drawable - extent_z * scale) * 0.5

    def to_canvas(x, z):
        u = int(round(cx_offset + (x - x_min) * scale))
        v = int(round(canvas_size - (cz_offset + (z - z_min) * scale)))
        return u, v

    canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)

    grid_step_m, grid_color = 5.0, (235, 235, 235)
    x_start = np.floor(x_min / grid_step_m) * grid_step_m
    while x_start <= x_max:
        u, _ = to_canvas(x_start, z_min)
        cv2.line(canvas, (u, margin), (u, canvas_size - margin), grid_color, 1)
        x_start += grid_step_m
    z_start = np.floor(z_min / grid_step_m) * grid_step_m
    while z_start <= z_max:
        _, v = to_canvas(x_min, z_start)
        cv2.line(canvas, (margin, v), (canvas_size - margin, v), grid_color, 1)
        z_start += grid_step_m

    cv2.rectangle(canvas, (margin, margin),
                  (canvas_size - margin, canvas_size - margin), (180, 180, 180), 1)

    if len(camera_xyz) >= 2:
        cam_pts = np.array([to_canvas(p[0], p[2]) for p in camera_xyz], dtype=np.int32)
        cv2.polylines(canvas, [cam_pts], False, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(canvas, tuple(cam_pts[0]), 6, (0, 0, 0), -1)
        cv2.circle(canvas, tuple(cam_pts[-1]), 5, (0, 0, 0), 2)

    sorted_tids = sorted(track_xyz.keys(), key=lambda s: int(s))
    for i, tid in enumerate(sorted_tids):
        color = _hsv_color(i, len(sorted_tids))
        pts_uv = np.array([to_canvas(p[0], p[2]) for p in track_xyz[tid]], dtype=np.int32)
        if len(pts_uv) >= 2:
            cv2.polylines(canvas, [pts_uv], False, color, 2, cv2.LINE_AA)
        cv2.circle(canvas, tuple(pts_uv[0]), 6, color, -1)
        cv2.putText(canvas, tid, (pts_uv[0, 0] + 8, pts_uv[0, 1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    title = f"Trajectories (top-down X-Z)  extent: {extent_x:.1f} x {extent_z:.1f} m"
    cv2.putText(canvas, title, (margin, margin - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)

    bar_m = 10.0
    bar_px = int(round(bar_m * scale))
    bar_y = canvas_size - margin // 2
    cv2.line(canvas, (margin, bar_y), (margin + bar_px, bar_y), (0, 0, 0), 2)
    cv2.line(canvas, (margin, bar_y - 5), (margin, bar_y + 5), (0, 0, 0), 2)
    cv2.line(canvas, (margin + bar_px, bar_y - 5), (margin + bar_px, bar_y + 5), (0, 0, 0), 2)
    cv2.putText(canvas, f"{int(bar_m)} m", (margin + bar_px // 2 - 14, bar_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), canvas)
    logger.info("Wrote top-down trajectory plot: %s", out_path)
    return str(out_path)
