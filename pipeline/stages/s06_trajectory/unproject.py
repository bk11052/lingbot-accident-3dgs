"""Single-pixel pinhole unprojection: (u, v, depth_m) → world (x, y, z)."""

import numpy as np


def pixel_to_world(u, v, depth, c2w, fx, fy, cx, cy):
    """픽셀(u,v)+메트릭 depth → world. c2w는 (4,4) camera-to-world."""
    x_cam = (u - cx) / fx * depth
    y_cam = (v - cy) / fy * depth
    pt_cam = np.array([x_cam, y_cam, depth, 1.0], dtype=np.float64)
    pt_world = c2w @ pt_cam
    return float(pt_world[0]), float(pt_world[1]), float(pt_world[2])
