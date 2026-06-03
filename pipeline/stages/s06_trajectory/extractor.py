"""Per-frame ROI / depth-percentile sampling for the trajectory stage."""

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def build_frame_index(images_colmap, registered_frames):
    """frame_idx ↔ filename ↔ pose_idx 매핑.

    s02는 sorted(glob("*.jpg"))의 enumerate 인덱스로 frame_idx를 매겼다(detect.py).
    여기서도 동일하게 맞춰 bbox_sequence의 frame_idx가 같은 파일명으로 해석되게 한다.
    """
    sorted_paths = sorted(Path(images_colmap).glob("*.jpg"))
    sorted_filenames = [p.name for p in sorted_paths]
    fname_to_pose_idx = {name: i for i, name in enumerate(registered_frames)}
    return sorted_filenames, fname_to_pose_idx


def build_per_frame_track_bboxes(bbox_sequence, target_ids):
    """frame_idx → [(track_id, bbox), ...] (target track만)."""
    out = defaultdict(list)
    for tid, tinfo in bbox_sequence.get("tracks", {}).items():
        if tid not in target_ids:
            continue
        for f_str, fdata in tinfo.get("frames", {}).items():
            out[int(f_str)].append((tid, tuple(int(v) for v in fdata["bbox"])))
    return out


def _bboxes_overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def sample_track_frame(mask_img, depth_map, bbox, other_bboxes, *,
                       lower_frac=0.40, pct_low=20.0, pct_high=30.0,
                       min_roi_pixels=20, min_pct_pixels=5):
    """단일 track ROI에서 robust (u, v, depth) 샘플.

    ROI = bbox ∩ (mask==255) ∩ bbox 하위 lower_frac ∩ NOT(다른 bbox). 너무 작으면 None.
    """
    H, W = depth_map.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(W, int(x1))); y1 = max(0, min(H, int(y1)))
    x2 = max(0, min(W, int(x2))); y2 = max(0, min(H, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return None

    bh = y2 - y1
    y_lower = y1 + int(round((1.0 - lower_frac) * bh))
    if y_lower >= y2:
        return None

    mask_crop = mask_img[y_lower:y2, x1:x2] == 255
    if not mask_crop.any():
        return None

    if other_bboxes:
        excl = np.zeros_like(mask_crop, dtype=bool)
        for ob in other_bboxes:
            ox1, oy1, ox2, oy2 = ob
            ox1 = max(x1, min(x2, int(ox1))); oy1 = max(y_lower, min(y2, int(oy1)))
            ox2 = max(x1, min(x2, int(ox2))); oy2 = max(y_lower, min(y2, int(oy2)))
            if ox2 > ox1 and oy2 > oy1:
                excl[oy1 - y_lower:oy2 - y_lower, ox1 - x1:ox2 - x1] = True
        mask_crop &= ~excl

    if int(mask_crop.sum()) < min_roi_pixels:
        return None

    depth_crop = depth_map[y_lower:y2, x1:x2]
    vs_local, us_local = np.nonzero(mask_crop)
    depths = depth_crop[vs_local, us_local]
    finite = np.isfinite(depths) & (depths > 0)
    if int(finite.sum()) < min_roi_pixels:
        return None
    vs_local, us_local, depths = vs_local[finite], us_local[finite], depths[finite]

    p_lo, p_hi = np.percentile(depths, [pct_low, pct_high])
    keep = (depths >= p_lo) & (depths <= p_hi)
    if int(keep.sum()) < min_pct_pixels:
        return None

    u_med = float(np.median(us_local[keep])) + x1
    v_med = float(np.median(vs_local[keep])) + y_lower
    d_med = float(np.median(depths[keep]))
    return u_med, v_med, d_med


def collect_other_bboxes(frame_track_bboxes, current_tid, current_bbox):
    """이 프레임에서 current_bbox와 겹치는 다른 target track의 bbox들."""
    out = []
    for tid, bb in frame_track_bboxes:
        if tid == current_tid:
            continue
        if _bboxes_overlap(bb, current_bbox):
            out.append(bb)
    return out
