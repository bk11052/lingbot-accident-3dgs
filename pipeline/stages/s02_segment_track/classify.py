"""Dynamic/static classification with state back-propagation."""

import logging
from collections import defaultdict

from .ego_motion import compute_object_motion

logger = logging.getLogger(__name__)


def classify_tracks(all_frame_detections, ego_flows, motion_threshold_px=3.0):
    """Classify each track_id 'dynamic' | 'static'.

    State back-propagation: if ANY frame shows ego-compensated motion >
    threshold, the WHOLE track becomes 'dynamic'. Untracked (id == -1) → dynamic.
    """
    track_history = defaultdict(list)
    for fd in all_frame_detections:
        for det in fd.detections:
            if det.track_id == -1:
                continue
            track_history[det.track_id].append((fd.frame_idx, det))

    track_states = {}
    for track_id, history in track_history.items():
        history.sort(key=lambda x: x[0])
        is_dynamic = False
        for j in range(1, len(history)):
            prev_idx, prev_det = history[j - 1]
            curr_idx, curr_det = history[j]
            frame_gap = curr_idx - prev_idx
            if frame_gap > 5:
                continue
            ego_flow = ego_flows[curr_idx] if curr_idx < len(ego_flows) else None
            motion = compute_object_motion(curr_det, prev_det, ego_flow)
            if frame_gap > 1:
                motion = motion / frame_gap
            if motion > motion_threshold_px:
                is_dynamic = True
                break
        track_states[track_id] = "dynamic" if is_dynamic else "static"

    n_dyn = sum(1 for s in track_states.values() if s == "dynamic")
    n_stat = sum(1 for s in track_states.values() if s == "static")
    logger.info("Track classification: %d dynamic, %d static (thresh=%.1f px)",
                n_dyn, n_stat, motion_threshold_px)
    return track_states
