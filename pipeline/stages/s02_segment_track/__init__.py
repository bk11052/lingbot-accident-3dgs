"""Stage s02: Instance Segmentation & Multi-Object Tracking.

YOLOv8-seg로 동적 객체(차량/보행자) 검출, ByteTrack로 추적, ego-motion 보정 후
state back-propagation으로 각 track을 동/정적 분류한다. (사용자 타겟 선택은 보류 —
기본 all_dynamic.)

Reads:  context["artifacts"]["images_colmap"]
Writes: segmentation_masks, bbox_sequence, target_ids("all_dynamic"), seg_overlay_sample
"""

import logging
from pathlib import Path

from .classify import classify_tracks
from .detect import run_tracking
from .ego_motion import compute_ego_flow
from .mask_writer import write_bbox_sequence, write_masks
from .visualizer import write_seg_overlay_sample

logger = logging.getLogger(__name__)


def run(context):
    images_dir = Path(context["artifacts"]["images_colmap"])
    out_root = Path(context["out_root"])
    seg_dir = out_root / "s02_segment_track"
    masks_dir = seg_dir / "masks"

    frame_paths = sorted(images_dir.glob("*.jpg"))
    if not frame_paths:
        raise ValueError(f"No .jpg frames found in {images_dir}")
    logger.info("Stage s02 starting: %d frames from %s", len(frame_paths), images_dir)

    logger.info("Step 1/4: YOLOv8-seg + ByteTrack...")
    all_detections = run_tracking(frame_paths)

    logger.info("Step 2/4: ego-motion via optical flow...")
    ego_flows = compute_ego_flow(frame_paths, all_detections)

    logger.info("Step 3/4: classify tracks (dynamic/static)...")
    track_states = classify_tracks(all_detections, ego_flows)

    logger.info("Step 4/4: write masks + bbox sequence...")
    masks_path = write_masks(frame_paths, all_detections, track_states, masks_dir)
    bbox_path = write_bbox_sequence(all_detections, track_states, seg_dir / "bbox_sequence.json")

    overlay_path = seg_dir / "seg_overlay_sample.png"
    write_seg_overlay_sample(frame_paths, all_detections, track_states, overlay_path)

    context["artifacts"]["segmentation_masks"] = masks_path
    context["artifacts"]["bbox_sequence"] = bbox_path
    context["artifacts"]["target_ids"] = "all_dynamic"
    context["artifacts"]["seg_overlay_sample"] = str(overlay_path)

    logger.info("Stage s02 complete: masks -> %s, bbox -> %s", masks_path, bbox_path)
    return context
