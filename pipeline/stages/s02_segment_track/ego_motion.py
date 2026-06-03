"""Ego-motion compensation via dense optical flow on background pixels."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _build_background_mask(frame_detections, h, w):
    """boolean mask, True = background (no detected objects)."""
    bg_mask = np.ones((h, w), dtype=bool)
    for det in frame_detections.detections:
        x1, y1, x2, y2 = det.bbox
        crop_h, crop_w = det.mask_crop.shape
        actual_h = min(crop_h, y2 - y1, h - y1)
        actual_w = min(crop_w, x2 - x1, w - x1)
        bg_mask[y1:y1 + actual_h, x1:x1 + actual_w] &= ~det.mask_crop[:actual_h, :actual_w]
    return bg_mask


def compute_ego_flow(frame_paths, all_frame_detections):
    """Per-frame ego-motion = median optical flow of background pixels.

    Returns list len(frame_paths); index 0 is None; rest are (dx, dy) arrays.
    """
    ego_flows = [None]
    prev_gray = cv2.imread(str(frame_paths[0]), cv2.IMREAD_GRAYSCALE)
    if prev_gray is None:
        logger.error("Failed to read first frame: %s", frame_paths[0])
        return [None] * len(frame_paths)

    for i in range(1, len(frame_paths)):
        curr_gray = cv2.imread(str(frame_paths[i]), cv2.IMREAD_GRAYSCALE)
        if curr_gray is None:
            logger.warning("Failed to read frame %d: %s", i, frame_paths[i])
            ego_flows.append(None)
            continue

        h, w = curr_gray.shape
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0)

        bg_mask = _build_background_mask(all_frame_detections[i - 1], h, w)
        bg_pixels = bg_mask.ravel()
        flow_x = flow[:, :, 0].ravel()[bg_pixels]
        flow_y = flow[:, :, 1].ravel()[bg_pixels]
        if len(flow_x) < 100:
            logger.warning("Frame %d: only %d bg pixels, using full-image flow", i, len(flow_x))
            flow_x = flow[:, :, 0].ravel()
            flow_y = flow[:, :, 1].ravel()

        ego_flows.append(np.array([float(np.median(flow_x)), float(np.median(flow_y))]))
        prev_gray = curr_gray
        if (i + 1) % 50 == 0 or i == len(frame_paths) - 1:
            logger.info("Ego-motion progress: %d/%d frames", i + 1, len(frame_paths))

    return ego_flows


def compute_object_motion(detection, prev_detection, ego_flow):
    """Ego-compensated motion magnitude (px) for a single detection."""
    if prev_detection is None or ego_flow is None:
        return 0.0
    dx = detection.centroid[0] - prev_detection.centroid[0]
    dy = detection.centroid[1] - prev_detection.centroid[1]
    return float(np.sqrt((dx - ego_flow[0]) ** 2 + (dy - ego_flow[1]) ** 2))
