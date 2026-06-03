"""YOLOv8-seg instance segmentation + ByteTrack multi-object tracking."""

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# COCO class IDs for target objects
TARGET_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass
class Detection:
    track_id: int  # ByteTrack-assigned ID, -1 if untracked
    class_id: int
    class_name: str
    bbox: tuple  # (x1, y1, x2, y2) in pixels
    confidence: float
    mask_crop: np.ndarray  # bool array, bbox-cropped region only
    centroid: tuple  # (cx, cy)


@dataclass
class FrameDetections:
    frame_idx: int
    frame_name: str
    height: int
    width: int
    detections: list = field(default_factory=list)


def load_model(model_name: str = "yolov8m-seg.pt"):
    """Load YOLOv8 segmentation model. Downloads weights on first call."""
    from ultralytics import YOLO

    logger.info("Loading model: %s", model_name)
    return YOLO(model_name)


def run_tracking(
    frame_paths: list,
    model_name: str = "yolov8m-seg.pt",
    conf_thresh: float = 0.3,
    iou_thresh: float = 0.5,
) -> list:
    """Run YOLOv8-seg with ByteTrack tracking on all frames (temporal order).

    Filters to TARGET_CLASSES; stores masks bbox-cropped for memory efficiency.
    Returns list of FrameDetections, one per frame.
    """
    import cv2

    model = load_model(model_name)
    all_detections = []

    for idx, frame_path in enumerate(frame_paths):
        img = cv2.imread(str(frame_path))
        if img is None:
            logger.warning("Failed to read frame: %s", frame_path)
            all_detections.append(
                FrameDetections(frame_idx=idx, frame_name=frame_path.name, height=0, width=0))
            continue

        h, w = img.shape[:2]
        fd = FrameDetections(frame_idx=idx, frame_name=frame_path.name, height=h, width=w)

        results = model.track(img, persist=True, tracker="bytetrack.yaml",
                              conf=conf_thresh, iou=iou_thresh, verbose=False)
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            all_detections.append(fd)
            continue

        boxes = result.boxes
        masks = result.masks
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            if cls_id not in TARGET_CLASSES:
                continue
            conf = float(boxes.conf[i].item())
            track_id = int(boxes.id[i].item()) if boxes.id is not None else -1

            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            if masks is not None and i < len(masks):
                full_mask = masks[i].data.cpu().numpy().squeeze()
                if full_mask.shape != (h, w):
                    full_mask = cv2.resize(full_mask.astype(np.float32), (w, h)) > 0.5
                mask_crop = full_mask[y1:y2, x1:x2].astype(bool)
            else:
                mask_crop = np.ones((y2 - y1, x2 - x1), dtype=bool)

            fd.detections.append(Detection(
                track_id=track_id, class_id=cls_id, class_name=TARGET_CLASSES[cls_id],
                bbox=(x1, y1, x2, y2), confidence=conf, mask_crop=mask_crop,
                centroid=(cx, cy)))

        all_detections.append(fd)
        if (idx + 1) % 50 == 0 or idx == len(frame_paths) - 1:
            logger.info("Detection progress: %d/%d frames, %d dets in current",
                        idx + 1, len(frame_paths), len(fd.detections))

    total = sum(len(fd.detections) for fd in all_detections)
    tids = {d.track_id for fd in all_detections for d in fd.detections if d.track_id != -1}
    logger.info("Tracking complete: %d frames, %d dets, %d unique tracks",
                len(all_detections), total, len(tids))
    return all_detections
