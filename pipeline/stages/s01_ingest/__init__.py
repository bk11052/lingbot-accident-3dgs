"""Stage s01: 프레임 추출.

영상(.mp4/.avi)을 ffmpeg로 프레임(.jpg)으로 분해해 images_colmap/ 에 저장한다.
(Waymo .tfrecord 입력은 추후 추가 — 현재는 video 입력에 집중.)

Reads:  context["input_path"], context["input_type"]
Writes: context["artifacts"]["images_colmap"], ["ingest_vis"]
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .video import extract_video_frames

logger = logging.getLogger(__name__)

__all__ = ["extract_video_frames", "run"]


def _write_sample_grid(frame_dir: Path, vis_path: Path, n: int = 9, cols: int = 3) -> None:
    """추출 프레임 중 균등 샘플 n장으로 contact-sheet PNG 저장(검수용)."""
    paths = sorted(frame_dir.glob("*.jpg"))
    if not paths:
        return
    idx = np.linspace(0, len(paths) - 1, min(n, len(paths))).round().astype(int)
    imgs = [cv2.imread(str(paths[i])) for i in idx]
    imgs = [im for im in imgs if im is not None]
    if not imgs:
        return
    h, w = imgs[0].shape[:2]
    imgs = [cv2.resize(im, (w, h)) for im in imgs]
    rows = (len(imgs) + cols - 1) // cols
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
    vis_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(vis_path), grid)


def run(context: dict) -> dict:
    input_path = Path(context["input_path"])
    input_type = context["input_type"]
    out_root = Path(context["out_root"])
    cfg = context.get("config", {}).get("ingest", {})

    if input_type != "video":
        raise ValueError(
            f"s01_ingest는 현재 video 입력만 지원합니다 (input_type={input_type}). "
            "Waymo 지원은 추후 추가 예정."
        )

    out_dir = out_root / "s01_ingest" / "images_colmap"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_path = out_root / "s01_ingest" / "sample_grid.png"

    fps = cfg.get("fps", 10)
    quality = cfg.get("quality", 2)
    logger.info("Extracting frames @ %dfps from %s", fps, input_path.name)
    extract_video_frames(input_path, out_dir, fps=fps, quality=quality)

    frame_count = len(list(out_dir.glob("*.jpg")))
    if frame_count == 0:
        raise RuntimeError(f"No frames extracted to {out_dir} — check input/ffmpeg.")
    logger.info("Extracted %d frames -> %s", frame_count, out_dir)

    _write_sample_grid(out_dir, vis_path)

    context["artifacts"]["images_colmap"] = str(out_dir)
    context["artifacts"]["ingest_vis"] = str(vis_path)
    return context
