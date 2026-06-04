"""Stage s01: 프레임 추출.

입력이 영상(.mp4/.avi/.mov)이면 ffmpeg로 프레임(.jpg)으로 분해하고,
입력이 **이미 추출된 프레임 폴더**(예: Waymo extract_waymo.py 산출물)면 그 안의
이미지(jpg/jpeg/png)를 제로패딩 .jpg 로 images_colmap/ 에 정리한다.
(Waymo .tfrecord 원본 직접 디코딩은 미지원 — 폴더로 미리 추출해 입력할 것.)

Reads:  context["input_path"], context["input_type"]
Writes: context["artifacts"]["images_colmap"], ["ingest_vis"]
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import cv2
import numpy as np

from .video import extract_video_frames

logger = logging.getLogger(__name__)

__all__ = ["extract_video_frames", "ingest_frame_folder", "run"]

# 폴더 입력에서 인식할 이미지 확장자(대소문자 무관).
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def ingest_frame_folder(src_dir: Path, out_dir: Path) -> int:
    """이미 추출된 프레임 폴더의 이미지를 frame_NNNNN.jpg 로 out_dir 에 정리.

    jpg/jpeg 는 무손실 복사, 그 외 포맷은 jpg 로 변환 저장한다. 반환값은 프레임 수.
    """
    paths = sorted(p for p in src_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)
    if not paths:
        raise ValueError(
            f"프레임 폴더에 이미지가 없습니다: {src_dir} "
            f"(찾는 확장자: {', '.join(_IMAGE_EXTS)})"
        )
    for i, p in enumerate(paths):
        dst = out_dir / f"frame_{i:05d}.jpg"
        if p.suffix.lower() in (".jpg", ".jpeg"):
            shutil.copy2(p, dst)
        else:
            img = cv2.imread(str(p))
            if img is None:
                logger.warning("이미지 로드 실패, 건너뜀: %s", p)
                continue
            cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return len(list(out_dir.glob("*.jpg")))


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

    out_dir = out_root / "s01_ingest" / "images_colmap"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_path = out_root / "s01_ingest" / "sample_grid.png"

    if input_path.is_dir():
        # 이미 추출된 프레임 폴더(예: Waymo) — 변환 없이 정리.
        logger.info("Ingesting pre-extracted frame folder: %s (type=%s)",
                    input_path, input_type)
        frame_count = ingest_frame_folder(input_path, out_dir)
    else:
        fps = cfg.get("fps", 10)
        quality = cfg.get("quality", 2)
        logger.info("Extracting frames @ %dfps from %s", fps, input_path.name)
        extract_video_frames(input_path, out_dir, fps=fps, quality=quality)
        frame_count = len(list(out_dir.glob("*.jpg")))

    if frame_count == 0:
        raise RuntimeError(f"No frames available in {out_dir} — check input/ffmpeg.")
    logger.info("Prepared %d frames -> %s", frame_count, out_dir)

    _write_sample_grid(out_dir, vis_path)

    context["artifacts"]["images_colmap"] = str(out_dir)
    context["artifacts"]["ingest_vis"] = str(vis_path)
    return context
