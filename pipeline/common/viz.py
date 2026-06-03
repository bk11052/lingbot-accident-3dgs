"""검수용 시각화 헬퍼 (contact-sheet 그리드)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def sample_evenly(seq: list, n: int) -> list:
    """seq 에서 균등 간격으로 최대 n개 추출."""
    if not seq:
        return []
    if len(seq) <= n:
        return list(seq)
    idx = np.linspace(0, len(seq) - 1, n).round().astype(int)
    return [seq[i] for i in idx]


def render_frame_grid(images: list[np.ndarray], out_path, cols: int = 3,
                      labels: list[str] | None = None) -> str:
    """이미지 리스트를 cols 열 그리드로 합쳐 PNG 저장. 저장 경로 반환."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not images:
        return str(out_path)
    h, w = images[0].shape[:2]
    norm = [cv2.resize(im, (w, h)) for im in images]
    if labels:
        for im, lab in zip(norm, labels):
            cv2.putText(im, str(lab), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2, cv2.LINE_AA)
    rows = (len(norm) + cols - 1) // cols
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, im in enumerate(norm):
        r, c = divmod(i, cols)
        grid[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
    cv2.imwrite(str(out_path), grid)
    return str(out_path)
