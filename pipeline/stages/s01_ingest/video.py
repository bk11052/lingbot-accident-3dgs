"""ffmpeg 기반 영상 → 프레임(.jpg) 추출."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Union


def extract_video_frames(
    video_path: Union[str, os.PathLike],
    out_dir: Union[str, os.PathLike],
    fps: int = 10,
    quality: int = 2,
    pattern: str = "frame_%06d.jpg",
) -> str:
    """*video_path* 를 *fps* 로 샘플링해 *out_dir* 에 JPEG 프레임으로 저장."""
    video_path = Path(video_path).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH.")

    out_pattern = str(out_dir / pattern)
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", str(quality),
        out_pattern,
    ]
    subprocess.run(cmd, check=True)
    return str(out_dir)
