"""파이프라인 오케스트레이터.

사용 예 (A5000 서버에서):
    # 전체 실행
    python -m pipeline.run_pipeline --input clip.mp4

    # 일부 스테이지만 (지오메트리 검증)
    python -m pipeline.run_pipeline --input clip.mp4 \
        --steps s01_ingest,s02_segment_track,s03_geometry

    # 이전 출력 이어서 재개
    python -m pipeline.run_pipeline --input clip.mp4 --steps s06_trajectory \
        --out_root outputs/run_YYYYMMDD_HHMMSS

각 스테이지 모듈은 ``run(context) -> context`` 를 노출한다.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# 패키지(import pipeline.*)를 스크립트로 직접 실행해도 찾도록 repo 루트 등록.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.core.context import make_context, restore_artifacts  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_STEPS = [
    "s01_ingest",
    "s02_segment_track",
    "s03_geometry",
    "s04_background_pcd",
    "s05_gaussian_train",
    "s06_trajectory",
    "s07_export",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LingBot 기반 단안 사고장면 재구성 파이프라인")
    p.add_argument("--input", required=True, help="입력 영상 경로(.mp4/.avi) 또는 .tfrecord")
    p.add_argument("--input_type", choices=["auto", "video", "waymo"], default="auto")
    p.add_argument("--out_root", default=None,
                   help="출력 루트(기본: pipeline/outputs/run_<timestamp>)")
    p.add_argument("--steps", default=None,
                   help="콤마 구분 스테이지 목록(기본: 전체)")
    p.add_argument("--config", default=None,
                   help="YAML 설정 파일(기본: pipeline/configs/default.yaml)")
    p.add_argument("--dry_run", action="store_true", help="실행 계획만 출력")
    return p.parse_args()


def _resolve_out_root(out_root: str | None) -> Path:
    if out_root:
        return Path(out_root).expanduser().resolve()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _REPO_ROOT / "pipeline" / "outputs" / f"run_{ts}"


def _detect_input_type(input_path: Path) -> str:
    return "waymo" if input_path.suffix.lower() == ".tfrecord" else "video"


def _load_config(config_path: str | None) -> dict:
    path = Path(config_path) if config_path else (_REPO_ROOT / "pipeline" / "configs" / "default.yaml")
    if not path.exists():
        logger.warning("Config not found (%s) — using empty config.", path)
        return {}
    try:
        import yaml
    except ImportError:
        logger.warning("pyyaml not installed — using empty config.")
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _setup_logging(out_root: Path) -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    root.addHandler(console)
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "pipeline.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    input_type = args.input_type
    if input_type == "auto":
        input_type = _detect_input_type(input_path)

    out_root = _resolve_out_root(args.out_root)
    steps = ([s.strip() for s in args.steps.split(",") if s.strip()]
             if args.steps else list(DEFAULT_STEPS))

    os.makedirs(out_root, exist_ok=True)
    _setup_logging(out_root)
    config = _load_config(args.config)

    logger.info("=== Pipeline Plan ===")
    logger.info("Input: %s (%s)", input_path, input_type)
    logger.info("Output root: %s", out_root)
    logger.info("Steps: %s", steps)

    if args.dry_run:
        logger.info("Dry run only. No execution.")
        return

    context = make_context(input_path, input_type, out_root, config)
    restore_artifacts(out_root, context)

    for step in steps:
        module_path = f"pipeline.stages.{step}"
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as e:
            raise SystemExit(f"Unknown/unimplemented step '{step}': {e}")
        if not hasattr(module, "run"):
            raise NotImplementedError(f"{module_path}.run is not implemented.")
        logger.info("--- Running %s ---", step)
        context = module.run(context)

    logger.info("Pipeline complete. Outputs in %s", out_root)


if __name__ == "__main__":
    main()
