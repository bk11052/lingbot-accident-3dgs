"""Stage s07: 최종 산출물 export (프론트엔드 인터페이스).

배경 3DGS(output.ply) → output.splat 변환, 궤적(trajectories.json) 패스스루.
이 두 파일이 웹 뷰어가 소비하는 출력 계약이다:
    <out>/s07_export/output.splat        — 배경 가우시안 스플랫
    <out>/s07_export/trajectories.json   — 동적 차량 월드 궤적(메트릭, 지면 z≈0)

기본은 내장 ply_to_splat 변환기를 쓴다. 외부 변환기를 쓰려면
export.splat_converter_cmd 에 {ply}/{splat} 플레이스홀더 명령을 지정한다.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import traceback
from pathlib import Path

from .splat import ply_to_splat

logger = logging.getLogger(__name__)


def run(context):
    try:
        return _run_impl(context)
    except Exception:
        logger.error("Stage s07 failed:\n%s", traceback.format_exc())
        raise


def _run_impl(context):
    artifacts = context["artifacts"]
    cfg = context.get("config", {}).get("export", {})
    out_root = Path(context["out_root"])
    workspace = out_root / "s07_export"
    workspace.mkdir(parents=True, exist_ok=True)

    if "output_ply" not in artifacts:
        raise FileNotFoundError("Stage s07 requires 'output_ply' (s05 3DGS 결과).")
    ply_path = Path(artifacts["output_ply"])
    splat_path = workspace / "output.splat"

    converter = cfg.get("splat_converter_cmd")
    if converter:
        cmd = converter.format(ply=ply_path, splat=splat_path)
        logger.info("External splat converter: %s", cmd)
        subprocess.run(cmd.split() if isinstance(cmd, str) else cmd, check=True)
    else:
        ply_to_splat(ply_path, splat_path)
    artifacts["output_splat"] = str(splat_path)

    # 궤적 패스스루 (출력 디렉토리로 복사해 프론트가 한 곳에서 읽도록).
    if "trajectories" in artifacts and Path(artifacts["trajectories"]).exists():
        dst = workspace / "trajectories.json"
        shutil.copy(artifacts["trajectories"], dst)
        artifacts["output_trajectories"] = str(dst)
        logger.info("Trajectories → %s", dst)
    else:
        logger.warning("trajectories 산출물 없음 — s06 먼저 실행 필요.")

    logger.info("Stage s07 complete: %s", splat_path)
    return context
