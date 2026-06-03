"""파이프라인 context dict 및 스테이지 간 아티팩트 복원.

각 스테이지는 ``run(context) -> context`` 계약을 따른다. context는:
    input_path  : 입력 영상 절대경로
    input_type  : "video" | "waymo"
    out_root    : 타임스탬프 출력 디렉토리
    config      : default.yaml에서 로드한 설정 dict
    artifacts   : 스테이지 간 상태 전달 dict (대부분 디스크 경로 문자열)

``restore_artifacts``는 out_root를 스캔해 이전 실행의 산출물을 artifacts에
다시 채워 넣는다 → 중간 스테이지부터 재개(resume) 가능.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# 아티팩트 키 → out_root 기준 상대경로 후보 목록(첫 번째로 존재하는 것 채택).
ARTIFACT_PATHS: dict[str, list[str]] = {
    # s01 ingest
    "images_colmap":       ["s01_ingest/images_colmap"],
    "ingest_vis":          ["s01_ingest/sample_grid.png"],
    # s02 segment + track
    "segmentation_masks":  ["s02_segment_track/masks"],
    "bbox_sequence":       ["s02_segment_track/bbox_sequence.json"],
    "seg_overlay_sample":  ["s02_segment_track/seg_overlay_sample.png"],
    # s03 geometry (lingbot)
    "poses":               ["s03_geometry/poses.npy"],
    "intrinsics":          ["s03_geometry/intrinsics.json"],
    "registered_frames":   ["s03_geometry/registered_frames.json"],
    "sparse_ply":          ["s03_geometry/sparse.ply"],
    "colmap_model_dir":    ["s03_geometry/sparse/0"],
    "depth_maps":          ["s03_geometry/depth_maps"],
    "scaled_depth_maps":   ["s03_geometry/scaled_depth_maps"],
    "scaled_depth_vis":    ["s03_geometry/scaled_depth_vis"],
    "images_3dgs":         ["s03_geometry/images_3dgs"],
    "align_info":          ["s03_geometry/align_info.json"],
    # s04 background point cloud
    "background_ply":      ["s04_background_pcd/background.ply"],
    "background_topdown":  ["s04_background_pcd/background_topdown.png"],
    # s05 3DGS training
    "gs_model_dir":        ["s05_gaussian_train/model"],
    # s06 trajectory
    "trajectories":        ["s06_trajectory/trajectories.json"],
    "trajectories_vis":    ["s06_trajectory/trajectories_topdown.png"],
    # s07 export
    "output_splat":        ["s07_export/output.splat"],
}


def make_context(
    input_path: Path,
    input_type: str,
    out_root: Path,
    config: dict | None = None,
) -> dict:
    return {
        "input_path": str(input_path),
        "input_type": input_type,
        "out_root": str(out_root),
        "config": config or {},
        "artifacts": {},
    }


def restore_artifacts(out_root: Path, context: dict) -> None:
    """out_root에서 이전 스테이지 산출물을 찾아 context['artifacts']에 채운다."""
    restored = []
    for key, rel_paths in ARTIFACT_PATHS.items():
        for rel in rel_paths:
            full = out_root / rel
            if full.exists():
                context["artifacts"][key] = str(full)
                restored.append(key)
                break
    # target_ids는 s02가 메모리로만 설정 → resume 시 기본값으로 복구.
    if "bbox_sequence" in context["artifacts"] and "target_ids" not in context["artifacts"]:
        context["artifacts"]["target_ids"] = "all_dynamic"
    if restored:
        logger.info("Restored %d artifact(s) from %s: %s", len(restored), out_root, restored)
