"""Stage s05: 3DGS 학습 (배경).

s03의 lingbot 포즈 + s04의 동적-제외 배경 포인트클라우드로 COLMAP 형식 scene을
조립하고, gaussian-splatting을 **별도 conda 환경**(diff rasterizer 빌드 포함)에서
서브프로세스로 학습한다. 산출물은 디스크로 주고받는다.

조립되는 scene:
    <ws>/scene/images/            ← s03 images_3dgs 프레임
    <ws>/scene/sparse/0/          ← cameras.txt/images.txt(s03) + points3D.txt(s04 배경)
    <ws>/scene/masks/             ← (선택) 동적 마스크 — mask-aware 포크가 loss 제외에 사용

기본 학습 명령(vanilla Inria gaussian-splatting):
    conda run -n {env} python {gs_root}/train.py -s {scene} -m {model} --iterations {iters}

⚠️ vanilla Inria train.py는 per-pixel 마스크 loss 제외를 기본 지원하지 않는다.
   동적 객체 loss 제외가 필요하면 mask-aware 포크를 쓰고 config의 train_cmd_template로
   명령을 덮어쓴다(아래 플레이스홀더 사용). 배경 init이 이미 동적-제외라 floater는 크게
   줄지만, 마스크 loss가 있으면 더 깨끗하다.

Reads:  colmap_model_dir, images_3dgs, background_ply (+ segmentation_masks 선택)
Writes: gs_model_dir (학습 모델 디렉토리; point_cloud/iteration_*/point_cloud.ply 포함)
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import traceback
from pathlib import Path

import numpy as np

from pipeline.stages.s03_geometry.colmap_writer import write_points3D_txt

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def run(context):
    try:
        return _run_impl(context)
    except Exception:
        logger.error("Stage s05 failed:\n%s", traceback.format_exc())
        raise


def _points3D_from_background(background_ply: Path, out_txt: Path) -> int:
    """배경 PLY → COLMAP points3D.txt (3DGS 초기화용, 동적 제외 클라우드)."""
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(background_ply))
    xyz = np.asarray(pcd.points, dtype=np.float32)
    cols = np.asarray(pcd.colors)
    rgb = (np.clip(cols, 0, 1) * 255).astype(np.uint8) if len(cols) else None
    write_points3D_txt(out_txt, xyz, rgb)
    return len(xyz)


def _assemble_scene(artifacts: dict, scene_dir: Path) -> None:
    """images/ + sparse/0/ (+ masks/) 조립."""
    images_src = Path(artifacts["images_3dgs"])
    colmap_src = Path(artifacts["colmap_model_dir"])    # s03 sparse/0
    background_ply = Path(artifacts["background_ply"])

    images_dst = scene_dir / "images"
    sparse_dst = scene_dir / "sparse" / "0"
    images_dst.mkdir(parents=True, exist_ok=True)
    sparse_dst.mkdir(parents=True, exist_ok=True)

    for p in sorted(images_src.glob("*.jpg")):
        shutil.copy(p, images_dst / p.name)
    # 포즈/카메라는 s03 것을 그대로, points3D는 s04 배경(동적 제외)으로 교체.
    shutil.copy(colmap_src / "cameras.txt", sparse_dst / "cameras.txt")
    shutil.copy(colmap_src / "images.txt", sparse_dst / "images.txt")
    n = _points3D_from_background(background_ply, sparse_dst / "points3D.txt")
    logger.info("Assembled scene: %d images, %d init points",
                len(list(images_dst.glob('*.jpg'))), n)

    # 동적 마스크(있으면) — mask-aware 포크가 쓸 수 있게 동일 stem으로 복사.
    mask_src = artifacts.get("segmentation_masks")
    if mask_src and Path(mask_src).is_dir():
        masks_dst = scene_dir / "masks"
        masks_dst.mkdir(parents=True, exist_ok=True)
        copied = 0
        for img in images_dst.glob("*.jpg"):
            m = Path(mask_src) / f"{img.stem}.png"
            if m.exists():
                shutil.copy(m, masks_dst / m.name)
                copied += 1
        logger.info("Copied %d dynamic masks into scene/masks/", copied)


def _run_impl(context):
    artifacts = context["artifacts"]
    cfg = context.get("config", {}).get("gaussian_train", {})
    out_root = Path(context["out_root"])
    workspace = out_root / "s05_gaussian_train"
    workspace.mkdir(parents=True, exist_ok=True)

    for k in ("colmap_model_dir", "images_3dgs", "background_ply"):
        if k not in artifacts:
            raise FileNotFoundError(f"Stage s05 missing required artifact: {k}")

    scene_dir = workspace / "scene"
    model_dir = workspace / "model"
    _assemble_scene(artifacts, scene_dir)

    gs_root = cfg.get("gaussian_splatting_root") or str(
        _REPO_ROOT / "third_party" / "gaussian-splatting")
    env = cfg.get("conda_env", "gs3dgs")
    iters = cfg.get("iterations", 30000)

    template = cfg.get("train_cmd_template")
    if template:
        cmd = template.format(env=env, gs_root=gs_root, scene=scene_dir,
                              model=model_dir, iters=iters)
        cmd_list = cmd.split() if isinstance(cmd, str) else cmd
    else:
        cmd_list = [
            "conda", "run", "-n", env, "python", f"{gs_root}/train.py",
            "-s", str(scene_dir), "-m", str(model_dir), "--iterations", str(iters),
        ]

    logger.info("Launching 3DGS training (env=%s, iters=%d):\n  %s",
                env, iters, " ".join(map(str, cmd_list)))
    logger.warning("vanilla train.py는 마스크 loss 제외 미지원 — 필요 시 "
                   "gaussian_train.train_cmd_template로 mask-aware 명령 지정.")
    subprocess.run(cmd_list, check=True)

    plys = sorted(model_dir.glob("point_cloud/iteration_*/point_cloud.ply"),
                  key=lambda p: int(p.parent.name.split("_")[-1]))
    if not plys:
        raise RuntimeError(f"3DGS 학습 후 point_cloud.ply 미발견: {model_dir}")
    artifacts["gs_model_dir"] = str(model_dir)
    artifacts["output_ply"] = str(plys[-1])
    logger.info("Stage s05 complete: %s", plys[-1])
    return context
