"""lingbot-map 스트리밍 추론 → numpy 출력.

GCTStream.inference_streaming / inference_windowed 를 호출하고 demo.py 의
후처리를 적용해, 나머지 스테이지가 numpy 로 다룰 수 있게 한다.

반환 dict:
    extrinsic_c2w      : (S, 3, 4) camera-to-world
    intrinsic          : (S, 3, 3) lingbot 처리해상도(H_p, W_p) 기준
    depth              : (S, H_p, W_p)
    depth_conf         : (S, H_p, W_p)
    world_points       : (S, H_p, W_p, 3)
    world_points_conf  : (S, H_p, W_p)
    processed_hw       : (H_p, W_p)
    images_proc        : (S, 3, H_p, W_p) 전처리 입력 텐서(CPU, [0,1])
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_SIZE = 518
DEFAULT_PATCH_SIZE = 14
DEFAULT_NUM_SCALE_FRAMES = 8
STREAMING_MAX_FRAMES = 320   # auto-keyframe 임계 (demo.py 와 동일)


def _lazy_import_lingbot():
    try:
        from lingbot_map.utils.load_fn import load_and_preprocess_images
        from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
        from lingbot_map.utils.geometry import closed_form_inverse_se3_general
    except ImportError as e:
        raise ImportError(
            "lingbot-map 미설치. 레포 루트에서 `pip install -e .` 하세요."
        ) from e
    return load_and_preprocess_images, pose_encoding_to_extri_intri, closed_form_inverse_se3_general


def _flashinfer_available() -> bool:
    try:
        import flashinfer  # noqa: F401
        return True
    except Exception:
        return False


def _load_model(model_path, device, *, mode, use_sdpa, image_size, patch_size,
                num_scale_frames, kv_cache_sliding_window, camera_num_iterations,
                max_frame_num, enable_3d_rope=True):
    """GCTStream 빌드 + 로컬 .pt 체크포인트 로드 (demo.py:load_model 미러)."""
    if mode == "windowed":
        from lingbot_map.models.gct_stream_window import GCTStream
    else:
        from lingbot_map.models.gct_stream import GCTStream

    logger.info("Building GCTStream (mode=%s, use_sdpa=%s)", mode, use_sdpa)
    model = GCTStream(
        img_size=image_size, patch_size=patch_size,
        enable_3d_rope=enable_3d_rope, max_frame_num=max_frame_num,
        kv_cache_sliding_window=kv_cache_sliding_window,
        kv_cache_scale_frames=num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=use_sdpa, camera_num_iterations=camera_num_iterations,
    )
    logger.info("Loading checkpoint: %s", model_path)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("  %d missing keys (e.g. %s)", len(missing), missing[:3])
    if unexpected:
        logger.warning("  %d unexpected keys (e.g. %s)", len(unexpected), unexpected[:3])
    return model.to(device).eval()


def run_inference(
    image_paths: list[Path],
    model_path: str,
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
    patch_size: int = DEFAULT_PATCH_SIZE,
    num_scale_frames: int = DEFAULT_NUM_SCALE_FRAMES,
    mode: str = "auto",
    window_size: int = 64,
    overlap_size: int = 16,
    keyframe_interval: int | None = None,
    kv_cache_sliding_window: int = 64,
    camera_num_iterations: int = 4,
    max_frame_num: int = 1024,
    use_sdpa: bool | None = None,
) -> dict:
    (load_and_preprocess_images,
     pose_encoding_to_extri_intri,
     closed_form_inverse_se3_general) = _lazy_import_lingbot()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    str_paths = [str(p) for p in image_paths]

    images = load_and_preprocess_images(
        str_paths, mode="crop", image_size=image_size, patch_size=patch_size)
    images = images.to(device)
    num_frames, _, H_p, W_p = images.shape
    logger.info("LingBot input: %d frames @ %d×%d", num_frames, H_p, W_p)

    if mode == "auto":
        mode = "windowed" if num_frames > 500 else "streaming"
    if keyframe_interval is None:
        if mode == "streaming" and num_frames > STREAMING_MAX_FRAMES:
            keyframe_interval = (num_frames + STREAMING_MAX_FRAMES - 1) // STREAMING_MAX_FRAMES
        else:
            keyframe_interval = 1
    logger.info("Inference mode=%s, keyframe_interval=%d", mode, keyframe_interval)

    if use_sdpa is None:
        use_sdpa = not _flashinfer_available()
        logger.info("Attention backend: %s (auto)", "SDPA" if use_sdpa else "FlashInfer")
    model = _load_model(
        model_path, device, mode=mode, use_sdpa=use_sdpa,
        image_size=image_size, patch_size=patch_size, num_scale_frames=num_scale_frames,
        kv_cache_sliding_window=kv_cache_sliding_window,
        camera_num_iterations=camera_num_iterations, max_frame_num=max_frame_num)

    dtype = torch.float32
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        if getattr(model, "aggregator", None) is not None and dtype != torch.float32:
            model.aggregator = model.aggregator.to(dtype=dtype)

    output_device = torch.device("cpu")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        if mode == "streaming":
            predictions = model.inference_streaming(
                images, num_scale_frames=min(num_scale_frames, num_frames),
                keyframe_interval=keyframe_interval, output_device=output_device)
        else:
            predictions = model.inference_windowed(
                images, window_size=window_size, overlap_size=overlap_size,
                num_scale_frames=min(num_scale_frames, num_frames),
                keyframe_interval=keyframe_interval, output_device=output_device)

    pose_enc = predictions["pose_enc"]
    extrinsic_w2c, intrinsic = pose_encoding_to_extri_intri(pose_enc, (H_p, W_p))
    ext_4x4 = torch.zeros((*extrinsic_w2c.shape[:-2], 4, 4),
                          device=extrinsic_w2c.device, dtype=extrinsic_w2c.dtype)
    ext_4x4[..., :3, :4] = extrinsic_w2c
    ext_4x4[..., 3, 3] = 1.0
    ext_4x4_c2w = closed_form_inverse_se3_general(ext_4x4)
    extrinsic_c2w = ext_4x4_c2w[..., :3, :4]

    def _np(t: torch.Tensor) -> np.ndarray:
        return t.detach().to("cpu").float().numpy()

    def _strip_batch(arr: np.ndarray) -> np.ndarray:
        return arr[0] if arr.ndim >= 1 and arr.shape[0] == 1 else arr

    result = {
        "extrinsic_c2w": _strip_batch(_np(extrinsic_c2w)),
        "intrinsic": _strip_batch(_np(intrinsic)),
        "depth": _strip_batch(_np(predictions["depth"]).squeeze(-1)) if "depth" in predictions else None,
        "depth_conf": _strip_batch(_np(predictions["depth_conf"])) if "depth_conf" in predictions else None,
        "world_points": _strip_batch(_np(predictions["world_points"])) if "world_points" in predictions else None,
        "world_points_conf": _strip_batch(_np(predictions["world_points_conf"])) if "world_points_conf" in predictions else None,
        "processed_hw": (H_p, W_p),
        "images_proc": _strip_batch(_np(predictions["images"])) if "images" in predictions else _strip_batch(_np(images)),
    }

    del model, predictions
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # world_points 헤드가 없는 체크포인트(예: lingbot-map-long)는 depth 역투영으로 대체.
    if result["world_points"] is None and result["depth"] is not None:
        logger.info("No world_points head — backprojecting from depth.")
        result["world_points"] = backproject_depth_to_world(
            result["depth"], result["intrinsic"], result["extrinsic_c2w"])
        if result["world_points_conf"] is None:
            result["world_points_conf"] = (
                result["depth_conf"] if result["depth_conf"] is not None
                else np.ones_like(result["depth"]))
    return result


def backproject_depth_to_world(
    depth: np.ndarray,          # (S, H, W)
    intrinsic: np.ndarray,      # (S, 3, 3)
    extrinsic_c2w: np.ndarray,  # (S, 3, 4)
) -> np.ndarray:
    """픽셀 depth → world points (S, H, W, 3). OpenCV 핀홀 + c2w."""
    S, H, W = depth.shape
    vs, us = np.meshgrid(np.arange(H, dtype=np.float32),
                         np.arange(W, dtype=np.float32), indexing="ij")
    world = np.empty((S, H, W, 3), dtype=np.float32)
    for s in range(S):
        fx, fy = intrinsic[s, 0, 0], intrinsic[s, 1, 1]
        cx, cy = intrinsic[s, 0, 2], intrinsic[s, 1, 2]
        d = depth[s]
        cam = np.stack([(us - cx) / fx * d, (vs - cy) / fy * d, d], axis=-1)
        R = extrinsic_c2w[s, :3, :3]
        t = extrinsic_c2w[s, :3, 3]
        world[s] = cam @ R.T + t
    return world.astype(np.float32)
