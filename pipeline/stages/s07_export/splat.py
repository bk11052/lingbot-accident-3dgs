"""3DGS point_cloud.ply → .splat 변환 (antimatter15 .splat 포맷).

.splat 레코드(가우시안당 32바이트):
    position : 3 × float32 (12B)
    scale    : 3 × float32 (12B)   = exp(scale_i)
    color    : 4 × uint8   (4B)    = rgb(SH DC→0.5+C0·f_dc), a=sigmoid(opacity)
    rotation : 4 × uint8   (4B)    = 정규화 쿼터니언 → clip(q·128+128, 0, 255)

외부 도구 없이 lingbot env 안에서 동작하도록 바이너리 PLY를 직접 파싱한다.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SH_C0 = 0.28209479177387814

_PLY_DTYPES = {
    "char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2",
    "int": "i4", "uint": "u4", "float": "f4", "double": "f8",
    "float32": "f4", "float64": "f8", "uint8": "u1", "int32": "i4",
}


def _read_binary_ply(path: Path) -> np.ndarray:
    """binary_little_endian PLY의 vertex 요소를 structured array로 반환."""
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic != b"ply":
            raise ValueError(f"Not a PLY file: {path}")
        fmt = f.readline().strip()
        if b"binary_little_endian" not in fmt:
            raise ValueError(f"Only binary_little_endian PLY supported (got {fmt!r}).")
        n_vertex = 0
        props: list[tuple[str, str]] = []
        in_vertex = False
        while True:
            line = f.readline().strip()
            if line == b"end_header":
                break
            toks = line.split()
            if toks[0] == b"element":
                in_vertex = toks[1] == b"vertex"
                if in_vertex:
                    n_vertex = int(toks[2])
            elif toks[0] == b"property" and in_vertex:
                ply_type = toks[1].decode()
                name = toks[2].decode()
                props.append((name, "<" + _PLY_DTYPES[ply_type]))
        dtype = np.dtype(props)
        data = np.frombuffer(f.read(n_vertex * dtype.itemsize), dtype=dtype, count=n_vertex)
    return data


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ply_to_splat(ply_path: Path, splat_path: Path) -> int:
    """3DGS PLY → .splat. 기록한 가우시안 개수 반환."""
    v = _read_binary_ply(ply_path)
    names = v.dtype.names
    n = len(v)

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
    scales = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1)).astype(np.float32)
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
    rgb = np.clip(0.5 + SH_C0 * f_dc, 0.0, 1.0)
    alpha = _sigmoid(v["opacity"]) if "opacity" in names else np.ones(n, np.float32)
    color = np.concatenate([rgb, alpha[:, None]], axis=1)
    color_u8 = np.clip(color * 255.0, 0, 255).astype(np.uint8)

    quat = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1).astype(np.float64)
    quat /= (np.linalg.norm(quat, axis=1, keepdims=True) + 1e-12)
    rot_u8 = np.clip(quat * 128.0 + 128.0, 0, 255).astype(np.uint8)

    # 중요도(부피×불투명도) 내림차순 정렬 — 스트리밍/블렌딩 품질용.
    importance = alpha * scales.prod(axis=1)
    order = np.argsort(-importance)

    rec = np.zeros(n, dtype=np.dtype([
        ("pos", "<f4", (3,)), ("scale", "<f4", (3,)),
        ("color", "u1", (4,)), ("rot", "u1", (4,))]))
    rec["pos"] = xyz[order]
    rec["scale"] = scales[order]
    rec["color"] = color_u8[order]
    rec["rot"] = rot_u8[order]

    splat_path.parent.mkdir(parents=True, exist_ok=True)
    with open(splat_path, "wb") as f:
        f.write(rec.tobytes())
    logger.info("Wrote %d gaussians → %s (%.1f MB)", n, splat_path,
                splat_path.stat().st_size / 1e6)
    return n
