"""Per-track 3D Kalman filter wrapper around AB3DMOT's `KF` class.

AB3DMOT은 비상업 연구용(CMU). KF state는 10-dim (x,y,z,theta,l,w,h,vx,vy,vz).
단안 depth 역투영으로는 (x,y,z)만 관측 가능 → theta/l/w/h 는 더미 상수로 채운다
(잔차 0 → 위치/속도 추정에 영향 없음). 즉 AB3DMOT를 "3D 상수속도 KF"로만 쓴다.
데이터 연관은 ByteTrack track_id 가 대신하므로 생략.
"""

import os
import sys
from pathlib import Path

import numpy as np

_AB3DMOT_ROOT = os.environ.get(
    "AB3DMOT_ROOT",
    str(Path(__file__).resolve().parents[3] / "third_party" / "AB3DMOT"),
)
if _AB3DMOT_ROOT not in sys.path:
    sys.path.insert(0, _AB3DMOT_ROOT)

from AB3DMOT_libs.kalman_filter import KF  # noqa: E402

_DUMMY_THETA = 0.0
_DUMMY_SIZE = 1.0


class Track3DKalman:
    """Per-track 3D 위치 칼만필터. predict()로 시간 전진, update(xyz)로 관측 융합."""

    def __init__(self, xyz_init, track_id: str, observation_noise: float = 1.0):
        bbox3d = np.array(
            [xyz_init[0], xyz_init[1], xyz_init[2],
             _DUMMY_THETA, _DUMMY_SIZE, _DUMMY_SIZE, _DUMMY_SIZE],
            dtype=np.float64)
        self._kf = KF(bbox3d, info=None, ID=track_id)
        self._kf.kf.R[0:3, 0:3] *= observation_noise
        self.frames_since_update = 0

    def predict(self) -> None:
        self._kf.kf.predict()
        self.frames_since_update += 1

    def update(self, xyz) -> None:
        z = np.array(
            [xyz[0], xyz[1], xyz[2],
             _DUMMY_THETA, _DUMMY_SIZE, _DUMMY_SIZE, _DUMMY_SIZE],
            dtype=np.float64)
        self._kf.kf.update(z)
        self.frames_since_update = 0

    def xyz(self) -> tuple[float, float, float]:
        x = self._kf.kf.x.flatten()
        return float(x[0]), float(x[1]), float(x[2])

    def velocity_per_frame(self) -> tuple[float, float, float]:
        x = self._kf.kf.x.flatten()
        return float(x[7]), float(x[8]), float(x[9])
