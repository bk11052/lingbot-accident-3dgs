"""trajectories.json serialiser."""

import json
from pathlib import Path


def write_trajectories(tracks_out, frame_count_total, skipped, out_path, *, params):
    """track_id 오름차순으로 직렬화. 좌표계: world (gravity-aligned, 메트릭)."""
    sorted_keys = sorted(tracks_out.keys(), key=lambda s: int(s))
    tracks_sorted = {tid: tracks_out[tid] for tid in sorted_keys}
    payload = {
        "metadata": {
            "num_tracks": len(tracks_sorted),
            "frame_count": frame_count_total,
            "coord_system": "world_gravity_aligned_metric",
            "source": "s06_trajectory",
            "params": params,
            "skipped": skipped,
        },
        "tracks": tracks_sorted,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    return str(out_path)
