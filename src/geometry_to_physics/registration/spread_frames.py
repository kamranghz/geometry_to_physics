from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SceneContext:
    dataset: str
    scene_id: str
    gt_scene_dir: Path
    anchor_frame_id: str


def scene_id_from_placement(placement: dict) -> str:
    dataset = placement["dataset"]
    scene_field = placement["scene"]
    return scene_field.replace(f"{dataset}_", "").split("_spread")[0]


def spread_frame_ids(dataset: str, scene_id: str, gt_scene_dir: Path) -> list[str]:
    """Three spread views for reference anchoring (legacy reference_anchor/gt_geometry)."""
    if dataset in ("ycbv", "tless"):
        cam = json.loads((gt_scene_dir / "scene_camera.json").read_text(encoding="utf-8"))
        keys = sorted(cam.keys(), key=lambda x: int(x))
        n = len(keys)
        return [keys[0], keys[n // 3], keys[2 * n // 3]]
    pose_dir = gt_scene_dir / "pose"
    frames = sorted((p.stem for p in pose_dir.glob("*.json")), key=int)
    mid = (len(frames) - 1) // 2
    return [frames[0], frames[mid], frames[-1]]


def spread_view_index(spread_ids: list[str], frame_id: str) -> int:
    fid_norm = str(int(frame_id)) if str(frame_id).isdigit() else str(frame_id)
    return next(
        (i for i, x in enumerate(spread_ids) if str(int(x)) == fid_norm or str(x) == fid_norm),
        0,
    )
