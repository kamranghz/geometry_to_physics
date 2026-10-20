from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_scene_json(gt_scene_dir: Path) -> tuple[dict, dict]:
    scene_gt = json.loads((gt_scene_dir / "scene_gt.json").read_text(encoding="utf-8"))
    scene_cam = json.loads((gt_scene_dir / "scene_camera.json").read_text(encoding="utf-8"))
    return scene_gt, scene_cam


def frame_key(scene_gt: dict, frame_stem: str) -> str:
    if frame_stem in scene_gt:
        return frame_stem
    alt = str(int(frame_stem))
    if alt in scene_gt:
        return alt
    raise KeyError(f"frame {frame_stem} not in scene_gt")


def object_gt_entry(scene_gt: dict, frame_key: str, obj_id: str) -> dict[str, Any]:
    objs = scene_gt[frame_key]
    hit = next((o for o in objs if str(o["obj_id"]) == str(obj_id)), None)
    if hit is None:
        raise KeyError(f"obj_id {obj_id} missing in frame {frame_key}")
    return hit
