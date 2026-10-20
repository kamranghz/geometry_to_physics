from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def picasso_object_name_map(gt_scene_dir: Path) -> dict[str, str]:
    gt = json.loads((gt_scene_dir / "scene_gt.json").read_text(encoding="utf-8"))
    m: dict[str, str] = {}
    for lst in gt.values():
        for o in lst:
            oid = str(o["object_id"])
            name = o.get("object_name", oid)
            m[oid] = "jenga" if str(name).startswith("jenga") else str(name)
    return m


def picasso_pose_object(gt_scene_dir: Path, frame_id: str, object_id: str) -> dict[str, Any]:
    pose = json.loads((gt_scene_dir / "pose" / f"{frame_id}.json").read_text(encoding="utf-8"))
    scene_gt = json.loads((gt_scene_dir / "scene_gt.json").read_text(encoding="utf-8"))
    gt_list = scene_gt.get(str(frame_id), scene_gt.get(frame_id, []))
    idx = next((i for i, o in enumerate(gt_list) if str(o["object_id"]) == str(object_id)), None)
    if idx is None or idx >= len(pose["objects"]):
        raise KeyError(f"Picasso obj {object_id} missing at frame {frame_id}")
    return pose["objects"][idx]


def picasso_m2c_matrix(pose_obj: dict) -> tuple[np.ndarray, np.ndarray]:
    R = np.asarray(pose_obj["rotation_matrix"], dtype=np.float64).reshape(3, 3)
    t = np.asarray(pose_obj["translation_vector"], dtype=np.float64).reshape(3)
    if np.linalg.norm(t) > 50:
        t = t / 1000.0
    return R, t
