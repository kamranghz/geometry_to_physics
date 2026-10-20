from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from geometry_to_physics.config import load_paths
from geometry_to_physics.constants import METHODS
from geometry_to_physics.data.paths import gt_scene_dir, recon_scene_dir
from geometry_to_physics.provenance.hashing import file_record


DERIVED_ARTIFACT_NAMES = {
    "placement_data.json",
    "object_property_pairs.csv",
    "trajectory",
}


def _is_derived_recon_extra(path: Path) -> bool:
    name = path.name.lower()
    if name in DERIVED_ARTIFACT_NAMES:
        return True
    if "physics_eval" in str(path):
        return True
    return False


def inventory_recon_scene(method: str, scene_folder: str) -> dict[str, Any]:
    root = recon_scene_dir(method, scene_folder)
    if not root.is_dir():
        return {"method": method, "scene_folder": scene_folder, "status": "missing"}
    roles = {
        "metadata.json": "recon_export_metadata",
        "camera_poses.npy": "recon_camera_poses_c2w",
        "pts3d.npy": "recon_dense_pts3d",
        "depth.npy": "recon_depth",
        "intrinsics.npy": "recon_intrinsics",
        "confidence.npy": "recon_confidence",
        "scene.ply": "recon_merged_pointcloud",
    }
    files: list[dict] = []
    for fname, role in roles.items():
        files.append(file_record(root / fname, role))
    meta = root / "metadata.json"
    rgb_list: list[dict] = []
    meta_obj: dict = {}
    if meta.is_file():
        meta_obj = json.loads(meta.read_text(encoding="utf-8"))
        for i, p in enumerate(meta_obj.get("image_paths", [])):
            pp = Path(p)
            rgb_list.append(
                {
                    "view_index": i,
                    **file_record(pp, "rgb_source_image"),
                    "derived_placement_hint": False,
                }
            )
    return {
        "method": method,
        "scene_folder": scene_folder,
        "recon_root": str(root),
        "status": "present",
        "coordinate_frame": meta_obj.get("coordinate_frame"),
        "scale_field": meta_obj.get("scale"),
        "scale_normalization": meta_obj.get("scale_normalization"),
        "artifacts": files,
        "rgb_provenance": rgb_list,
        "validation_status": "RAW_RECON_EXPORT",
    }


def inventory_gt_scene(dataset: str, scene_id: str) -> dict[str, Any]:
    gdir = gt_scene_dir(dataset, scene_id)
    if not gdir.is_dir():
        return {"dataset": dataset, "scene_id": scene_id, "status": "missing"}
    base_files = []
    for fname in ("scene_gt.json", "scene_camera.json", "scene_gt_info.json"):
        base_files.append(file_record(gdir / fname, "bop_scene_metadata"))
    rgb_dir = gdir / "rgb"
    rgb_count = len(list(rgb_dir.glob("*.png"))) if rgb_dir.is_dir() else 0
    picasso_pose_count = None
    if dataset == "picasso":
        pose_dir = gdir / "pose"
        picasso_pose_count = len(list(pose_dir.glob("*.json"))) if pose_dir.is_dir() else 0
    return {
        "dataset": dataset,
        "scene_id": scene_id,
        "gt_scene_dir": str(gdir),
        "status": "present",
        "bop_metadata": base_files,
        "rgb_image_count": rgb_count,
        "picasso_pose_json_count": picasso_pose_count,
        "units": {
            "bop_camera_translation": "mm in JSON, converted to m in code",
            "bop_m2c_translation": "mm in JSON, converted to m in code",
            "recon_scaled_exports": "metadata scale_normalization when present",
        },
        "validation_status": "RAW_GT",
    }


def build_manifest(cases_yaml: Path, out_path: Path) -> dict[str, Any]:
    paths = load_paths()
    cases = yaml.safe_load(cases_yaml.read_text(encoding="utf-8"))["cases"]
    manifest: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gt_root": str(paths.gt_root),
        "recon_root": str(paths.recon_root),
        "note": "Does not include PhysFidelity3D canonical placements or simulation outputs.",
        "methods_listed": list(METHODS),
        "cases": [],
        "recon_method_scene_counts": {},
    }
    for method in METHODS:
        mroot = paths.recon_root / method
        if mroot.is_dir():
            manifest["recon_method_scene_counts"][method] = len([p for p in mroot.iterdir() if p.is_dir()])
    for c in cases:
        manifest["cases"].append(
            {
                "case_id": c["id"],
                "gt": inventory_gt_scene(c["dataset"], c["scene_id"]),
                "recon": inventory_recon_scene(c["method"], c["scene_folder"]),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
