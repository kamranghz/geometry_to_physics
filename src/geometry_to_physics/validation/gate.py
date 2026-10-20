from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from geometry_to_physics.config import load_paths
from geometry_to_physics.data.paths import recon_scene_dir
from geometry_to_physics.data.recon import load_recon_bundle
from geometry_to_physics.provenance.blocklist import classify_placement_file, placement_data_path
from geometry_to_physics.registration.camera_checks import verify_rgb_pose_indexing
from geometry_to_physics.registration.gt_chain import gt_vertices_recon_world


def run_validation_gate(
    cases_yaml: Path,
    out_dir: Path,
    *,
    picasso_mesh_names: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Engineering preflight only — not TCSVT scientific success criteria."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = yaml.safe_load(cases_yaml.read_text(encoding="utf-8"))["cases"]
    paths = load_paths()
    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gate_version": "engineering-preflight-1",
        "role": "engineering_preflight_only",
        "gt_root": str(paths.gt_root),
        "recon_root": str(paths.recon_root),
        "canonical_root": str(paths.canonical_root) if paths.canonical_root else None,
        "cases": [],
        "summary": {"PASS": 0, "FAIL": 0},
    }
    for c in cases:
        case_out: dict[str, Any] = {"case_id": c["id"], "checks": {}}
        pmn = (picasso_mesh_names or {}).get(c["id"])
        recon_path = recon_scene_dir(c["method"], c["scene_folder"])
        if not recon_path.is_dir():
            case_out["checks"]["recon_directory"] = {"status": "FAIL", "path": str(recon_path)}
            case_out["overall"] = "FAIL"
            report["summary"]["FAIL"] += 1
            report["cases"].append(case_out)
            continue

        meta_ok = (recon_path / "metadata.json").is_file()
        poses_ok = (recon_path / "camera_poses.npy").is_file()
        case_out["checks"]["recon_artifacts"] = {
            "status": "PASS" if meta_ok and poses_ok else "FAIL",
            "metadata_json": meta_ok,
            "camera_poses_npy": poses_ok,
        }

        try:
            recon = load_recon_bundle(recon_path)
            case_out["checks"]["rgb_pose_indexing"] = verify_rgb_pose_indexing(recon)
        except Exception as e:  # noqa: BLE001
            case_out["checks"]["rgb_pose_indexing"] = {"status": "FAIL", "error": str(e)}
            recon = None

        if recon is not None:
            try:
                gt0 = gt_vertices_recon_world(
                    c["dataset"], c["scene_id"], c["object_id"], recon, 0, picasso_mesh_name=pmn
                )
                ok = gt0.vertices_recon_world_m.size > 0 and bool(_np_isfinite(gt0.vertices_recon_world_m))
                case_out["checks"]["gt_transform_sanity"] = {
                    "status": "PASS" if ok else "FAIL",
                    "n_vertices": int(gt0.vertices_recon_world_m.shape[0]),
                    "chain": gt0.chain,
                }
            except Exception as e:  # noqa: BLE001
                case_out["checks"]["gt_transform_sanity"] = {"status": "FAIL", "error": str(e)}

        pl_path = placement_data_path(c["method"], c["scene_folder"])
        if pl_path is None:
            case_out["checks"]["canonical_placement"] = {
                "status": "UNVERIFIED",
                "reason": "GTP_CANONICAL_ROOT unset",
            }
        else:
            st = classify_placement_file(pl_path)
            case_out["checks"]["canonical_placement"] = {
                "status": st,
                "path": str(pl_path),
                "exists": pl_path.is_file(),
            }

        fail = False
        for ch in case_out["checks"].values():
            if not isinstance(ch, dict):
                continue
            st = ch.get("status")
            if st in ("FAIL", "FAILED"):
                fail = True
            if ch.get("metadata_json") is False or ch.get("camera_poses_npy") is False:
                fail = True
        case_out["overall"] = "FAIL" if fail else "PASS"
        report["summary"][case_out["overall"]] += 1
        report["cases"].append(case_out)

    out_json = out_dir / "validation_gate_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _np_isfinite(arr) -> bool:
    import numpy as np

    return bool(np.isfinite(np.asarray(arr)).all())
