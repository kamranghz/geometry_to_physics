"""
Optional scientific diagnostics (BOP spread, camera Sim(3), multiview recon chain).

NOT used by the engineering validation gate. Does not define paper success criteria.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from geometry_to_physics.data.paths import recon_scene_dir
from geometry_to_physics.data.recon import load_recon_bundle
from geometry_to_physics.registration.bop_world import bop_world_object_center_consistency
from geometry_to_physics.registration.camera_alignment import align_bop_to_recon_cameras
from geometry_to_physics.registration.camera_checks import relative_baseline_ratio
from geometry_to_physics.registration.multiview import multiview_gt_center_consistency


def run_optional_diagnostics(cases_yaml: Path) -> dict[str, Any]:
    cases = yaml.safe_load(cases_yaml.read_text(encoding="utf-8"))["cases"]
    out: dict[str, Any] = {"cases": []}
    for c in cases:
        recon = load_recon_bundle(recon_scene_dir(c["method"], c["scene_folder"]))
        out["cases"].append(
            {
                "case_id": c["id"],
                "bop_world_object_static": bop_world_object_center_consistency(
                    c["dataset"], c["scene_id"], c["object_id"], recon
                ).to_dict(),
                "camera_sim3_alignment": align_bop_to_recon_cameras(
                    c["dataset"], c["scene_id"], recon
                ).to_dict(),
                "multiview_per_view_recon_c2w": multiview_gt_center_consistency(
                    c["dataset"], c["scene_id"], c["object_id"], recon
                ).to_dict(),
                "camera_baseline_scale": relative_baseline_ratio(c["dataset"], c["scene_id"], recon, 0, 1),
            }
        )
    return out
