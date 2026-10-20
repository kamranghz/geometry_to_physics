"""
DIAGNOSTIC ONLY — provisional camera-center Sim(3) reference (not TCSVT protocol).

Do not use for paper reference construction. See tcsvt/protocol.py for intended conventions.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from geometry_to_physics.data.recon import ReconBundle
from geometry_to_physics.registration.bop_world import bop_world_object_center_consistency
from geometry_to_physics.registration.camera_alignment import (
    align_bop_to_recon_cameras,
    apply_sim3_to_points,
    sim3_from_alignment_report,
)
def reference_object_center_recon_world(
    dataset: str,
    scene_id: str,
    object_id: str,
    recon: ReconBundle,
) -> dict[str, Any]:
    """
    Defensible reference: mean BOP-world GT center mapped by camera-only Sim(3) into recon world.
    Does not use per-view recon c2w for object placement.
    """
    align = align_bop_to_recon_cameras(dataset, scene_id, recon)
    if align.translation_residual_m_max is None or align.translation_residual_m_max > 0.05:
        return {"status": "UNVERIFIED", "reason": "camera_center_sim3_translation_residual_too_large"}
    sim3 = sim3_from_alignment_report(dataset, scene_id, recon)
    if sim3 is None:
        return {"status": "UNVERIFIED"}
    scale, R, t = sim3
    bop_rep = bop_world_object_center_consistency(dataset, scene_id, object_id, recon)
    if not bop_rep.per_view_center_bop_world_m:
        return {"status": "UNVERIFIED"}
    centers_bop = np.asarray(bop_rep.per_view_center_bop_world_m)
    center_bop_mean = centers_bop.mean(0)
    center_recon = apply_sim3_to_points(center_bop_mean.reshape(1, 3), scale, R, t)[0]
    return {
        "status": "VERIFIED",
        "center_recon_world_m": center_recon.tolist(),
        "bop_world_center_mean_m": center_bop_mean.tolist(),
        "bop_world_max_pairwise_mm": bop_rep.max_pairwise_mm,
        "sim3_scale": scale,
    }


