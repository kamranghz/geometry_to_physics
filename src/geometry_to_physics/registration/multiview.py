from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geometry_to_physics.data.recon import ReconBundle
from geometry_to_physics.registration.gt_chain import gt_vertices_recon_world


@dataclass(frozen=True)
class MultiviewReport:
    per_view_center_m: list[list[float]]
    pairwise_mm: list[float]
    max_pairwise_mm: float | None
    static_object_assumption: str  # PASS | FAIL | UNVERIFIED

    def to_dict(self) -> dict:
        return {
            "per_view_center_recon_world_m": self.per_view_center_m,
            "pairwise_translation_disagreement_mm": self.pairwise_mm,
            "max_pairwise_mm": self.max_pairwise_mm,
            "static_object_assumption": self.static_object_assumption,
            "policy_note": (
                "Disagreement may reflect recon pose inconsistency, GT labeling, correspondence, "
                "or violated static-object assumption; not attributed to a single cause."
            ),
        }


def multiview_gt_center_consistency(
    dataset: str,
    scene_id: str,
    object_id: str,
    recon: ReconBundle,
    *,
    threshold_mm: float = 5.0,
    picasso_mesh_name: str | None = None,
) -> MultiviewReport:
    centers = []
    for vi in range(recon.n_views):
        try:
            gt = gt_vertices_recon_world(dataset, scene_id, object_id, recon, vi, picasso_mesh_name)
            centers.append(gt.vertices_recon_world_m.mean(0))
        except (KeyError, FileNotFoundError):
            continue
    if len(centers) < 2:
        return MultiviewReport([], [], None, "UNVERIFIED")
    C = np.asarray(centers)
    pairwise = []
    for i in range(len(C)):
        for j in range(i + 1, len(C)):
            pairwise.append(float(np.linalg.norm(C[i] - C[j]) * 1000.0))
    mx = max(pairwise) if pairwise else None
    if mx is None:
        status = "UNVERIFIED"
    elif mx > threshold_mm:
        status = "FAIL"
    else:
        status = "PASS"
    return MultiviewReport(C.tolist(), pairwise, mx, status)
