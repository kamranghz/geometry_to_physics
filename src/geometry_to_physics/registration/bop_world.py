from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geometry_to_physics.data.recon import ReconBundle
from geometry_to_physics.registration.gt_chain import gt_vertices_bop_world


@dataclass(frozen=True)
class BopWorldStaticReport:
    per_view_center_bop_world_m: list[list[float]]
    frame_ids: list[str]
    pairwise_mm: list[float]
    max_pairwise_mm: float | None
    static_object_assumption: str

    def to_dict(self) -> dict:
        return {
            "frame_ids": self.frame_ids,
            "per_view_center_bop_world_m": self.per_view_center_bop_world_m,
            "pairwise_translation_disagreement_mm": self.pairwise_mm,
            "max_pairwise_mm": self.max_pairwise_mm,
            "static_object_assumption": self.static_object_assumption,
            "method": "cad_m2c_scene_camera_only_no_recon_poses",
            "threshold_mm": 1.0,
        }


def bop_world_object_center_consistency(
    dataset: str,
    scene_id: str,
    object_id: str,
    recon: ReconBundle,
    *,
    threshold_mm: float = 1.0,
) -> BopWorldStaticReport:
    """Object COM in BOP metric world from scene_gt + scene_camera only."""
    if dataset not in ("ycbv", "tless"):
        return BopWorldStaticReport([], [], [], None, "UNVERIFIED")
    centers = []
    frames = []
    for p in recon.image_paths:
        stem = p.stem
        try:
            v = gt_vertices_bop_world(dataset, scene_id, object_id, stem)
            centers.append(v.mean(0))
            frames.append(stem)
        except (KeyError, FileNotFoundError, ValueError):
            continue
    if len(centers) < 2:
        return BopWorldStaticReport([], frames, [], None, "UNVERIFIED")
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
    return BopWorldStaticReport(C.tolist(), frames, pairwise, mx, status)
