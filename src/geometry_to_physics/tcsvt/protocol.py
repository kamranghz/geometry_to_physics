"""
TCSVT reference-anchored study constants recovered from PhysFidelity3D cross_engine/reference_anchor/config.py
and placement.py. Executable parity with the legacy pipeline is UNVERIFIED until ported and tested.
"""
from __future__ import annotations

from typing import Literal

from geometry_to_physics.constants import METHODS

SEEDS: tuple[int, ...] = (101, 202, 303)
ENGINES: tuple[str, ...] = ("pybullet", "physx", "newton")
SCALE_LEVELS: tuple[float, ...] = (0.95, 0.98, 1.0, 1.02, 1.05)

KNOWN_EXCLUSIONS: tuple[dict[str, str], ...] = (
    {"method": "MoVieS", "scene_folder": "picasso_jenga_2_spread_scaled", "object_id": "0"},
)

ProtocolKind = Literal["B_recon", "B_ref", "A_recon", "A_ref"]

# Reference geometry (legacy specification — implementation pending in this repo):
# - reference_vertices_table: CAD -> m2c -> recon c2w[anchor view] -> table frame (PlacementPlane).
# - frame_registration.compute_frame_registration: BOP SE(3) avg for diagnostics, not Umeyama Sim(3) reference.
