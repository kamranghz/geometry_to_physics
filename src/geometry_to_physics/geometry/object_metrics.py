"""
Static geometric and mass-property comparisons (TCSVT reference_anchor/metrics).

Reconstruction geometry uses canonical v1 table vertices unless callers pass
pre-expanded vertex arrays.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from geometry_to_physics.geometry.chamfer import (
    DEFAULT_CHAMFER_SAMPLE_COUNT,
    DEFAULT_CHAMFER_SEED,
    chamfer_m_cpu,
)
from geometry_to_physics.geometry.convex_hull import convex_hull_vertices
from geometry_to_physics.geometry.mass_properties import MassProperties, from_convex_hull_vertices
from geometry_to_physics.geometry.placement_vertices import table_vertices_from_canonical_placement


def _sym3(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.float64).reshape(3, 3)
    return 0.5 * (M + M.T)


def mass_props_table(vertices_table: np.ndarray, nominal_mass_kg: float = 1.0) -> MassProperties:
    hull_v = convex_hull_vertices(vertices_table)
    return from_convex_hull_vertices(hull_v, nominal_mass_kg=nominal_mass_kg)


def inertia_orientation_angle_rad(I_ref: np.ndarray, I_rec: np.ndarray) -> float:
    evals_r, evecs_r = np.linalg.eigh(_sym3(I_ref))
    _, evecs_c = np.linalg.eigh(_sym3(I_rec))
    angles = []
    for j in range(3):
        v = evecs_r[:, j]
        best = max(abs(float(np.dot(v, evecs_c[:, k]))) for k in range(3))
        angles.append(float(np.arccos(np.clip(best, 0.0, 1.0))))
    return float(max(angles))


def mutual_hull_penetration_depth(a: np.ndarray, b: np.ndarray, samples: int = 500) -> float:
    rng = np.random.default_rng(0)
    if len(a) > samples:
        a = a[rng.choice(len(a), samples, replace=False)]
    if len(b) > samples:
        b = b[rng.choice(len(b), samples, replace=False)]
    tb = cKDTree(b)
    ta = cKDTree(a)
    d_ab = tb.query(a, k=1)[0]
    d_ba = ta.query(b, k=1)[0]
    return float(max(np.percentile(d_ab, 5), np.percentile(d_ba, 5)))


def table_penetration_depth(vertices_table: np.ndarray, z_floor: float = 0.0) -> float:
    return float(max(0.0, z_floor - float(np.min(vertices_table[:, 2]))))


def compare_static_objects(
    recon_cen_table: list[float],
    recon_hull_local: list[list[float]],
    ref_vertices_table: np.ndarray,
    *,
    chamfer_n: int = DEFAULT_CHAMFER_SAMPLE_COUNT,
    chamfer_seed: int = DEFAULT_CHAMFER_SEED,
) -> dict[str, Any]:
    """
    Compare canonical v1 reconstruction placement to reference CAD vertices in table frame.
    """
    v_rec = table_vertices_from_canonical_placement(recon_cen_table, recon_hull_local)
    v_ref = np.asarray(ref_vertices_table, dtype=np.float64)
    mp_rec = mass_props_table(v_rec)
    mp_ref = mass_props_table(v_ref)
    dV = float(mp_rec.volume_m3 - mp_ref.volume_m3)
    rel_V = abs(dV) / max(mp_ref.volume_m3, 1e-12)
    d_com = float(np.linalg.norm(mp_rec.com - mp_ref.com))
    J_rec = np.sort(mp_rec.principal_moments)
    J_ref = np.sort(mp_ref.principal_moments)
    rel_J = float(np.max(np.abs(J_rec - J_ref) / np.maximum(J_ref, 1e-12)))
    frob_I = float(
        np.linalg.norm(_sym3(mp_rec.inertia_com) - _sym3(mp_ref.inertia_com))
        / max(np.linalg.norm(_sym3(mp_ref.inertia_com)), 1e-12)
    )
    orn = inertia_orientation_angle_rad(mp_ref.inertia_com, mp_rec.inertia_com)
    return {
        "volume_rec_m3": mp_rec.volume_m3,
        "volume_ref_m3": mp_ref.volume_m3,
        "volume_abs_err_m3": dV,
        "volume_rel_err": rel_V,
        "com_displacement_m": d_com,
        "principal_moment_max_rel_err": rel_J,
        "inertia_frob_rel_err": frob_I,
        "inertia_orientation_max_axis_angle_rad": orn,
        "hull_chamfer_m": chamfer_m_cpu(v_rec, v_ref, n=chamfer_n, seed=chamfer_seed),
        "hull_mutual_penetration_proxy_m": mutual_hull_penetration_depth(v_rec, v_ref),
        "table_penetration_rec_m": table_penetration_depth(v_rec),
        "table_penetration_ref_m": table_penetration_depth(v_ref),
        "n_vertices_rec": int(v_rec.shape[0]),
        "n_vertices_ref": int(v_ref.shape[0]),
    }
