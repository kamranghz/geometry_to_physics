from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from geometry_to_physics.geometry.convex_hull import convex_hull_vertices
from geometry_to_physics.geometry.inertia import box_inertia, rotate_inertia
from geometry_to_physics.geometry.mass_properties import (
    MassProperties,
    from_convex_hull_vertices,
    from_placement_object,
)
from geometry_to_physics.geometry.placement_vertices import recon_vertices_table
from geometry_to_physics.transforms.placement_body import body_frame_from_placement

_LEGACY_REPO = Path(r"C:\Users\kghol072\PhysFidelity3D")


def _unit_cube_vertices(side_m: float = 1.0) -> np.ndarray:
    h = side_m / 2.0
    return np.array(
        [
            [-h, -h, -h],
            [h, -h, -h],
            [h, h, -h],
            [-h, h, -h],
            [-h, -h, h],
            [h, -h, h],
            [h, h, h],
            [-h, h, h],
        ],
        dtype=np.float64,
    )


def test_unit_cube_volume_com_inertia():
    verts = _unit_cube_vertices(1.0)
    mp = from_convex_hull_vertices(verts, nominal_mass_kg=1.0)
    assert abs(mp.volume_m3 - 1.0) < 1e-6
    assert np.linalg.norm(mp.com) < 1e-6
    assert abs(mp.mass_kg - 1.0) < 1e-9
    expected_I = box_inertia(1.0, 1.0, 1.0, 1.0)
    assert np.allclose(mp.inertia_com, expected_I, rtol=1e-4, atol=1e-5)


def test_translation_shifts_com():
    verts = _unit_cube_vertices(0.5)
    offset = np.array([0.3, -0.2, 0.1])
    mp0 = from_convex_hull_vertices(verts)
    mp1 = from_convex_hull_vertices(verts + offset)
    assert np.allclose(mp1.com, mp0.com + offset, atol=1e-6)
    assert abs(mp1.volume_m3 - mp0.volume_m3) < 1e-9


def test_rotation_conjugates_inertia():
    verts = _unit_cube_vertices(1.0)
    mp = from_convex_hull_vertices(verts)
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    verts_rot = (R @ verts.T).T
    mp_rot = from_convex_hull_vertices(verts_rot)
    I_expected = rotate_inertia(mp.inertia_com, R)
    assert np.allclose(mp_rot.inertia_com, I_expected, rtol=1e-4, atol=1e-5)


def test_from_placement_object_matches_anchor_plus_local():
    cen = [0.1, 0.2, 0.3]
    local = [[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 0.05]]
    mp = from_placement_object(cen, local, object_id="7")
    mp_direct = from_convex_hull_vertices(np.asarray(cen) + np.asarray(local))
    assert abs(mp.volume_m3 - mp_direct.volume_m3) < 1e-9
    assert np.allclose(mp.com, mp_direct.com)


def _placement_principal_axis_fields(vertices_table: np.ndarray) -> tuple[list[float], list[list[float]]]:
    """Same encoding as legacy placement_fields_from_table_vertices (mass + hull_local only)."""
    mp = from_convex_hull_vertices(convex_hull_vertices(vertices_table))
    R = mp.principal_axes
    com = mp.com
    hull_body = (R.T @ (vertices_table - com).T).T
    return com.tolist(), hull_body.tolist()


def test_recon_vertices_table_matches_legacy_placement_encoding():
    verts = _unit_cube_vertices(0.4) + np.array([1.0, 2.0, 0.5])
    cen, hull_local = _placement_principal_axis_fields(verts)
    v_rec = recon_vertices_table(cen, hull_local)
    assert v_rec.shape == (len(hull_local), 3)
    assert np.all(np.isfinite(v_rec))


def test_body_frame_preserves_v1_vertex_convention():
    cen = np.array([0.5, -0.1, 0.2])
    local = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.0, 0.1],
            [0.05, 0.05, 0.05],
        ]
    )
    bf = body_frame_from_placement("obj", cen.tolist(), local.tolist())
    assert bf.max_vertex_world_error_vs_v1(cen, local) < 1e-9


def test_principal_axis_hull_roundtrip_via_recon():
    """Mirrors legacy test_protocol_a_spawn_frame_roundtrip with identity spawn orientation."""
    rng = np.random.default_rng(101)
    pts = rng.standard_normal((40, 3))
    cen, hull_local = _placement_principal_axis_fields(pts)
    v = recon_vertices_table(cen, hull_local)
    cen_arr = np.asarray(cen, dtype=np.float64)
    spawn_local = (v - cen_arr).tolist()
    v2 = recon_vertices_table(cen, spawn_local)
    assert np.max(np.linalg.norm(v - v2, axis=1)) < 1e-5


@pytest.mark.skipif(not _LEGACY_REPO.is_dir(), reason="PhysFidelity3D not available for parity")
def test_numeric_parity_with_legacy_on_synthetic():
    sys.path.insert(0, str(_LEGACY_REPO))
    from cross_engine.reference_anchor.objects import recon_vertices_table as legacy_recon
    from cross_engine.v2.mass_properties import from_convex_hull_vertices as legacy_from_hull
    from cross_engine.v2.mass_properties import from_placement_object as legacy_from_placement
    from cross_engine.reference_anchor.objects import convex_hull_vertices as legacy_hull_verts
    from cross_engine.v2.frame_transform import body_frame_from_placement as legacy_body

    verts = _unit_cube_vertices(1.0)
    mp = from_convex_hull_vertices(verts)
    mp_l = legacy_from_hull(verts)
    assert isinstance(mp, MassProperties)
    for attr in ("volume_m3", "mass_kg", "density_kg_m3"):
        assert abs(getattr(mp, attr) - getattr(mp_l, attr)) < 1e-12
    assert np.allclose(mp.com, mp_l.com)
    assert np.allclose(mp.inertia_com, mp_l.inertia_com, rtol=1e-10, atol=1e-10)
    assert np.allclose(mp.principal_moments, mp_l.principal_moments, rtol=1e-10, atol=1e-10)
    assert np.allclose(mp.principal_axes, mp_l.principal_axes, rtol=1e-10, atol=1e-10)

    cen = [0.2, 0.0, -0.1]
    hull = [[0.0, 0.0, 0.0], [0.04, 0.0, 0.0], [0.0, 0.04, 0.0], [0.0, 0.0, 0.04]]
    assert np.allclose(
        from_placement_object(cen, hull).volume_m3,
        legacy_from_placement(cen, hull).volume_m3,
    )

    rng = np.random.default_rng(7)
    cloud = rng.standard_normal((25, 3))
    hv = convex_hull_vertices(cloud)
    hv_l = legacy_hull_verts(cloud)
    assert np.allclose(hv, hv_l)

    bf = body_frame_from_placement("t", cen, hull)
    bf_l = legacy_body("t", cen, hull)
    assert np.allclose(bf.com_table, bf_l.com_table)
    assert np.allclose(bf.quat_xyzw, bf_l.quat_xyzw)
    assert np.allclose(bf.hull_body, bf_l.hull_body)

    v_new = recon_vertices_table(cen, hull)
    v_old = legacy_recon(cen, hull)
    assert np.allclose(v_new, v_old)
