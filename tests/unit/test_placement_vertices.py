from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from geometry_to_physics.geometry.placement_vertices import (
    recon_vertices_table,
    table_vertices_from_canonical_placement,
)
from geometry_to_physics.transforms.placement_body import body_frame_from_placement

_CANONICAL_ROOT = Path(
    r"C:\Users\kghol072\PhysFidelity3D\outputs\cross_engine\full_local_rerun\canonical"
)
_YCBV_PLACEMENT = _CANONICAL_ROOT / "DUSt3R" / "ycbv_000048_spread_scaled" / "placement_data.json"
_PICASSO_PLACEMENT = _CANONICAL_ROOT / "DUSt3R" / "picasso_holder_cup_spread_scaled" / "placement_data.json"


def test_canonical_v1_synthetic_exact():
    cen = np.array([1.0, -0.5, 0.2])
    hull = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.03, 0.0, 0.0],
            [0.0, 0.04, 0.0],
            [0.0, 0.0, 0.05],
        ]
    )
    v = table_vertices_from_canonical_placement(cen, hull)
    assert v.shape == (4, 3)
    assert np.allclose(v, cen + hull)


def test_translation_preserves_relative_geometry():
    cen = np.array([0.0, 0.0, 0.0])
    hull = np.random.default_rng(1).standard_normal((12, 3)) * 0.05
    offset = np.array([2.0, -1.0, 0.5])
    v0 = table_vertices_from_canonical_placement(cen, hull)
    v1 = table_vertices_from_canonical_placement(cen + offset, hull)
    assert np.allclose(v1 - v0, offset)


def test_invalid_inputs_raise():
    cen = [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="at least one"):
        table_vertices_from_canonical_placement(cen, np.zeros((0, 3)))
    with pytest.raises(ValueError, match="shape"):
        table_vertices_from_canonical_placement(cen, np.zeros((3, 2)))


def test_body_frame_preserves_canonical_v1_vertices():
    cen = np.array([0.3, 0.1, 0.05])
    hull = np.array(
        [
            [0.01, 0.0, 0.0],
            [-0.02, 0.03, 0.01],
            [0.0, -0.01, 0.02],
            [0.02, 0.02, -0.01],
        ]
    )
    v = table_vertices_from_canonical_placement(cen, hull)
    bf = body_frame_from_placement("obj", cen.tolist(), hull.tolist())
    err = bf.max_vertex_world_error_vs_v1(cen, hull)
    assert err < 1e-9
    assert np.allclose(v, bf.v_table)


def _physics_v1_formula(cen_table: list, hull_local: list) -> np.ndarray:
    cen = np.asarray(cen_table, dtype=np.float64)
    return cen + np.asarray(hull_local, dtype=np.float64)


def _max_reconstruction_error_m(placement_path: Path) -> float:
    pl = json.loads(placement_path.read_text(encoding="utf-8"))
    max_err = 0.0
    for obj in pl["objects"].values():
        cen = obj["cen_table"]
        hull = obj["hull_local"]
        v = table_vertices_from_canonical_placement(cen, hull)
        v_phys = _physics_v1_formula(cen, hull)
        max_err = max(max_err, float(np.max(np.linalg.norm(v - v_phys, axis=1))))
    return max_err


@pytest.mark.skipif(not _YCBV_PLACEMENT.is_file(), reason="canonical YCB-V 000048 placement not on disk")
def test_real_canonical_ycbv_000048():
    err = _max_reconstruction_error_m(_YCBV_PLACEMENT)
    assert err < 1e-12
    pl = json.loads(_YCBV_PLACEMENT.read_text(encoding="utf-8"))
    obj = pl["objects"]["1"]
    v_canon = table_vertices_from_canonical_placement(obj["cen_table"], obj["hull_local"])
    v_pa = recon_vertices_table(obj["cen_table"], obj["hull_local"])
    assert float(np.max(np.linalg.norm(v_canon - v_pa, axis=1))) > 0.01


@pytest.mark.skipif(
    not _PICASSO_PLACEMENT.is_file(),
    reason="canonical Picasso holder/cup placement not on disk",
)
def test_real_canonical_picasso_holder_cup():
    err = _max_reconstruction_error_m(_PICASSO_PLACEMENT)
    assert err < 1e-12


def test_report_max_error_on_available_real_data():
    """Emit measurable max error for completion reporting (not a scientific gate)."""
    errors: list[tuple[str, float]] = []
    for label, path in (
        ("ycbv_000048", _YCBV_PLACEMENT),
        ("picasso_holder_cup", _PICASSO_PLACEMENT),
    ):
        if path.is_file():
            errors.append((label, _max_reconstruction_error_m(path)))
    if not errors:
        pytest.skip("no canonical placement files available")
    for _label, err in errors:
        assert err < 1e-12
