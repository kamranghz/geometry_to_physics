from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from geometry_to_physics.data.paths import gt_scene_dir, recon_scene_dir
from geometry_to_physics.data.recon import load_recon_bundle
from geometry_to_physics.geometry.reference_vertices import (
    reference_vertices_table,
    scene_context_for_placement,
)
from geometry_to_physics.registration.spread_frames import spread_frame_ids, spread_view_index
from geometry_to_physics.transforms.bop import cad_local_to_recon_world, m2c_transform
from geometry_to_physics.transforms.placement_plane import PlacementPlane
from geometry_to_physics.transforms.se3 import as_homogeneous
from geometry_to_physics.transforms.table import plane_table_rotation_translation, world_to_table

GT = Path(r"C:\Users\kghol072\physical_fidelity\datasets")
RECON = Path(r"C:\Users\kghol072\physical_fidelity\outputs")
CANONICAL = Path(r"C:\Users\kghol072\PhysFidelity3D\outputs\cross_engine\full_local_rerun\canonical")
_LEGACY = Path(r"C:\Users\kghol072\PhysFidelity3D")

_YCBV_PLACEMENT = CANONICAL / "DUSt3R" / "ycbv_000048_spread_scaled" / "placement_data.json"
_PICASSO_PLACEMENT = CANONICAL / "DUSt3R" / "picasso_holder_cup_spread_scaled" / "placement_data.json"


def test_synthetic_cad_chain_table_frame():
    cad = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=np.float64)
    m2c = m2c_transform(np.eye(3).tolist(), [0.0, 0.0, 0.0])
    c2w = as_homogeneous(np.eye(3), np.array([0.5, 0.0, 0.0]))
    v_w = cad_local_to_recon_world(cad, m2c, c2w)
    plane = PlacementPlane(0.0, 0.0, 1.0, 0.0)
    R_tw, p0 = plane_table_rotation_translation(plane.a, plane.b, plane.c, plane.d)
    v_t = world_to_table(v_w, R_tw, p0)
    assert v_t.shape == (2, 3)
    assert np.allclose(v_w[0], [0.5, 0.0, 0.0])
    assert np.allclose(v_t, world_to_table(v_w, R_tw, p0))
    assert np.allclose(v_t[1, 0], 0.6)


def test_spread_view_index_anchor():
    spread = ["12", "48", "96"]
    assert spread_view_index(spread, "12") == 0
    assert spread_view_index(spread, "48") == 1
    assert spread_view_index(spread, "000048") == 1


@pytest.mark.skipif(not GT.is_dir(), reason="GT datasets missing")
def test_spread_frame_ids_ycbv_048():
    os.environ["GTP_GT_ROOT"] = str(GT)
    gdir = gt_scene_dir("ycbv", "000048")
    ids = spread_frame_ids("ycbv", "000048", gdir)
    assert len(ids) == 3
    assert ids[0] == min(ids, key=int)


@pytest.mark.skipif(
    not GT.is_dir() or not RECON.is_dir() or not _YCBV_PLACEMENT.is_file(),
    reason="YCB-V canonical/GT/recon missing",
)
def test_real_ycbv_000048_reference_vertices():
    os.environ["GTP_GT_ROOT"] = str(GT)
    os.environ["GTP_RECON_ROOT"] = str(RECON)
    pl = json.loads(_YCBV_PLACEMENT.read_text(encoding="utf-8"))
    ctx = scene_context_for_placement(pl)
    plane = PlacementPlane.from_placement_dict(pl)
    recon = load_recon_bundle(recon_scene_dir("DUSt3R", "ycbv_000048_spread_scaled"))
    v = reference_vertices_table(ctx, "1", plane, recon_poses=recon.camera_poses)
    assert v.ndim == 2 and v.shape[1] == 3
    assert v.shape[0] > 100
    assert np.all(np.isfinite(v))
    assert np.max(np.abs(v)) < 50.0


@pytest.mark.skipif(
    not GT.is_dir() or not RECON.is_dir() or not _PICASSO_PLACEMENT.is_file(),
    reason="Picasso canonical/GT/recon missing",
)
def test_real_picasso_holder_cup_reference_vertices():
    os.environ["GTP_GT_ROOT"] = str(GT)
    os.environ["GTP_RECON_ROOT"] = str(RECON)
    pl = json.loads(_PICASSO_PLACEMENT.read_text(encoding="utf-8"))
    ctx = scene_context_for_placement(pl)
    plane = PlacementPlane.from_placement_dict(pl)
    recon = load_recon_bundle(recon_scene_dir("DUSt3R", "picasso_holder_cup_spread_scaled"))
    oid = next(iter(pl["objects"]))
    v = reference_vertices_table(ctx, oid, plane, recon_poses=recon.camera_poses)
    assert v.ndim == 2 and v.shape[1] == 3
    assert v.shape[0] > 10
    assert np.all(np.isfinite(v))


@pytest.mark.skipif(
    not _LEGACY.is_dir() or not GT.is_dir() or not RECON.is_dir() or not _YCBV_PLACEMENT.is_file(),
    reason="legacy or data missing",
)
def test_legacy_parity_ycbv_000048_obj1():
    os.environ["GTP_GT_ROOT"] = str(GT)
    os.environ["GTP_RECON_ROOT"] = str(RECON)
    sys.path.insert(0, str(_LEGACY))
    from cross_engine.reference_anchor.gt_geometry import SceneContext as LegacyCtx
    from cross_engine.reference_anchor.gt_geometry import gt_scene_dir as legacy_gt_scene_dir
    from cross_engine.reference_anchor.gt_geometry import spread_frame_ids as legacy_spread
    from cross_engine.reference_anchor.objects import reference_vertices_table as legacy_ref
    from cross_engine.v2.ground_plane import PlacementPlane as LegacyPlane

    pl = json.loads(_YCBV_PLACEMENT.read_text(encoding="utf-8"))
    plane = LegacyPlane.from_placement_dict(pl)
    dataset = "ycbv"
    scene_id = "000048"
    gdir = legacy_gt_scene_dir(dataset, scene_id)
    anchor = legacy_spread(dataset, scene_id, gdir)[0]
    ctx_l = LegacyCtx(dataset, scene_id, gdir, anchor)
    poses = load_recon_bundle(recon_scene_dir("DUSt3R", "ycbv_000048_spread_scaled")).camera_poses
    v_legacy = legacy_ref(ctx_l, "1", plane, recon_poses=poses)

    ctx_n = scene_context_for_placement(pl)
    plane_n = PlacementPlane.from_placement_dict(pl)
    v_new = reference_vertices_table(ctx_n, "1", plane_n, recon_poses=poses)

    max_err = float(np.max(np.linalg.norm(v_legacy - v_new, axis=1)))
    assert max_err < 1e-9
