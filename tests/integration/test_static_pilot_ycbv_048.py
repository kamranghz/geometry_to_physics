import os
from pathlib import Path

import numpy as np
import pytest

from geometry_to_physics.evaluation.static_pilot import (
    ANCHOR_SENSITIVITY_METRIC_KEYS,
    evaluate_canonical_static_scene,
    evaluate_static_anchor_sensitivity,
)

GT = Path(r"C:\Users\kghol072\physical_fidelity\datasets")
RECON = Path(r"C:\Users\kghol072\physical_fidelity\outputs")
PLACEMENT = Path(
    r"C:\Users\kghol072\PhysFidelity3D\outputs\cross_engine\full_local_rerun\canonical"
    r"\DUSt3R\ycbv_000048_spread_scaled\placement_data.json"
)


@pytest.mark.skipif(
    not GT.is_dir() or not RECON.is_dir() or not PLACEMENT.is_file(),
    reason="real GT, recon, or canonical placement missing",
)
def test_static_pilot_dust3r_ycbv_048_real_data_no_mocks():
    os.environ["GTP_GT_ROOT"] = str(GT)
    os.environ["GTP_RECON_ROOT"] = str(RECON)
    report = evaluate_canonical_static_scene(PLACEMENT, "DUSt3R", "ycbv_000048_spread_scaled")
    assert report["n_objects"] == 5
    assert report["dataset"] == "ycbv"
    for oid, m in report["objects"].items():
        assert m.get("status") == "ok", f"object {oid}: {m}"
    for m in report["objects"].values():
        assert m["n_vertices_rec"] > 0
        assert m["n_vertices_ref"] > 100
        for key in (
            "hull_chamfer_m",
            "volume_rel_err",
            "com_displacement_m",
            "inertia_frob_rel_err",
        ):
            assert np.isfinite(m[key])
        if m["hull_chamfer_cuda_m"] is not None:
            assert m["chamfer_cpu_cuda_abs_diff_m"] is not None
            assert m["chamfer_cpu_cuda_abs_diff_m"] < 1e-4


@pytest.mark.skipif(
    not GT.is_dir() or not RECON.is_dir() or not PLACEMENT.is_file(),
    reason="real GT, recon, or canonical placement missing",
)
def test_static_anchor_sensitivity_dust3r_ycbv_048_real_data():
    os.environ["GTP_GT_ROOT"] = str(GT)
    os.environ["GTP_RECON_ROOT"] = str(RECON)
    baseline = evaluate_canonical_static_scene(PLACEMENT, "DUSt3R", "ycbv_000048_spread_scaled")
    report = evaluate_static_anchor_sensitivity(
        PLACEMENT, "DUSt3R", "ycbv_000048_spread_scaled", include_cuda_on_primary=True
    )
    assert report["n_spread_anchors"] == 3
    assert len(report["spread_frame_ids"]) == 3
    assert report["primary_anchor_frame_id"] == report["spread_frame_ids"][0]

    for fid in report["spread_frame_ids"]:
        block = report["sensitivity_by_anchor"][fid]
        assert block["anchor_frame_id"] == fid
        assert len(block["objects"]) == 5

    for oid, m in report["primary"]["objects"].items():
        assert m.get("status") == "ok", f"primary object {oid}: {m}"
        base = baseline["objects"][oid]
        assert base.get("status") == "ok"
        for key in ANCHOR_SENSITIVITY_METRIC_KEYS:
            assert m[key] == base[key], f"object {oid} primary mismatch on {key}"
        assert m["n_vertices_rec"] == base["n_vertices_rec"]
        assert m.get("hull_chamfer_cuda_m") == base.get("hull_chamfer_cuda_m")
        assert m.get("chamfer_cpu_cuda_abs_diff_m") == base.get("chamfer_cpu_cuda_abs_diff_m")

    for fid in report["spread_frame_ids"]:
        for oid, m in report["sensitivity_by_anchor"][fid]["objects"].items():
            if m.get("status") != "ok":
                continue
            assert m["n_vertices_rec"] > 0
            for key in ANCHOR_SENSITIVITY_METRIC_KEYS:
                assert np.isfinite(m[key]), f"anchor {fid} object {oid} {key}"

    primary_fid = report["primary_anchor_frame_id"]
    primary_obj = report["primary"]["objects"]["1"]
    if primary_obj.get("hull_chamfer_cuda_m") is not None:
        assert primary_obj["chamfer_cpu_cuda_abs_diff_m"] is not None
        assert primary_obj["chamfer_cpu_cuda_abs_diff_m"] < 1e-4
    for fid in report["spread_frame_ids"]:
        if fid == primary_fid:
            continue
        alt = report["sensitivity_by_anchor"][fid]["objects"]["1"]
        assert alt.get("hull_chamfer_cuda_m") is None
