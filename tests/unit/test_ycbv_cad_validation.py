from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from geometry_to_physics.data.cad import (
    _validate_ycbv_mesh_aabb_against_models_info,
    load_cad_vertices,
)

GT = Path(r"C:\Users\kghol072\physical_fidelity\datasets")
PILOT_OBJECTS = ("1", "6", "14", "19", "20")


@pytest.mark.skipif(not GT.is_dir(), reason="YCB-V datasets missing")
def test_ycbv_pilot_objects_load_and_match_models_info_sizes():
    os.environ["GTP_GT_ROOT"] = str(GT)
    mi = json.loads((GT / "YCBV" / "ycbv_models" / "models" / "models_info.json").read_text())
    for oid in PILOT_OBJECTS:
        res = load_cad_vertices("ycbv", oid)
        info = mi[oid]
        expected_m = np.array([info["size_x"], info["size_y"], info["size_z"]]) / 1000.0
        actual_m = res.vertices_m.max(0) - res.vertices_m.min(0)
        max_diff = float(np.max(np.abs(actual_m - expected_m)))
        assert max_diff <= 0.001
        assert res.vertices_m.shape[0] > 1000


def test_ycbv_rejects_inconsistent_models_info_aabb():
    info = {"size_x": 100.0, "size_y": 100.0, "size_z": 100.0}
    verts_m = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.0, 0.2],
        ]
    )
    with pytest.raises(ValueError, match="max axis discrepancy"):
        _validate_ycbv_mesh_aabb_against_models_info(verts_m, info, "test")
