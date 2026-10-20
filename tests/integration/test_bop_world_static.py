import os
from pathlib import Path

import pytest

from geometry_to_physics.data.paths import recon_scene_dir
from geometry_to_physics.data.recon import load_recon_bundle
from geometry_to_physics.registration.bop_world import bop_world_object_center_consistency


GT = Path(r"C:\Users\kghol072\physical_fidelity\datasets")
RECON = Path(r"C:\Users\kghol072\physical_fidelity\outputs")


@pytest.mark.skipif(not GT.is_dir(), reason="datasets missing")
def test_ycbv_048_bop_world_multiview_spread_regression():
    """BOP-only spread on spread frames (~21mm); documents annotation residual, not recon error."""
    os.environ["GTP_GT_ROOT"] = str(GT)
    os.environ["GTP_RECON_ROOT"] = str(RECON)
    recon = load_recon_bundle(recon_scene_dir("DUSt3R", "ycbv_000048_spread_scaled"))
    rep = bop_world_object_center_consistency("ycbv", "000048", "1", recon, threshold_mm=1.0)
    assert rep.static_object_assumption == "FAIL"
    assert rep.max_pairwise_mm is not None
    assert 15.0 < rep.max_pairwise_mm < 30.0
