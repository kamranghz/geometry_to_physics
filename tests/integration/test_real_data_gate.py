import os
from pathlib import Path

import pytest

GT = Path(r"C:\Users\kghol072\physical_fidelity\datasets")
RECON = Path(r"C:\Users\kghol072\physical_fidelity\outputs")
CANONICAL = Path(r"C:\Users\kghol072\PhysFidelity3D\outputs\cross_engine\full_local_rerun\canonical")


@pytest.mark.skipif(not GT.is_dir() or not RECON.is_dir(), reason="local datasets missing")
def test_validation_gate_engineering_preflight():
    os.environ["GTP_GT_ROOT"] = str(GT)
    os.environ["GTP_RECON_ROOT"] = str(RECON)
    if CANONICAL.is_dir():
        os.environ["GTP_CANONICAL_ROOT"] = str(CANONICAL)
    from geometry_to_physics.validation.gate import run_validation_gate

    out = Path("outputs/test_validation")
    report = run_validation_gate(Path("configs/validation_cases.yaml"), out)
    assert report["role"] == "engineering_preflight_only"
    assert len(report["cases"]) == 5
    assert report["summary"]["PASS"] + report["summary"]["FAIL"] == 5
    y048 = next(c for c in report["cases"] if c["case_id"] == "ycbv_dust3r_048")
    assert y048["overall"] == "PASS"
    assert y048["checks"]["gt_transform_sanity"]["status"] == "PASS"
