import json
from pathlib import Path

import yaml

from geometry_to_physics.validation.gate import run_validation_gate


def test_gate_engineering_schema(tmp_path: Path):
    cases = tmp_path / "cases.yaml"
    cases.write_text(yaml.dump({"cases": []}), encoding="utf-8")
    out = tmp_path / "out"
    report = run_validation_gate(cases, out)
    assert report["gate_version"] == "engineering-preflight-1"
    assert report["role"] == "engineering_preflight_only"
    assert "PASS" in report["summary"]
    assert (out / "validation_gate_report.json").is_file()
