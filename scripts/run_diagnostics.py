#!/usr/bin/env python3
"""Optional scientific diagnostics (not engineering gate)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, default=Path("configs/validation_cases.yaml"))
    ap.add_argument("--out", type=Path, default=Path("outputs/diagnostics/report.json"))
    args = ap.parse_args()
    os.environ.setdefault("GTP_GT_ROOT", r"C:\Users\kghol072\physical_fidelity\datasets")
    os.environ.setdefault("GTP_RECON_ROOT", r"C:\Users\kghol072\physical_fidelity\outputs")
    from geometry_to_physics.validation.diagnostics import run_optional_diagnostics

    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = run_optional_diagnostics(args.cases)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
