#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, default=Path("configs/validation_cases.yaml"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/validation"))
    args = ap.parse_args()
    # Default local paths if unset
    os.environ.setdefault("GTP_GT_ROOT", r"C:\Users\kghol072\physical_fidelity\datasets")
    os.environ.setdefault("GTP_RECON_ROOT", r"C:\Users\kghol072\physical_fidelity\outputs")
    os.environ.setdefault("GTP_OUTPUT_ROOT", str(Path("outputs").resolve()))

    from geometry_to_physics.validation.gate import run_validation_gate

    report = run_validation_gate(args.cases, args.out_dir)
    print(f"summary: {report['summary']}")
    print(f"report: {(args.out_dir / 'validation_gate_report.json').resolve()}")


if __name__ == "__main__":
    main()
