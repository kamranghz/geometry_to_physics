#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from geometry_to_physics.provenance.inventory import build_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, default=Path("configs/validation_cases.yaml"))
    ap.add_argument("--out", type=Path, default=Path("audit/provenance_manifest.json"))
    args = ap.parse_args()
    build_manifest(args.cases, args.out)
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
