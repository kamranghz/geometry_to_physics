#!/usr/bin/env python3
"""Write numeric multiview center table (inspectable evidence, not a validity claim)."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import yaml

from geometry_to_physics.data.recon import load_recon_bundle
from geometry_to_physics.registration.multiview import multiview_gt_center_consistency


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=Path, default=Path("configs/validation_cases.yaml"))
    ap.add_argument("--out", type=Path, default=Path("outputs/validation/multiview_centers.csv"))
    args = ap.parse_args()
    os.environ.setdefault("GTP_GT_ROOT", r"C:\Users\kghol072\physical_fidelity\datasets")
    os.environ.setdefault("GTP_RECON_ROOT", r"C:\Users\kghol072\physical_fidelity\outputs")
    cases = yaml.safe_load(args.cases.read_text(encoding="utf-8"))["cases"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "case_id",
                "view_index",
                "center_x_m",
                "center_y_m",
                "center_z_m",
                "max_pairwise_mm",
                "static_assumption",
            ]
        )
        for c in cases:
            recon = load_recon_bundle(
                Path(os.environ["GTP_RECON_ROOT"]) / c["method"] / c["scene_folder"]
            )
            rep = multiview_gt_center_consistency(c["dataset"], c["scene_id"], c["object_id"], recon)
            for vi, cen in enumerate(rep.per_view_center_m):
                w.writerow(
                    [
                        c["id"],
                        vi,
                        cen[0],
                        cen[1],
                        cen[2],
                        rep.max_pairwise_mm,
                        rep.static_object_assumption,
                    ]
                )
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()
