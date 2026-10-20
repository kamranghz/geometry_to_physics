from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    gt_root: Path
    recon_root: Path
    output_root: Path
    canonical_root: Path | None


def load_paths() -> Paths:
    canon = os.environ.get("GTP_CANONICAL_ROOT")
    return Paths(
        gt_root=Path(os.environ.get("GTP_GT_ROOT", ".")).resolve(),
        recon_root=Path(os.environ.get("GTP_RECON_ROOT", ".")).resolve(),
        output_root=Path(os.environ.get("GTP_OUTPUT_ROOT", "outputs")).resolve(),
        canonical_root=Path(canon).resolve() if canon else None,
    )
