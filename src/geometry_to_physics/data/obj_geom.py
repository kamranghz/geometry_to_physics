from __future__ import annotations

from pathlib import Path

import numpy as np


def load_obj_vertices(path: Path) -> np.ndarray:
    """Parse OBJ vertex lines only (no MTL/texture loads)."""
    verts: list[list[float]] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not verts:
        raise ValueError(f"No vertices in OBJ: {path}")
    return np.asarray(verts, dtype=np.float64)
