from __future__ import annotations

import numpy as np


def box_inertia(mass: float, sx: float, sy: float, sz: float) -> np.ndarray:
    """Inertia about center for axis-aligned box with side lengths sx,sy,sz."""
    ix = mass * (sy**2 + sz**2) / 12.0
    iy = mass * (sx**2 + sz**2) / 12.0
    iz = mass * (sx**2 + sy**2) / 12.0
    return np.diag([ix, iy, iz]).astype(np.float64)


def sphere_inertia(mass: float, radius: float) -> np.ndarray:
    i = 0.4 * mass * radius**2
    return np.diag([i, i, i]).astype(np.float64)


def rotate_inertia(I_body: np.ndarray, R_world_from_body: np.ndarray) -> np.ndarray:
    R = np.asarray(R_world_from_body, dtype=np.float64).reshape(3, 3)
    I = np.asarray(I_body, dtype=np.float64).reshape(3, 3)
    return R @ I @ R.T


def com_uniform_vertices(vertices: np.ndarray) -> np.ndarray:
    return np.asarray(vertices, dtype=np.float64).mean(0)
