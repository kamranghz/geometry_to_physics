from __future__ import annotations

import numpy as np

from geometry_to_physics.transforms.se3 import as_homogeneous


def plane_table_rotation_translation(a: float, b: float, c: float, d: float) -> tuple[np.ndarray, np.ndarray]:
    """Map reconstruction-world points to table frame: z up, origin on plane."""
    norm = float(np.sqrt(a * a + b * b + c * c))
    n = np.array([a, b, c], dtype=np.float64) / norm
    p0 = -d / norm * n
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, z)
    s = float(np.linalg.norm(v))
    co = float(np.dot(n, z))
    if s < 1e-10:
        R = np.eye(3) if co > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
        R = np.eye(3) + vx + (vx @ vx) * ((1.0 - co) / (s**2))
    return R.astype(np.float64), p0.astype(np.float64)


def world_to_table(pts_w: np.ndarray, R_tw: np.ndarray, p0: np.ndarray) -> np.ndarray:
    return (R_tw @ (np.asarray(pts_w, dtype=np.float64) - p0).T).T


def table_transform_matrix(R_tw: np.ndarray, p0: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_tw
    T[:3, 3] = -R_tw @ p0
    return T
