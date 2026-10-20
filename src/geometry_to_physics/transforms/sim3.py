from __future__ import annotations

import numpy as np

from geometry_to_physics.transforms.se3 import as_homogeneous, invert


def scale_matrix(s: float) -> np.ndarray:
    return np.diag([s, s, s, 1.0]).astype(np.float64)


def apply_sim3(T: np.ndarray, s: float, points: np.ndarray) -> np.ndarray:
    """Apply similarity: x' = s * (R x + t) with 4x4 T."""
    pts = np.asarray(points, dtype=np.float64)
    R, t = T[:3, :3], T[:3, 3]
    return (s * (pts @ R.T)) + t


def compose_sim3(T2: np.ndarray, s2: float, T1: np.ndarray, s1: float) -> tuple[np.ndarray, float]:
    """Compose (T2,s2) after (T1,s1) on column vectors: x2 = s2 R2 (s1 R1 x + t1) + t2."""
    R1, t1 = T1[:3, :3], T1[:3, 3]
    R2, t2 = T2[:3, :3], T2[:3, 3]
    s = s2 * s1
    R = R2 @ R1
    t = s2 * (R2 @ t1) + t2
    return as_homogeneous(R, t), s


def roundtrip_error(T: np.ndarray, s: float, points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=np.float64)
    mid = apply_sim3(T, s, pts)
    back = apply_sim3(invert(T), 1.0 / s, mid)
    return float(np.max(np.linalg.norm(back - pts, axis=1)))
