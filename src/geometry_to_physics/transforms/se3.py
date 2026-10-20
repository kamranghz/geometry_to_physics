from __future__ import annotations

import numpy as np


def as_homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def apply(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1))
    return (np.hstack([pts, ones]) @ np.asarray(T, dtype=np.float64).T)[:, :3]
