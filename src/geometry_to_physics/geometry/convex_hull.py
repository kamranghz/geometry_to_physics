from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull


def convex_hull_vertices(points: np.ndarray, max_verts: int = 256) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 4:
        raise ValueError("need >=4 points for hull")
    hull = ConvexHull(pts)
    hv = pts[hull.vertices]
    if len(hv) > max_verts:
        idx = np.linspace(0, len(hv) - 1, max_verts, dtype=int)
        hv = hv[idx]
    return hv
