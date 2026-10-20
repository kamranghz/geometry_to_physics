from __future__ import annotations

import numpy as np
import trimesh


def mesh_volume_com_inertia(vertices: np.ndarray, faces: np.ndarray, density: float = 1.0) -> dict:
    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    tm.density = density
    mp = tm.mass_properties
    return {
        "volume_m3": float(tm.volume),
        "mass_kg": float(mp["mass"]),
        "com_m": np.asarray(mp["center_mass"], dtype=np.float64).tolist(),
        "inertia_kg_m2": np.asarray(mp["inertia"], dtype=np.float64).tolist(),
    }
