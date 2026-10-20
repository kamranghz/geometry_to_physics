from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from geometry_to_physics.config import load_paths
from geometry_to_physics.data.obj_geom import load_obj_vertices


@dataclass(frozen=True)
class CadLoadResult:
    vertices_m: np.ndarray
    path: Path
    unit_source: str
    models_info_diameter_mm: float | None


def _ycbv_models_info() -> dict:
    p = load_paths().gt_root / "YCBV" / "ycbv_models" / "models" / "models_info.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _extent_m(verts: np.ndarray) -> float:
    return float(np.linalg.norm(verts.max(0) - verts.min(0)))


def load_cad_vertices(dataset: str, obj_id: str, picasso_mesh_name: str | None = None) -> CadLoadResult:
    """Load CAD mesh; units from dataset metadata, not inferred from bbox alone."""
    gt = load_paths().gt_root
    diameter_mm: float | None = None
    if dataset == "ycbv":
        mesh_path = gt / "YCBV" / "ycbv_models" / "models" / f"obj_{int(obj_id):06d}.ply"
        info = _ycbv_models_info().get(str(int(obj_id)))
        if info:
            diameter_mm = float(info["diameter"])
        unit_source = "ycbv_models_info_mm_vertex_coords"
    elif dataset == "tless":
        candidates = [
            gt / "TLESS" / "models_cad" / f"obj_{int(obj_id):06d}.ply",
            gt / "TLESS" / "models_eval" / f"obj_{int(obj_id):06d}.ply",
            gt / "TLESS" / "tless_models" / "models" / f"obj_{int(obj_id):06d}.ply",
        ]
        mesh_path = next((p for p in candidates if p.is_file()), None)
        if mesh_path is None:
            raise FileNotFoundError(f"TLESS CAD missing for obj {obj_id}")
        unit_source = "tless_ply_mm_vertex_coords"
    elif dataset == "picasso":
        mn = picasso_mesh_name or obj_id
        if str(mn).startswith("jenga"):
            mn = "jenga"
        mesh_path = gt / "Picasso" / "model" / mn / f"{mn}.obj"
        unit_source = "picasso_obj_mm_if_extent_large"
    else:
        raise ValueError(dataset)

    if not mesh_path.is_file():
        raise FileNotFoundError(mesh_path)

    if mesh_path.suffix.lower() == ".obj":
        verts = load_obj_vertices(mesh_path)
    else:
        tm = trimesh.load(str(mesh_path), force="mesh", process=False)
        verts = np.asarray(tm.vertices, dtype=np.float64)

    if dataset == "ycbv" and diameter_mm is not None:
        ext_mm = _extent_m(verts)
        if ext_mm > 5.0:
            verts = verts / 1000.0
        expected_m = diameter_mm / 1000.0
        if abs(_extent_m(verts) - expected_m) / expected_m > 0.35:
            raise ValueError(
                f"YCB-V obj {obj_id}: extent {_extent_m(verts):.4f} m vs models_info diameter {expected_m:.4f} m"
            )
    elif dataset == "tless":
        if _extent_m(verts) > 5.0:
            verts = verts / 1000.0
    elif dataset == "picasso":
        if np.linalg.norm(verts.std(0)) > 5.0:
            verts = verts / 1000.0

    return CadLoadResult(vertices_m=verts, path=mesh_path, unit_source=unit_source, models_info_diameter_mm=diameter_mm)
