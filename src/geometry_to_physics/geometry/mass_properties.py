"""
Convex-hull mass properties for collision-proxy polyhedra (TCSVT v2 policy).

Nominal mass = 1 kg; uniform density rho = m / V. Inertia about CoM in SI; principal axes via eigh.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import ConvexHull


class MassPropertiesError(ValueError):
    """Invalid or degenerate geometry for mass-property integration."""


@dataclass(frozen=True)
class MassProperties:
    mass_kg: float
    com: np.ndarray  # (3,) in input vertex frame
    inertia_com: np.ndarray  # (3, 3) symmetric, kg·m² about CoM
    principal_moments: np.ndarray  # (3,) ascending eigenvalues
    principal_axes: np.ndarray  # (3, 3) columns = body x,y,z in input frame; det = +1
    volume_m3: float
    density_kg_m3: float
    provenance: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "mass_kg": float(self.mass_kg),
            "com": self.com.tolist(),
            "inertia_com": self.inertia_com.tolist(),
            "principal_moments": self.principal_moments.tolist(),
            "principal_axes": self.principal_axes.tolist(),
            "volume_m3": float(self.volume_m3),
            "density_kg_m3": float(self.density_kg_m3),
            "provenance": self.provenance,
        }


def _principal_frame(I_com: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w, R = np.linalg.eigh(I_com)
    if R.shape != (3, 3):
        raise MassPropertiesError("eigendecomposition failed")
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1.0
    return w, R


def from_convex_hull_vertices(
    hull_vertices: np.ndarray,
    nominal_mass_kg: float = 1.0,
    provenance: dict[str, Any] | None = None,
) -> MassProperties:
    """
    Integrate on the convex hull of ``hull_vertices`` (metres, SI).

    Raises MassPropertiesError on degenerate (near-zero volume) geometry.
    """
    verts = np.asarray(hull_vertices, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise MassPropertiesError(f"expected (N, 3) vertices, got {verts.shape}")
    if verts.shape[0] < 4:
        raise MassPropertiesError("need at least 4 hull vertices")

    try:
        hull = ConvexHull(verts)
    except Exception as exc:
        raise MassPropertiesError(f"ConvexHull failed: {exc}") from exc

    if hull.volume <= 1e-12:
        raise MassPropertiesError(f"degenerate hull volume {hull.volume}")

    mesh = trimesh.convex.convex_hull(verts)
    if mesh is None or mesh.is_empty:
        raise MassPropertiesError("convex_hull mesh construction failed")
    vol = float(abs(mesh.volume))
    if vol <= 1e-12:
        raise MassPropertiesError(f"degenerate mesh volume {vol}")
    if abs(vol - hull.volume) > max(1e-9, 1e-5 * hull.volume):
        raise MassPropertiesError(f"volume mismatch: mesh={vol} hull={hull.volume}")

    rho = nominal_mass_kg / vol
    mass = nominal_mass_kg
    com = np.asarray(mesh.center_mass, dtype=np.float64)
    mesh.density = rho
    I_com = np.asarray(mesh.moment_inertia, dtype=np.float64)

    evals, axes = _principal_frame(I_com)
    if np.any(evals < -1e-9) or np.any(~np.isfinite(evals)):
        raise MassPropertiesError(f"invalid principal moments: {evals}")

    prov = dict(provenance or {})
    prov.setdefault("integration", "convex_hull_trimesh")
    prov.setdefault("n_hull_vertices", int(verts.shape[0]))
    prov.setdefault("n_faces", int(hull.simplices.shape[0]))

    return MassProperties(
        mass_kg=mass,
        com=com,
        inertia_com=I_com,
        principal_moments=evals,
        principal_axes=axes,
        volume_m3=float(vol),
        density_kg_m3=float(rho),
        provenance=prov,
    )


def from_placement_object(
    cen_table: list[float],
    hull_local: list[list[float]],
    nominal_mass_kg: float = 1.0,
    object_id: str | None = None,
) -> MassProperties:
    """Mass properties in table frame from canonical placement fields."""
    anchor = np.asarray(cen_table, dtype=np.float64)
    local = np.asarray(hull_local, dtype=np.float64)
    v_table = anchor + local
    prov = {
        "source": "placement_data.json",
        "object_id": object_id,
        "anchor": "cen_table",
        "collision_proxy": "hull_local",
    }
    return from_convex_hull_vertices(v_table, nominal_mass_kg=nominal_mass_kg, provenance=prov)
