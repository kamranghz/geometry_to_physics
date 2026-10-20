"""
Placement cen_table + hull_local to principal-axis body frame (TCSVT v2).

Preserves world-space table vertices when interpreting canonical placement_data fields.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geometry_to_physics.geometry.mass_properties import MassProperties, from_placement_object


@dataclass(frozen=True)
class BodyFramePlacement:
    """Rigid-body spawn data preserving world-space collision geometry."""

    object_id: str
    com_table: np.ndarray  # body origin in table frame
    quat_xyzw: np.ndarray  # principal axes as body->table rotation (PyBullet layout)
    hull_body: np.ndarray  # (N, 3) collision vertices in body frame
    mass_props: MassProperties
    v_table: np.ndarray  # (N, 3) reference world/table vertices

    def max_vertex_world_error_vs_v1(self, cen_table: np.ndarray, hull_local: np.ndarray) -> float:
        """Max |v_world_v2 - v_world_v1| using frozen v1 convention."""
        anchor = np.asarray(cen_table, dtype=np.float64)
        local = np.asarray(hull_local, dtype=np.float64)
        v1 = anchor + local
        R = self.mass_props.principal_axes
        v2 = self.com_table + (R @ self.hull_body.T).T
        return float(np.max(np.linalg.norm(v2 - v1, axis=1)))


def rotation_matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (body->world) to quaternion [x, y, z, w]."""
    R = np.asarray(R, dtype=np.float64)
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    q /= np.linalg.norm(q)
    return q


def body_frame_from_placement(
    object_id: str,
    cen_table: list[float],
    hull_local: list[list[float]],
    nominal_mass_kg: float = 1.0,
) -> BodyFramePlacement:
    mp = from_placement_object(cen_table, hull_local, nominal_mass_kg=nominal_mass_kg, object_id=object_id)
    anchor = np.asarray(cen_table, dtype=np.float64)
    local = np.asarray(hull_local, dtype=np.float64)
    v_table = anchor + local
    R = mp.principal_axes  # body basis columns in table frame
    hull_body = (R.T @ (v_table - mp.com).T).T
    quat = rotation_matrix_to_quat_xyzw(R)
    return BodyFramePlacement(
        object_id=object_id,
        com_table=mp.com.copy(),
        quat_xyzw=quat,
        hull_body=hull_body,
        mass_props=mp,
        v_table=v_table,
    )
