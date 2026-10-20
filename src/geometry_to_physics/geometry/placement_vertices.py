"""
Table-frame collision vertices from placement_data.json fields.

Two explicit conventions (call site chooses; no automatic detection):

- **Canonical v1 (physics prep):** ``cen_table`` is centroid in table frame;
  ``hull_local`` rows are table-axis offsets. World/table vertices are
  ``cen_table + hull_local``.

- **Principal-axis v2 (reference-anchor encoding):** ``cen_table`` is convex-hull
  CoM; ``hull_local`` rows are body-frame coordinates. Use ``recon_vertices_table``.
"""
from __future__ import annotations

import numpy as np

from geometry_to_physics.transforms.placement_body import body_frame_from_placement


def table_vertices_from_canonical_placement(
    cen_table: list[float] | np.ndarray,
    hull_local: list[list[float]] | np.ndarray,
) -> np.ndarray:
    """
    Expand canonical v1 reconstruction placement to table-frame vertices (metres).

    ``vertices[i] = cen_table + hull_local[i]`` with no principal-axis rotation.
    """
    cen = np.asarray(cen_table, dtype=np.float64).reshape(3)
    local = np.asarray(hull_local, dtype=np.float64)
    if local.ndim != 2 or local.shape[1] != 3:
        raise ValueError(f"hull_local expected shape (N, 3), got {local.shape}")
    if local.shape[0] == 0:
        raise ValueError("hull_local must contain at least one vertex")
    if not np.all(np.isfinite(cen)) or not np.all(np.isfinite(local)):
        raise ValueError("cen_table and hull_local must be finite")
    return cen + local


def recon_vertices_table(cen_table: list, hull_local: list) -> np.ndarray:
    """
    Table-frame vertices from principal-axis v2 placement fields.

    Expects ``hull_local`` in the body frame of ``mass_props.principal_axes`` with
    ``cen_table`` the corresponding CoM (as produced by reference-anchor
    ``placement_fields_from_table_vertices``). Do not use for canonical v1 physics
    prep JSON; use ``table_vertices_from_canonical_placement`` instead.
    """
    bf = body_frame_from_placement("x", cen_table, hull_local)
    R = bf.mass_props.principal_axes
    return bf.com_table + (R @ np.asarray(hull_local).T).T
