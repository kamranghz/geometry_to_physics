from __future__ import annotations

import numpy as np

from geometry_to_physics.transforms.placement_body import body_frame_from_placement


def recon_vertices_table(cen_table: list, hull_local: list) -> np.ndarray:
    bf = body_frame_from_placement("x", cen_table, hull_local)
    R = bf.mass_props.principal_axes
    return bf.com_table + (R @ np.asarray(hull_local).T).T
