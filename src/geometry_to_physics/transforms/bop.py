from __future__ import annotations

import numpy as np

from geometry_to_physics.transforms.se3 import apply, as_homogeneous


def bop_w2c_to_c2w(cam_R_w2c: list, cam_t_w2c_mm: list) -> np.ndarray:
    """BOP scene_camera: world-to-camera, translation in mm -> 4x4 c2w in metres."""
    R_w2c = np.asarray(cam_R_w2c, dtype=np.float64).reshape(3, 3)
    t_w2c = np.asarray(cam_t_w2c_mm, dtype=np.float64).reshape(3) / 1000.0
    R_c2w = R_w2c.T
    t_c2w = -R_c2w @ t_w2c
    return as_homogeneous(R_c2w, t_c2w)


def m2c_transform(cam_R_m2c: list, cam_t_m2c_mm: list) -> np.ndarray:
    R = np.asarray(cam_R_m2c, dtype=np.float64).reshape(3, 3)
    t = np.asarray(cam_t_m2c_mm, dtype=np.float64).reshape(3) / 1000.0
    return as_homogeneous(R, t)


def cad_local_to_bop_world(verts_m: np.ndarray, m2c: np.ndarray, c2w_bop: np.ndarray) -> np.ndarray:
    v_cam = apply(m2c, verts_m)
    return apply(c2w_bop, v_cam)


def cad_local_to_recon_world(verts_m: np.ndarray, m2c: np.ndarray, c2w_recon: np.ndarray) -> np.ndarray:
    v_cam = apply(m2c, verts_m)
    return apply(c2w_recon, v_cam)
