from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from geometry_to_physics.data import bop as bop_data
from geometry_to_physics.data import cad as cad_data
from geometry_to_physics.data import picasso as picasso_data
from geometry_to_physics.data.paths import gt_scene_dir
from geometry_to_physics.data.recon import ReconBundle
from geometry_to_physics.transforms import bop as bop_xf
from geometry_to_physics.transforms.se3 import as_homogeneous


@dataclass(frozen=True)
class GtChainResult:
    vertices_recon_world_m: np.ndarray
    frame_id: str
    view_index: int
    chain: str
    cad_path: str


def gt_vertices_recon_world(
    dataset: str,
    scene_id: str,
    object_id: str,
    recon: ReconBundle,
    view_index: int,
    picasso_mesh_name: str | None = None,
) -> GtChainResult:
    """CAD local -> GT camera (m2c) -> reconstruction world via recon c2w[view]."""
    if view_index < 0 or view_index >= recon.n_views:
        raise IndexError(view_index)
    frame_stem = recon.image_paths[view_index].stem
    c2w = recon.camera_poses[view_index]

    if dataset in ("ycbv", "tless"):
        cad = cad_data.load_cad_vertices(dataset, object_id, picasso_mesh_name)
        gdir = gt_scene_dir(dataset, scene_id)
        scene_gt, _ = bop_data.load_scene_json(gdir)
        fk = bop_data.frame_key(scene_gt, frame_stem)
        hit = bop_data.object_gt_entry(scene_gt, fk, object_id)
        m2c = bop_xf.m2c_transform(hit["cam_R_m2c"], hit["cam_t_m2c"])
        v_w = bop_xf.cad_local_to_recon_world(cad.vertices_m, m2c, c2w)
        return GtChainResult(
            vertices_recon_world_m=v_w,
            frame_id=fk,
            view_index=view_index,
            chain="cad_m2c_bop_mm_to_m_then_recon_c2w",
            cad_path=str(cad.path),
        )

    if dataset == "picasso":
        gdir = gt_scene_dir(dataset, scene_id)
        names = picasso_data.picasso_object_name_map(gdir)
        mn = picasso_mesh_name or names.get(str(object_id))
        if mn is None:
            raise KeyError(f"Picasso object_id {object_id} not in scene_gt name map")
        cad = cad_data.load_cad_vertices(dataset, object_id, mn)
        po = picasso_data.picasso_pose_object(gdir, frame_stem, object_id)
        R, t = picasso_data.picasso_m2c_matrix(po)
        m2c = as_homogeneous(R, t)
        v_w = bop_xf.cad_local_to_recon_world(cad.vertices_m, m2c, c2w)
        return GtChainResult(
            vertices_recon_world_m=v_w,
            frame_id=frame_stem,
            view_index=view_index,
            chain="cad_picasso_camera_m2c_then_recon_c2w",
            cad_path=str(cad.path),
        )

    raise ValueError(dataset)


def gt_vertices_bop_world(
    dataset: str,
    scene_id: str,
    object_id: str,
    frame_stem: str,
    picasso_mesh_name: str | None = None,
) -> np.ndarray:
    """Legacy BOP world frame (for diagnostic comparison only)."""
    gdir = gt_scene_dir(dataset, scene_id)
    cad = cad_data.load_cad_vertices(dataset, object_id, picasso_mesh_name)
    if dataset not in ("ycbv", "tless"):
        raise ValueError("BOP world only for ycbv/tless")
    scene_gt, scene_cam = bop_data.load_scene_json(gdir)
    fk = bop_data.frame_key(scene_gt, frame_stem)
    hit = bop_data.object_gt_entry(scene_gt, fk, object_id)
    cam = scene_cam[fk]
    c2w_bop = bop_xf.bop_w2c_to_c2w(cam["cam_R_w2c"], cam["cam_t_w2c"])
    m2c = bop_xf.m2c_transform(hit["cam_R_m2c"], hit["cam_t_m2c"])
    return bop_xf.cad_local_to_bop_world(cad.vertices_m, m2c, c2w_bop)
