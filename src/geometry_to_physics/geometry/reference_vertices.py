from __future__ import annotations

import numpy as np

from geometry_to_physics.data import bop as bop_data
from geometry_to_physics.data import cad as cad_data
from geometry_to_physics.data import picasso as picasso_data
from geometry_to_physics.data.paths import gt_scene_dir
from geometry_to_physics.registration.spread_frames import SceneContext, spread_frame_ids, spread_view_index
from geometry_to_physics.transforms import bop as bop_xf
from geometry_to_physics.transforms.placement_plane import PlacementPlane
from geometry_to_physics.transforms.se3 import as_homogeneous
from geometry_to_physics.transforms.table import plane_table_rotation_translation, world_to_table


def _cad_vertices_recon_world(
    ctx: SceneContext,
    object_id: str,
    frame_id: str,
    c2w: np.ndarray,
) -> np.ndarray:
    """CAD local -> GT m2c -> reconstruction world (metres)."""
    if ctx.dataset in ("ycbv", "tless"):
        cad = cad_data.load_cad_vertices(ctx.dataset, object_id, None)
        scene_gt, _ = bop_data.load_scene_json(ctx.gt_scene_dir)
        fk = str(int(frame_id)) if str(frame_id).isdigit() else str(frame_id)
        hit = bop_data.object_gt_entry(scene_gt, fk, object_id)
        m2c = bop_xf.m2c_transform(hit["cam_R_m2c"], hit["cam_t_m2c"])
        return bop_xf.cad_local_to_recon_world(cad.vertices_m, m2c, c2w)

    if ctx.dataset == "picasso":
        names = picasso_data.picasso_object_name_map(ctx.gt_scene_dir)
        mesh_name = names.get(str(object_id))
        if mesh_name is None:
            raise KeyError(f"Picasso object_id {object_id} not in scene_gt name map")
        cad = cad_data.load_cad_vertices(ctx.dataset, object_id, mesh_name)
        po = picasso_data.picasso_pose_object(ctx.gt_scene_dir, str(frame_id), object_id)
        R, t = picasso_data.picasso_m2c_matrix(po)
        m2c = as_homogeneous(R, t)
        return bop_xf.cad_local_to_recon_world(cad.vertices_m, m2c, c2w)

    raise ValueError(ctx.dataset)


def reference_vertices_table(
    ctx: SceneContext,
    object_id: str,
    plane: PlacementPlane,
    frame_id: str | None = None,
    *,
    recon_poses: np.ndarray | None = None,
) -> np.ndarray:
    """
    GT CAD vertices in reconstruction-derived table frame.

    Chain: CAD -> scene_gt m2c -> recon camera c2w[spread view] -> table plane.
    Anchor frame is the first spread id unless ``frame_id`` is set.
    ``recon_poses`` must align with spread indices (legacy reference_anchor convention).
    """
    if recon_poses is None:
        raise ValueError("reference_vertices_table requires recon camera_poses")

    spread_ids = spread_frame_ids(ctx.dataset, ctx.scene_id, ctx.gt_scene_dir)
    fid = frame_id or spread_ids[0]
    view_idx = spread_view_index(spread_ids, fid)
    c2w = np.asarray(recon_poses[view_idx], dtype=np.float64)

    v_w = _cad_vertices_recon_world(ctx, object_id, fid, c2w)
    R_tw, p0 = plane_table_rotation_translation(plane.a, plane.b, plane.c, plane.d)
    return world_to_table(v_w, R_tw, p0)


def scene_context_for_placement(placement: dict) -> SceneContext:
    """Build SceneContext from canonical placement_data.json header fields."""
    from geometry_to_physics.registration.spread_frames import scene_id_from_placement

    dataset = placement["dataset"]
    scene_id = scene_id_from_placement(placement)
    gdir = gt_scene_dir(dataset, scene_id)
    anchor = spread_frame_ids(dataset, scene_id, gdir)[0]
    return SceneContext(dataset=dataset, scene_id=scene_id, gt_scene_dir=gdir, anchor_frame_id=anchor)
