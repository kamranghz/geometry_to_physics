"""Single-scene static geometry evaluation (pilot; not the full 7×42 matrix)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from geometry_to_physics.data.paths import recon_scene_dir
from geometry_to_physics.data.recon import load_recon_bundle
from geometry_to_physics.geometry.chamfer import chamfer_m_cuda
from geometry_to_physics.geometry.object_metrics import compare_static_objects
from geometry_to_physics.geometry.placement_vertices import table_vertices_from_canonical_placement
from geometry_to_physics.geometry.reference_vertices import reference_vertices_table, scene_context_for_placement
from geometry_to_physics.registration.spread_frames import SceneContext, spread_frame_ids
from geometry_to_physics.transforms.placement_plane import PlacementPlane

# Scalar metrics compared across spread anchors (recon-side counts excluded).
ANCHOR_SENSITIVITY_METRIC_KEYS: tuple[str, ...] = (
    "hull_chamfer_m",
    "volume_rec_m3",
    "volume_ref_m3",
    "volume_abs_err_m3",
    "volume_rel_err",
    "com_displacement_m",
    "principal_moment_max_rel_err",
    "inertia_frob_rel_err",
    "inertia_orientation_max_axis_angle_rad",
    "hull_mutual_penetration_proxy_m",
    "table_penetration_rec_m",
    "table_penetration_ref_m",
    "n_vertices_ref",
)


def _attach_cuda_chamfer(
    metrics: dict[str, Any],
    cen_table: list[float],
    hull_local: list[list[float]],
    v_ref: np.ndarray,
) -> dict[str, Any]:
    ch_cpu = metrics["hull_chamfer_m"]
    ch_cuda = None
    cuda_name = None
    try:
        v_rec = table_vertices_from_canonical_placement(cen_table, hull_local)
        cuda_out = chamfer_m_cuda(v_rec, v_ref)
        ch_cuda = cuda_out.distance_m
        cuda_name = cuda_out.cuda_device_name
    except RuntimeError:
        pass
    return {
        **metrics,
        "hull_chamfer_cuda_m": ch_cuda,
        "cuda_device_name": cuda_name,
        "chamfer_cpu_cuda_abs_diff_m": abs(ch_cpu - ch_cuda) if ch_cuda is not None else None,
    }


def evaluate_static_object_metrics(
    ctx: SceneContext,
    object_id: str,
    placement_object: dict[str, Any],
    plane: PlacementPlane,
    recon_poses: np.ndarray,
    *,
    anchor_frame_id: str | None = None,
    include_cuda: bool = False,
) -> dict[str, Any]:
    """
    Static metrics for one object and one reference anchor (spread frame).

    ``anchor_frame_id=None`` uses the protocol primary anchor (first spread frame).
    """
    cen = placement_object["cen_table"]
    hull = placement_object["hull_local"]
    frame_kw = {} if anchor_frame_id is None else {"frame_id": anchor_frame_id}
    v_ref = reference_vertices_table(ctx, object_id, plane, recon_poses=recon_poses, **frame_kw)
    metrics = compare_static_objects(cen, hull, v_ref)
    if include_cuda:
        return {"status": "ok", **_attach_cuda_chamfer(metrics, cen, hull, v_ref)}
    return {"status": "ok", **metrics}


def _variation_per_object(
    by_anchor: dict[str, dict[str, Any]],
    spread_frame_ids: list[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    object_ids = set()
    for fid in spread_frame_ids:
        object_ids.update(by_anchor[fid]["objects"].keys())

    for oid in sorted(object_ids, key=lambda x: int(x) if str(x).isdigit() else x):
        per_metric: dict[str, Any] = {}
        for key in ANCHOR_SENSITIVITY_METRIC_KEYS:
            values: dict[str, float | None] = {}
            numeric: list[float] = []
            for fid in spread_frame_ids:
                row = by_anchor[fid]["objects"].get(oid)
                if row is None or row.get("status") != "ok":
                    values[fid] = None
                    continue
                val = row.get(key)
                if val is None or not np.isfinite(val):
                    values[fid] = None
                    continue
                values[fid] = float(val)
                numeric.append(float(val))
            if not numeric:
                per_metric[key] = {
                    "by_anchor": values,
                    "min": None,
                    "max": None,
                    "range": None,
                    "mean": None,
                }
                continue
            arr = np.array(numeric, dtype=np.float64)
            per_metric[key] = {
                "by_anchor": values,
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "range": float(np.max(arr) - np.min(arr)),
                "mean": float(np.mean(arr)),
            }
        out[oid] = per_metric
    return out


def evaluate_static_anchor_sensitivity(
    placement_path: Path,
    method: str,
    scene_folder: str,
    *,
    include_cuda_on_primary: bool = True,
) -> dict[str, Any]:
    """
    Protocol primary-anchor static results plus per-spread-frame sensitivity diagnostics.

    Reconstruction geometry and placement plane are fixed; only the GT reference anchor
    (spread camera frame for ``reference_vertices_table``) varies across spread views.
    """
    placement = json.loads(placement_path.read_text(encoding="utf-8"))
    ctx = scene_context_for_placement(placement)
    plane = PlacementPlane.from_placement_dict(placement)
    recon = load_recon_bundle(recon_scene_dir(method, scene_folder))
    poses = recon.camera_poses
    spread_ids = spread_frame_ids(ctx.dataset, ctx.scene_id, ctx.gt_scene_dir)
    primary_fid = spread_ids[0]

    by_anchor: dict[str, dict[str, Any]] = {}
    for fid in spread_ids:
        per_object: dict[str, Any] = {}
        for oid, pl in placement["objects"].items():
            try:
                per_object[oid] = evaluate_static_object_metrics(
                    ctx,
                    oid,
                    pl,
                    plane,
                    poses,
                    anchor_frame_id=fid,
                    include_cuda=include_cuda_on_primary and fid == primary_fid,
                )
            except (ValueError, KeyError, FileNotFoundError) as exc:
                per_object[oid] = {"status": "reference_failed", "error": str(exc)}
        by_anchor[fid] = {
            "anchor_frame_id": fid,
            "is_primary_protocol_anchor": fid == primary_fid,
            "objects": per_object,
        }

    variation = _variation_per_object(by_anchor, spread_ids)
    primary_block = by_anchor[primary_fid]

    return {
        "method": method,
        "scene_folder": scene_folder,
        "dataset": placement["dataset"],
        "placement_path": str(placement_path.resolve()),
        "recon_root": str(recon.root.resolve()),
        "spread_frame_ids": spread_ids,
        "primary_anchor_frame_id": primary_fid,
        "n_spread_anchors": len(spread_ids),
        "n_objects": len(placement["objects"]),
        "primary": {
            "anchor_frame_id": primary_fid,
            "objects": primary_block["objects"],
        },
        "sensitivity_by_anchor": by_anchor,
        "variation_per_object": variation,
    }


def evaluate_canonical_static_scene(
    placement_path: Path,
    method: str,
    scene_folder: str,
) -> dict[str, Any]:
    """Primary protocol anchor only (first spread frame)."""
    placement = json.loads(placement_path.read_text(encoding="utf-8"))
    ctx = scene_context_for_placement(placement)
    plane = PlacementPlane.from_placement_dict(placement)
    recon = load_recon_bundle(recon_scene_dir(method, scene_folder))
    poses = recon.camera_poses

    per_object: dict[str, Any] = {}
    for oid, pl in placement["objects"].items():
        try:
            per_object[oid] = evaluate_static_object_metrics(
                ctx,
                oid,
                pl,
                plane,
                poses,
                anchor_frame_id=None,
                include_cuda=True,
            )
        except (ValueError, KeyError, FileNotFoundError) as exc:
            per_object[oid] = {"status": "reference_failed", "error": str(exc)}

    return {
        "method": method,
        "scene_folder": scene_folder,
        "dataset": placement["dataset"],
        "placement_path": str(placement_path.resolve()),
        "recon_root": str(recon.root.resolve()),
        "n_objects": len(per_object),
        "objects": per_object,
    }
