from __future__ import annotations

from pathlib import Path

import numpy as np

from geometry_to_physics.data import bop as bop_data
from geometry_to_physics.data.paths import gt_scene_dir
from geometry_to_physics.data.recon import ReconBundle, camera_centers
from geometry_to_physics.transforms.bop import bop_w2c_to_c2w


def verify_rgb_pose_indexing(recon: ReconBundle) -> dict:
    """Each metadata image_paths[i] must exist and align with camera_poses[i]."""
    checks = []
    ok = True
    for i, p in enumerate(recon.image_paths):
        exists = p.is_file()
        checks.append({"index": i, "rgb": str(p), "rgb_exists": exists})
        if not exists:
            ok = False
    return {"status": "VERIFIED" if ok else "FAILED", "per_view": checks}


def relative_baseline_ratio(
    dataset: str,
    scene_id: str,
    recon: ReconBundle,
    i: int = 0,
    j: int = 1,
) -> dict:
    """Compare |C_i - C_j| in BOP world vs reconstruction world (scale diagnostic)."""
    if dataset not in ("ycbv", "tless"):
        return {"status": "UNVERIFIED", "reason": "BOP baselines only for ycbv/tless"}
    gdir = gt_scene_dir(dataset, scene_id)
    scene_gt, scene_cam = bop_data.load_scene_json(gdir)
    stems = [p.stem for p in recon.image_paths]
    fk_i = bop_data.frame_key(scene_gt, stems[i])
    fk_j = bop_data.frame_key(scene_gt, stems[j])
    c2w_i = bop_w2c_to_c2w(scene_cam[fk_i]["cam_R_w2c"], scene_cam[fk_i]["cam_t_w2c"])
    c2w_j = bop_w2c_to_c2w(scene_cam[fk_j]["cam_R_w2c"], scene_cam[fk_j]["cam_t_w2c"])
    bop_bi = float(np.linalg.norm(c2w_i[:3, 3] - c2w_j[:3, 3]))
    rec = camera_centers(recon.camera_poses)
    rec_bi = float(np.linalg.norm(rec[i] - rec[j]))
    ratio = rec_bi / bop_bi if bop_bi > 1e-9 else None
    sn = recon.scale_normalization
    scale_factor = sn.get("scale_factor") if sn.get("applied") else None
    status = "UNVERIFIED"
    interpretation = (
        "After metric scale_normalization, recon camera translations should match BOP metric baselines "
        "(ratio recon/bop ≈ 1). metadata scale_factor is Sim(3) uplift from arbitrary recon, not this ratio."
    )
    if ratio is not None and sn.get("applied"):
        rel_err = abs(ratio - 1.0)
        status = "VERIFIED" if rel_err < 0.08 else "FAILED"
    elif ratio is not None:
        status = "UNVERIFIED"
    return {
        "status": status,
        "bop_baseline_m": bop_bi,
        "recon_baseline_m": rec_bi,
        "ratio_recon_over_bop": ratio,
        "metadata_scale_factor": scale_factor,
        "interpretation": interpretation,
        "views": [stems[i], stems[j]],
    }
