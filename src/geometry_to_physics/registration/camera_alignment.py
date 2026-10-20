"""DIAGNOSTIC ONLY — Umeyama Sim(3) on camera centers; not legacy frame_registration SE(3)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from geometry_to_physics.data import bop as bop_data
from geometry_to_physics.data.paths import gt_scene_dir
from geometry_to_physics.data.recon import ReconBundle, camera_centers
from geometry_to_physics.transforms.bop import bop_w2c_to_c2w
def umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate s, R, t with dst ≈ s * R @ src + t (row vectors: apply as (s * src @ R.T) + t).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    xs = src - mu_s
    xd = dst - mu_d
    cov = (xd.T @ xs) / n
    u, d, vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        S[2, 2] = -1.0
    R = u @ S @ vt
    var_s = float((xs**2).sum() / n)
    scale = float(np.trace(np.diag(d) @ S) / var_s) if var_s > 1e-12 else 1.0
    t = mu_d - scale * (mu_s @ R.T)
    return scale, R, t


def rotation_geodesic_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R = R_a.T @ R_b
    c = (np.trace(R) - 1.0) / 2.0
    c = float(np.clip(c, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


@dataclass(frozen=True)
class CameraAlignmentReport:
    scale: float | None
    rotation_align_deg_mean: float | None
    translation_residual_m_mean: float | None
    translation_residual_m_max: float | None
    rotation_residual_deg_mean: float | None
    rotation_residual_deg_max: float | None
    per_view: list[dict[str, Any]]
    status: str

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "estimation": "umeyama_sim3_on_camera_centers_only",
            "scale_recon_over_bop": self.scale,
            "rotation_align_deg_mean": self.rotation_align_deg_mean,
            "translation_residual_m_mean": self.translation_residual_m_mean,
            "translation_residual_m_max": self.translation_residual_m_max,
            "rotation_residual_deg_mean": self.rotation_residual_deg_mean,
            "rotation_residual_deg_max": self.rotation_residual_deg_max,
            "per_view": self.per_view,
            "thresholds": {
                "translation_residual_m_max": 0.05,
                "rotation_residual_deg_max": 5.0,
            },
        }


def bop_camera_poses_for_recon_views(dataset: str, scene_id: str, recon: ReconBundle) -> list[np.ndarray]:
    if dataset not in ("ycbv", "tless"):
        return []
    gdir = gt_scene_dir(dataset, scene_id)
    scene_gt, scene_cam = bop_data.load_scene_json(gdir)
    poses = []
    for p in recon.image_paths:
        fk = bop_data.frame_key(scene_gt, p.stem)
        cam = scene_cam[fk]
        poses.append(bop_w2c_to_c2w(cam["cam_R_w2c"], cam["cam_t_w2c"]))
    return poses


def align_bop_to_recon_cameras(
    dataset: str,
    scene_id: str,
    recon: ReconBundle,
) -> CameraAlignmentReport:
    if dataset not in ("ycbv", "tless"):
        return CameraAlignmentReport(None, None, None, None, None, None, [], "UNVERIFIED")
    bop_c2w = bop_camera_poses_for_recon_views(dataset, scene_id, recon)
    if len(bop_c2w) != recon.n_views:
        return CameraAlignmentReport(None, None, None, None, None, None, [], "UNVERIFIED")
    C_bop = np.asarray([T[:3, 3] for T in bop_c2w])
    C_rec = camera_centers(recon.camera_poses)
    scale, R, t = umeyama_sim3(C_bop, C_rec)
    per_view = []
    trans_res = []
    rot_res = []
    for i, (Tb, Tr) in enumerate(zip(bop_c2w, recon.camera_poses)):
        c_bop = Tb[:3, 3]
        c_pred = scale * (c_bop @ R.T) + t
        c_rec = Tr[:3, 3]
        t_err = float(np.linalg.norm(c_pred - c_rec))
        R_bop = Tb[:3, :3]
        R_rec = Tr[:3, :3]
        R_pred = R @ R_bop
        r_err = rotation_geodesic_deg(R_pred, R_rec)
        trans_res.append(t_err)
        rot_res.append(r_err)
        per_view.append(
            {
                "view_index": i,
                "frame_stem": recon.image_paths[i].stem,
                "translation_residual_m": t_err,
                "rotation_residual_deg": r_err,
            }
        )
    t_max = max(trans_res)
    r_max = max(rot_res)
    if t_max > 0.05:
        status = "FAILED"
    elif r_max > 5.0:
        status = "UNVERIFIED"
    else:
        status = "VERIFIED"
    return CameraAlignmentReport(
        scale=scale,
        rotation_align_deg_mean=float(np.mean(rot_res)),
        translation_residual_m_mean=float(np.mean(trans_res)),
        translation_residual_m_max=t_max,
        rotation_residual_deg_mean=float(np.mean(rot_res)),
        rotation_residual_deg_max=r_max,
        per_view=per_view,
        status=status,
    )


def apply_sim3_to_points(points: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    return scale * (pts @ R.T) + t


def sim3_from_alignment_report(
    dataset: str,
    scene_id: str,
    recon: ReconBundle,
) -> tuple[float, np.ndarray, np.ndarray] | None:
    if dataset not in ("ycbv", "tless"):
        return None
    bop_c2w = bop_camera_poses_for_recon_views(dataset, scene_id, recon)
    if len(bop_c2w) != recon.n_views:
        return None
    C_bop = np.asarray([T[:3, 3] for T in bop_c2w])
    C_rec = camera_centers(recon.camera_poses)
    return umeyama_sim3(C_bop, C_rec)
