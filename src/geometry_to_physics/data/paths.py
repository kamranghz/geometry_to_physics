from __future__ import annotations

from pathlib import Path

from geometry_to_physics.config import load_paths


def gt_scene_dir(dataset: str, scene_id: str) -> Path:
    root = load_paths().gt_root
    if dataset == "ycbv":
        return root / "YCBV" / "ycbv_test_bop19" / "test" / scene_id
    if dataset == "tless":
        return root / "TLESS" / "test_primesense" / scene_id
    if dataset == "picasso":
        return root / "Picasso" / "test" / scene_id
    raise ValueError(dataset)


def recon_scene_dir(method: str, scene_folder: str) -> Path:
    return load_paths().recon_root / method / scene_folder
