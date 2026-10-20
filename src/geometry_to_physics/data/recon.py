from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ReconBundle:
    root: Path
    metadata: dict
    camera_poses: np.ndarray  # (V, 4, 4) cam-to-world in reconstruction frame
    image_paths: list[Path]

    @property
    def n_views(self) -> int:
        return int(self.camera_poses.shape[0])

    @property
    def scale_normalization(self) -> dict:
        return self.metadata.get("scale_normalization") or {}

    @property
    def coordinate_frame(self) -> str | None:
        return self.metadata.get("coordinate_frame")


def load_recon_bundle(recon_dir: Path) -> ReconBundle:
    meta_path = recon_dir / "metadata.json"
    poses_path = recon_dir / "camera_poses.npy"
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    if not poses_path.is_file():
        raise FileNotFoundError(poses_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    poses = np.load(poses_path)
    rgb = [Path(p) for p in meta.get("image_paths", [])]
    if len(rgb) != poses.shape[0]:
        raise ValueError(f"image_paths ({len(rgb)}) != camera_poses views ({poses.shape[0]})")
    return ReconBundle(root=recon_dir, metadata=meta, camera_poses=poses, image_paths=rgb)


def camera_centers(poses: np.ndarray) -> np.ndarray:
    return poses[:, :3, 3].copy()
