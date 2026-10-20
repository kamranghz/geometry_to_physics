"""7-method × 42-scene catalog (recovered from PhysFidelity3D cross_engine/scene_catalog.py)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import yaml

from geometry_to_physics.constants import METHODS
from geometry_to_physics.tcsvt.protocol import ENGINES

DEFAULT_MANIFEST = Path(__file__).resolve().parents[3] / "configs" / "tcsvt_batch_manifest.yaml"

MODEL_SCENE_SUFFIX = {
    "DUSt3R": "_spread_scaled",
    "MASt3R": "_spread_scaled",
    "MonST3R": "_spread_scaled",
    "MoVieS": "_spread_scaled",
    "VGGT": "_spread",
    "VGGT-Omega": "_spread",
    "VGG-TTT": "_spread",
}


@dataclass(frozen=True)
class SceneJob:
    method: str
    dataset: str
    scene_id: str
    scene_folder: str
    engines: tuple[str, ...] = ENGINES

    @property
    def key(self) -> str:
        return f"{self.method}/{self.dataset}/{self.scene_id}"


def load_manifest(path: Path | None = None) -> dict:
    p = path or DEFAULT_MANIFEST
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def iter_scene_jobs(
    manifest: dict | None = None,
    methods: list[str] | None = None,
) -> Iterator[SceneJob]:
    m = manifest or load_manifest()
    models_cfg = m.get("models", {})
    want = methods or [name for name, cfg in models_cfg.items() if cfg.get("enabled", True)]
    if not want:
        want = list(METHODS)
    for ds_name, ds_block in m.get("datasets", {}).items():
        prefix = ds_block.get("prefix", f"{ds_name}_")
        for scene_id in ds_block.get("scene_ids", []):
            for method_name in want:
                mcfg = models_cfg.get(method_name, {})
                suffix = mcfg.get("scene_suffix", MODEL_SCENE_SUFFIX.get(method_name, "_spread_scaled"))
                scene_folder = f"{prefix}{scene_id}{suffix}"
                yield SceneJob(
                    method=method_name,
                    dataset=ds_name,
                    scene_id=str(scene_id),
                    scene_folder=scene_folder,
                )


def count_scene_jobs(methods: list[str] | None = None) -> int:
    return sum(1 for _ in iter_scene_jobs(methods=methods))
