from __future__ import annotations

import os
from pathlib import Path

from geometry_to_physics.config import load_paths

# Historical / invalid scientific outputs — never use as new pipeline inputs.
BLOCKED_BASENAMES = frozenset(
    {
        "object_property_pairs.csv",
        "object_property_rows.jsonl",
        "trajectory.npz",
        "trajectory.json",
        "INVALID_REFERENCE_FRAME_UNTIL_RERUN.json",
    }
)

BLOCKED_PATH_SUBSTRINGS = (
    "tcsvt_reference_anchor/properties",
    "tcsvt_reference_anchor/dynamics",
    "tcsvt_reference_anchor/archive",
    "physics_eval/",
    "physics_eval_legacy",
)

PLACEMENT_BASENAME = "placement_data.json"


def _canonical_root() -> Path | None:
    return load_paths().canonical_root


def is_blocked_historical_output(path: Path | str) -> bool:
    p = Path(path)
    if p.name in BLOCKED_BASENAMES:
        return True
    s = str(p).replace("\\", "/").lower()
    return any(sub.lower() in s for sub in BLOCKED_PATH_SUBSTRINGS)


def placement_data_path(method: str, scene_folder: str) -> Path | None:
    root = _canonical_root()
    if root is None:
        return None
    return root / method / scene_folder / PLACEMENT_BASENAME


def classify_placement_file(path: Path | str) -> str:
    """
    ALLOWED_REQUIRED — under GTP_CANONICAL_ROOT, correct layout, file exists.
    UNVERIFIED — canonical root unset, file missing, or path outside canonical root.
    BLOCKED — historical invalid artifact patterns.
    """
    p = Path(path).resolve()
    if is_blocked_historical_output(p):
        return "BLOCKED"
    if p.name != PLACEMENT_BASENAME:
        return "UNVERIFIED"
    root = _canonical_root()
    if root is None:
        return "UNVERIFIED"
    try:
        p.relative_to(root.resolve())
    except ValueError:
        return "UNVERIFIED"
    if not p.is_file():
        return "UNVERIFIED"
    return "ALLOWED_REQUIRED"


def assert_allowed_source(path: Path | str) -> None:
    p = Path(path)
    if is_blocked_historical_output(p):
        raise ValueError(f"Blocked historical/invalid artifact: {path}")
    if p.name == PLACEMENT_BASENAME and classify_placement_file(p) != "ALLOWED_REQUIRED":
        raise ValueError(f"placement_data.json not allowed (status UNVERIFIED or outside canonical root): {path}")
