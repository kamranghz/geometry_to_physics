from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PlacementPlane:
    """Support plane ax + by + cz + d = 0 in reconstruction table/world coordinates (metres)."""

    a: float
    b: float
    c: float
    d: float

    @classmethod
    def from_placement_dict(cls, placement: dict[str, Any]) -> PlacementPlane:
        if placement.get("plane_coefficients_abcd"):
            abcd = placement["plane_coefficients_abcd"]
            return cls(float(abcd[0]), float(abcd[1]), float(abcd[2]), float(abcd[3]))
        plane = placement.get("plane") or {}
        return cls(
            float(plane["a"]),
            float(plane["b"]),
            float(plane["c"]),
            float(plane["d"]),
        )
