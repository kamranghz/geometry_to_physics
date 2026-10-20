from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from geometry_to_physics.geometry.chamfer import (
    CPU_CUDA_CHAMFER_ABS_TOL_M,
    ChamferResult,
    chamfer_m,
    chamfer_m_cpu,
    chamfer_m_cuda,
    subsample_point_cloud,
)

_LEGACY_REPO = Path(r"C:\Users\kghol072\PhysFidelity3D")


def _grid_points(n_side: int, spacing: float = 0.01) -> np.ndarray:
    xs = np.linspace(0.0, spacing * (n_side - 1), n_side)
    pts = np.array([[x, y, z] for x in xs for y in xs for z in xs], dtype=np.float64)
    return pts


def test_identical_clouds_zero_distance():
    pts = _grid_points(4)
    assert chamfer_m_cpu(pts, pts) == 0.0


def test_known_translation():
    pts = np.array([[float(i * 10.0), 0.0, 0.0] for i in range(8)], dtype=np.float64)
    shift = np.array([0.15, 0.0, 0.0])
    d = chamfer_m_cpu(pts, pts + shift, n=100, seed=0)
    assert abs(d - 0.15) < 1e-12


def test_different_counts_subsampled_deterministic():
    rng = np.random.default_rng(99)
    a = rng.standard_normal((500, 3))
    b = rng.standard_normal((120, 3))
    d1 = chamfer_m_cpu(a, b, n=80, seed=7)
    d2 = chamfer_m_cpu(a, b, n=80, seed=7)
    d3 = chamfer_m_cpu(a, b, n=80, seed=8)
    assert d1 == d2
    assert d1 != d3


def test_subsample_matches_legacy_rng():
    rng = np.random.default_rng(3)
    pts = rng.standard_normal((100, 3))
    sub = subsample_point_cloud(pts, 25, seed=3)
    idx = np.random.default_rng(3).choice(100, 25, replace=False)
    assert np.allclose(sub, pts[idx])


def test_empty_cloud_raises():
    a = np.zeros((0, 3))
    b = np.ones((3, 3))
    with pytest.raises(ValueError, match="at least one point"):
        chamfer_m_cpu(a, b)
    with pytest.raises(ValueError, match="at least one point"):
        chamfer_m_cpu(b, a)


def test_invalid_shape_raises():
    with pytest.raises(ValueError, match="shape"):
        chamfer_m_cpu(np.zeros((4, 2)), np.zeros((4, 3)))


def test_chamfer_m_explicit_cpu_backend():
    pts = _grid_points(3)
    out = chamfer_m(pts, pts, device="cpu")
    assert isinstance(out, ChamferResult)
    assert out.backend == "cpu"
    assert out.cuda_device_name is None
    assert out.distance_m == 0.0


def test_cuda_requested_without_device_raises():
    import torch

    if torch.cuda.is_available():
        pytest.skip("CUDA available; use cuda agreement test instead")
    with pytest.raises(RuntimeError, match="CUDA"):
        chamfer_m_cuda(np.ones((4, 3)), np.ones((4, 3)) + 0.1)


def test_cpu_cuda_agreement():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    rng = np.random.default_rng(202)
    a = rng.standard_normal((3000, 3))
    b = rng.standard_normal((2500, 3))
    cpu = chamfer_m_cpu(a, b, n=2000, seed=0)
    gpu = chamfer_m_cuda(a, b, n=2000, seed=0, chunk_size=512)
    assert gpu.backend == "cuda"
    assert gpu.cuda_device_name
    assert abs(cpu - gpu.distance_m) <= CPU_CUDA_CHAMFER_ABS_TOL_M

    routed = chamfer_m(a, b, n=2000, seed=0, device="cuda")
    assert routed.backend == "cuda"
    assert abs(routed.distance_m - cpu) <= CPU_CUDA_CHAMFER_ABS_TOL_M


@pytest.mark.skipif(not _LEGACY_REPO.is_dir(), reason="PhysFidelity3D not available")
def test_legacy_cpu_parity():
    sys.path.insert(0, str(_LEGACY_REPO))
    from cross_engine.reference_anchor.metrics import chamfer_m as legacy_chamfer_m

    rng = np.random.default_rng(42)
    cases = [
        (rng.standard_normal((50, 3)), rng.standard_normal((60, 3)), 2000, 0),
        (_grid_points(6), _grid_points(6) + np.array([0.02, 0.0, 0.0]), 500, 1),
        (rng.standard_normal((5000, 3)), rng.standard_normal((3000, 3)), 2000, 0),
    ]
    for a, b, n, seed in cases:
        legacy = legacy_chamfer_m(a, b, n=n, seed=seed)
        ours = chamfer_m_cpu(a, b, n=n, seed=seed)
        assert ours == legacy
