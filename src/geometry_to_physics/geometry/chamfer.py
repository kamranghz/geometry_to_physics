"""
Symmetric mean Chamfer distance in metres (TCSVT reference_anchor/metrics.chamfer_m).

Definition: subsample to at most ``n`` points per cloud (seeded), then
0.5 * (mean_{p in A} min_{q in B} ||p-q|| + mean_{q in B} min_{p in A} ||q-p||).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.spatial import cKDTree

DEFAULT_CHAMFER_SAMPLE_COUNT = 2000
DEFAULT_CHAMFER_SEED = 0

# Documented tolerance for CPU (cKDTree) vs CUDA (chunked torch.cdist) agreement.
CPU_CUDA_CHAMFER_ABS_TOL_M = 1e-5

ChamferBackend = Literal["cpu", "cuda"]


@dataclass(frozen=True)
class ChamferResult:
    distance_m: float
    backend: ChamferBackend
    cuda_device_name: str | None = None


def subsample_point_cloud(points: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    """Deterministic subsample for a single cloud (``default_rng(seed).choice``)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"expected point cloud shape (N, 3), got {pts.shape}")
    if pts.shape[0] == 0:
        raise ValueError("point cloud must contain at least one point")
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    if pts.shape[0] <= max_points:
        return pts
    rng = np.random.default_rng(seed)
    idx = rng.choice(pts.shape[0], max_points, replace=False)
    return pts[idx]


def _subsample_chamfer_pair(
    a: np.ndarray,
    b: np.ndarray,
    n: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy pairing: one RNG stream; A subsample then B subsample when both exceed ``n``."""
    rng = np.random.default_rng(seed)
    if a.shape[0] > n:
        a = a[rng.choice(a.shape[0], n, replace=False)]
    if b.shape[0] > n:
        b = b[rng.choice(b.shape[0], n, replace=False)]
    return a, b


def _validate_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"cloud A expected shape (N, 3), got {a.shape}")
    if b.ndim != 2 or b.shape[1] != 3:
        raise ValueError(f"cloud B expected shape (M, 3), got {b.shape}")
    if a.shape[0] == 0 or b.shape[0] == 0:
        raise ValueError("both point clouds must contain at least one point")
    return a, b


def chamfer_m_cpu(
    a: np.ndarray,
    b: np.ndarray,
    n: int = DEFAULT_CHAMFER_SAMPLE_COUNT,
    seed: int = DEFAULT_CHAMFER_SEED,
) -> float:
    """CPU reference: scipy cKDTree nearest-neighbour means (legacy parity)."""
    a, b = _validate_pair(a, b)
    a, b = _subsample_chamfer_pair(a, b, n, seed)
    tree_b, tree_a = cKDTree(b), cKDTree(a)
    return float(0.5 * (tree_b.query(a, k=1)[0].mean() + tree_a.query(b, k=1)[0].mean()))


def _mean_min_distances_chunked(
    query: "torch.Tensor",
    reference: "torch.Tensor",
    chunk_size: int,
) -> "torch.Tensor":
    import torch

    if query.shape[0] == 0:
        raise ValueError("query point cloud is empty")
    mins: list[torch.Tensor] = []
    for start in range(0, query.shape[0], chunk_size):
        q_chunk = query[start : start + chunk_size]
        dist = torch.cdist(q_chunk, reference, p=2)
        mins.append(dist.min(dim=1).values)
    return torch.cat(mins).mean()


def chamfer_m_cuda(
    a: np.ndarray,
    b: np.ndarray,
    n: int = DEFAULT_CHAMFER_SAMPLE_COUNT,
    seed: int = DEFAULT_CHAMFER_SEED,
    *,
    device: str = "cuda",
    chunk_size: int = 4096,
) -> ChamferResult:
    """
    CUDA execution: subsample on CPU (deterministic), transfer once, chunked cdist on GPU.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA chamfer requested but torch.cuda.is_available() is False")

    a, b = _validate_pair(a, b)
    a, b = _subsample_chamfer_pair(a, b, n, seed)

    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise ValueError(f"chamfer_m_cuda requires a CUDA device, got {torch_device}")

    a_gpu = torch.from_numpy(a).to(device=torch_device, dtype=torch.float64)
    b_gpu = torch.from_numpy(b).to(device=torch_device, dtype=torch.float64)

    mean_a_to_b = _mean_min_distances_chunked(a_gpu, b_gpu, chunk_size)
    mean_b_to_a = _mean_min_distances_chunked(b_gpu, a_gpu, chunk_size)
    distance = float(0.5 * (mean_a_to_b + mean_b_to_a).item())
    name = torch.cuda.get_device_name(torch_device)
    return ChamferResult(distance_m=distance, backend="cuda", cuda_device_name=name)


def chamfer_m(
    a: np.ndarray,
    b: np.ndarray,
    n: int = DEFAULT_CHAMFER_SAMPLE_COUNT,
    seed: int = DEFAULT_CHAMFER_SEED,
    *,
    device: ChamferBackend = "cpu",
    cuda_device: str = "cuda",
    chunk_size: int = 4096,
) -> ChamferResult:
    """
    Symmetric mean Chamfer distance with explicit backend selection.

    ``device="cpu"`` always uses cKDTree. ``device="cuda"`` requires CUDA; no silent fallback.
    """
    if device == "cpu":
        return ChamferResult(distance_m=chamfer_m_cpu(a, b, n=n, seed=seed), backend="cpu", cuda_device_name=None)
    if device == "cuda":
        return chamfer_m_cuda(
            a,
            b,
            n=n,
            seed=seed,
            device=cuda_device,
            chunk_size=chunk_size,
        )
    raise ValueError(f"device must be 'cpu' or 'cuda', got {device!r}")
