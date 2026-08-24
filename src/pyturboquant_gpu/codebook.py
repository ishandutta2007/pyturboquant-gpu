"""
Lloyd-Max codebook computation for the Beta coordinate distribution.

After a random orthogonal rotation, each coordinate of a unit-norm vector
in R^d follows a Beta((d-1)/2, (d-1)/2) distribution on [-1, 1].

This module solves the 1-D Lloyd-Max quantisation problem for that
distribution using SciPy (one-time cost) and caches the results as
PyTorch tensors on the requested device.
"""

from __future__ import annotations

import functools
import math
from typing import Optional, Tuple

import numpy as np
import torch
from scipy import integrate, special


def _beta_pdf(x: float, dim: int) -> float:
    """Scalar PDF evaluation for the Beta coordinate distribution."""
    if abs(x) >= 1.0:
        return 0.0
    d = dim
    log_norm = (
        special.gammaln(d / 2.0)
        - 0.5 * math.log(math.pi)
        - special.gammaln((d - 1) / 2.0)
    )
    exponent = (d - 3) / 2.0
    if exponent == 0:
        return math.exp(log_norm)
    return math.exp(log_norm + exponent * math.log(1.0 - x * x))


def _conditional_expectation(a: float, b: float, dim: int) -> float:
    """Compute E[X | a ≤ X ≤ b] for the Beta coordinate distribution."""
    if b - a < 1e-15:
        return (a + b) / 2.0
    numerator, _ = integrate.quad(
        lambda x: x * _beta_pdf(x, dim), a, b
    )
    denominator, _ = integrate.quad(
        lambda x: _beta_pdf(x, dim), a, b
    )
    if abs(denominator) < 1e-30:
        return (a + b) / 2.0
    return numerator / denominator


@functools.lru_cache(maxsize=256)
def lloyd_max_codebook_np(
    dim: int,
    bits: int,
    max_iter: int = 300,
    tol: float = 1e-14,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Lloyd-Max codebook (returned as NumPy arrays).

    Parameters
    ----------
    dim : int
        Vector dimension *d*.
    bits : int
        Bit-width per coordinate.
    max_iter : int
        Maximum Lloyd iterations.
    tol : float
        Convergence tolerance.

    Returns
    -------
    centroids : ndarray, shape (2**bits,)
    boundaries : ndarray, shape (2**bits - 1,)
    """
    n_levels = 1 << bits

    centroids = np.linspace(-1.0, 1.0, n_levels + 2)[1:-1].copy()

    for _ in range(max_iter):
        boundaries = 0.5 * (centroids[:-1] + centroids[1:])
        edges = np.empty(n_levels + 1, dtype=np.float64)
        edges[0] = -1.0
        edges[-1] = 1.0
        edges[1:-1] = boundaries

        new_centroids = np.empty_like(centroids)
        for i in range(n_levels):
            new_centroids[i] = _conditional_expectation(
                edges[i], edges[i + 1], dim
            )

        delta = np.max(np.abs(new_centroids - centroids))
        centroids = new_centroids
        if delta < tol:
            break

    boundaries = 0.5 * (centroids[:-1] + centroids[1:])
    return centroids.astype(np.float64), boundaries.astype(np.float64)


# Cache for torch tensor codebooks keyed by (dim, bits, device, dtype)
_codebook_cache: dict = {}


def get_codebook(
    dim: int,
    bits: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a cached Lloyd-Max codebook as PyTorch tensors.

    Parameters
    ----------
    dim : int
        Vector dimension.
    bits : int
        Bit-width per coordinate.
    device : torch.device or None
        Target device.
    dtype : torch.dtype or None
        Target dtype (default: float32).

    Returns
    -------
    centroids : Tensor, shape (2**bits,)
    boundaries : Tensor, shape (2**bits - 1,)
    """
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32

    key = (dim, bits, str(device), dtype)
    if key not in _codebook_cache:
        c_np, b_np = lloyd_max_codebook_np(dim, bits)
        c_t = torch.from_numpy(c_np).to(device=device, dtype=dtype)
        b_t = torch.from_numpy(b_np).to(device=device, dtype=dtype)
        _codebook_cache[key] = (c_t, b_t)
    return _codebook_cache[key]
