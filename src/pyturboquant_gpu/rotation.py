"""
Random orthogonal rotation matrices (Haar-distributed) using PyTorch.

TurboQuant rotates input vectors with a uniformly random orthogonal matrix
so that each coordinate follows a known Beta distribution.
"""

from __future__ import annotations

from typing import Optional

import torch


def generate_rotation_matrix(
    dim: int,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Generate a uniformly random orthogonal matrix via QR decomposition.

    Parameters
    ----------
    dim : int
        Matrix dimension *d*.
    seed : int or None
        Random seed for reproducibility.
    device : torch.device or None
        Target device.
    dtype : torch.dtype or None
        Target dtype (default: float32).

    Returns
    -------
    Q : Tensor, shape (d, d)
        Orthogonal matrix.
    """
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32

    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)

    # Generate random Gaussian matrix on CPU, then move
    Z = torch.randn(dim, dim, generator=gen, dtype=dtype, device="cpu")
    Q, R = torch.linalg.qr(Z)

    # Correct signs for Haar measure
    d = torch.sign(torch.diag(R))
    d[d == 0] = 1.0
    Q = Q * d.unsqueeze(0)

    return Q.to(device=device)


def rotate(
    vectors: torch.Tensor, rotation_matrix: torch.Tensor
) -> torch.Tensor:
    """Apply rotation: y = vectors @ Π^T.

    Parameters
    ----------
    vectors : Tensor, shape (..., d)
    rotation_matrix : Tensor, shape (d, d)

    Returns
    -------
    Tensor, shape (..., d)
    """
    return vectors @ rotation_matrix.T


def inverse_rotate(
    vectors: torch.Tensor, rotation_matrix: torch.Tensor
) -> torch.Tensor:
    """Apply inverse rotation: x = vectors @ Π.

    Parameters
    ----------
    vectors : Tensor, shape (..., d)
    rotation_matrix : Tensor, shape (d, d)

    Returns
    -------
    Tensor, shape (..., d)
    """
    return vectors @ rotation_matrix
