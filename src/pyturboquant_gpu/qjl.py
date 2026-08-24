"""
Quantized Johnson–Lindenstrauss (QJL) 1-bit inner-product quantiser.

PyTorch implementation for GPU acceleration.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


def generate_qjl_matrix(
    dim: int,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Generate a d×d random Gaussian matrix for QJL.

    Parameters
    ----------
    dim : int
    seed : int or None
    device : torch.device or None
    dtype : torch.dtype or None

    Returns
    -------
    S : Tensor, shape (d, d)
    """
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32

    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)

    S = torch.randn(dim, dim, generator=gen, dtype=dtype, device="cpu")
    return S.to(device=device)


def qjl_quantize(
    vectors: torch.Tensor, S: torch.Tensor
) -> torch.Tensor:
    """Apply QJL quantisation: sign(S · x).

    Parameters
    ----------
    vectors : Tensor, shape (..., d)
    S : Tensor, shape (d, d)

    Returns
    -------
    signs : Tensor, shape (..., d)  of -1.0 / +1.0
    """
    projected = vectors @ S.T
    signs = torch.sign(projected)
    signs[signs == 0] = 1.0
    return signs


def qjl_dequantize(
    signs: torch.Tensor, S: torch.Tensor
) -> torch.Tensor:
    """QJL inverse map: Q_qjl^{-1}(z) = √(π/2) · S^T · z / d.

    Parameters
    ----------
    signs : Tensor, shape (..., d)
    S : Tensor, shape (d, d)

    Returns
    -------
    reconstructed : Tensor, shape (..., d)
    """
    d = S.shape[0]
    scale = math.sqrt(math.pi / 2.0) / d
    return scale * (signs.to(dtype=S.dtype) @ S)
