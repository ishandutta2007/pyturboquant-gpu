"""
Core TurboQuant quantisation and dequantisation routines (PyTorch backend).

Provides two modes:

* **TurboQuant_MSE** — minimises mean-squared reconstruction error.
* **TurboQuant_Prod** — unbiased inner-product estimator (MSE stage +
  1-bit QJL residual correction).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from pyturboquant_gpu.codebook import get_codebook
from pyturboquant_gpu.packing import (
    pack_indices,
    pack_signs,
    unpack_indices,
    unpack_signs,
)
from pyturboquant_gpu.qjl import (
    generate_qjl_matrix,
    qjl_dequantize,
    qjl_quantize,
)
from pyturboquant_gpu.rotation import (
    generate_rotation_matrix,
    inverse_rotate,
    rotate,
)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class QuantizedMSE:
    """Container for MSE-quantised vectors.

    Attributes
    ----------
    packed_indices : Tensor of uint8
        Bit-packed codebook indices for all vectors.
    norms : Tensor, shape (n,)
        Original L2 norms of each input vector.
    rotation_matrix : Tensor, shape (d, d)
        The random orthogonal matrix Π used for rotation.
    bits : int
        Bit-width per coordinate.
    dim : int
        Original vector dimension *d*.
    original_shape : tuple
        Shape of the input tensor (before flattening into 2-D).
    num_vectors : int
        Number of vectors that were quantised.
    centroids : Tensor, shape (2**bits,)
        Lloyd-Max codebook centroids.
    boundaries : Tensor, shape (2**bits - 1,)
        Decision boundaries for scalar quantisation.
    """

    packed_indices: torch.Tensor
    norms: torch.Tensor
    rotation_matrix: torch.Tensor
    bits: int
    dim: int
    original_shape: tuple
    num_vectors: int
    centroids: torch.Tensor
    boundaries: torch.Tensor


@dataclass
class QuantizedProd:
    """Container for inner-product-optimal quantised vectors.

    Combines an MSE quantisation at ``(bits - 1)`` bits with a 1-bit QJL
    residual correction, giving an unbiased inner-product estimator at
    ``bits`` total bits per coordinate.

    Attributes
    ----------
    mse_component : QuantizedMSE
        MSE quantisation result using ``bits - 1`` bits.
    packed_signs : Tensor of uint8
        Packed QJL sign bits for each vector.
    residual_norms : Tensor, shape (n,)
        L2 norms of the residual vectors (γ).
    qjl_matrix : Tensor, shape (d, d)
        Random Gaussian matrix used for QJL.
    bits : int
        Total bit budget (the MSE component uses ``bits - 1``).
    """

    mse_component: QuantizedMSE
    packed_signs: torch.Tensor
    residual_norms: torch.Tensor
    qjl_matrix: torch.Tensor
    bits: int


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prepare_vectors(
    vectors: torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, tuple, int, int, torch.device, torch.dtype]:
    """Reshape arbitrary-shaped input into (n, d) and return metadata."""
    if not isinstance(vectors, torch.Tensor):
        vectors = torch.as_tensor(vectors)

    original_shape = vectors.shape
    dev = device if device is not None else vectors.device
    dt = dtype if dtype is not None else vectors.dtype
    if dt not in (torch.float32, torch.float64):
        dt = torch.float32

    vectors = vectors.to(device=dev, dtype=dt)

    if vectors.ndim == 1:
        vectors_2d = vectors.unsqueeze(0)
    else:
        vectors_2d = vectors.reshape(-1, vectors.shape[-1])
    n, d = vectors_2d.shape
    return vectors_2d, original_shape, n, d, dev, dt


@torch.no_grad()
def _scalar_quantize(
    values: torch.Tensor,
    centroids: torch.Tensor,
    boundaries: torch.Tensor,
) -> torch.Tensor:
    """Map each scalar value to the index of its nearest centroid.

    Uses ``torch.searchsorted`` for O(log k) lookup.
    """
    flat = values.reshape(-1)
    indices = torch.searchsorted(boundaries, flat)
    return indices.reshape(values.shape)


def _scalar_dequantize(
    indices: torch.Tensor, centroids: torch.Tensor
) -> torch.Tensor:
    """Map codebook indices back to centroid values."""
    return centroids[indices.long()]


# ---------------------------------------------------------------------------
# MSE quantisation
# ---------------------------------------------------------------------------

@torch.no_grad()
def quantize_mse(
    vectors: torch.Tensor,
    bits: int,
    dim: Optional[int] = None,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> QuantizedMSE:
    """Quantise vectors using TurboQuant_MSE.

    Parameters
    ----------
    vectors : Tensor, shape (..., d)
        Input vectors.
    bits : int
        Bit-width per coordinate (1–8).
    dim : int or None
        Override the detected vector dimension.
    seed : int or None
        Random seed for the rotation matrix.
    device : torch.device or None
        Target device for computation.

    Returns
    -------
    QuantizedMSE
    """
    if not 1 <= bits <= 8:
        raise ValueError(f"bits must be in [1, 8], got {bits}")

    vectors_2d, original_shape, n, d, dev, dt = _prepare_vectors(
        vectors, device=device
    )
    if dim is not None:
        d = dim

    # 1. Store norms and normalise
    norms = torch.linalg.norm(vectors_2d, dim=1)
    safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
    unit_vectors = vectors_2d / safe_norms.unsqueeze(1)

    # 2. Random rotation
    Pi = generate_rotation_matrix(d, seed=seed, device=dev, dtype=dt)
    rotated = rotate(unit_vectors, Pi)

    # 3. Scalar quantisation
    centroids, boundaries = get_codebook(d, bits, device=dev, dtype=dt)
    indices = _scalar_quantize(rotated, centroids, boundaries)

    # 4. Pack indices
    packed = pack_indices(indices.reshape(-1), bits)

    return QuantizedMSE(
        packed_indices=packed,
        norms=norms,
        rotation_matrix=Pi,
        bits=bits,
        dim=d,
        original_shape=original_shape,
        num_vectors=n,
        centroids=centroids,
        boundaries=boundaries,
    )


@torch.no_grad()
def dequantize_mse(quantized: QuantizedMSE) -> torch.Tensor:
    """Reconstruct vectors from a TurboQuant_MSE quantisation result.

    Parameters
    ----------
    quantized : QuantizedMSE

    Returns
    -------
    Tensor
        Reconstructed vectors with the same shape as the original input.
    """
    n = quantized.num_vectors
    d = quantized.dim
    bits = quantized.bits

    # 1. Unpack indices
    total = n * d
    indices = unpack_indices(quantized.packed_indices, bits, total)
    indices = indices.reshape(n, d)

    # 2. Map to centroids
    reconstructed_rotated = _scalar_dequantize(indices, quantized.centroids)

    # 3. Inverse rotation
    reconstructed_unit = inverse_rotate(
        reconstructed_rotated, quantized.rotation_matrix
    )

    # 4. Rescale by original norms
    reconstructed = reconstructed_unit * quantized.norms.unsqueeze(1)

    return reconstructed.reshape(quantized.original_shape)


# ---------------------------------------------------------------------------
# Inner-product-optimal quantisation
# ---------------------------------------------------------------------------

@torch.no_grad()
def quantize_prod(
    vectors: torch.Tensor,
    bits: int,
    dim: Optional[int] = None,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> QuantizedProd:
    """Quantise vectors using TurboQuant_Prod (unbiased inner products).

    Parameters
    ----------
    vectors : Tensor, shape (..., d)
    bits : int
        Total bit-width per coordinate (2–8).
    dim : int or None
    seed : int or None
    device : torch.device or None

    Returns
    -------
    QuantizedProd
    """
    if not 2 <= bits <= 8:
        raise ValueError(
            f"bits must be in [2, 8] for Prod mode, got {bits}"
        )

    vectors_2d, original_shape, n, d, dev, dt = _prepare_vectors(
        vectors, device=device
    )
    if dim is not None:
        d = dim

    # 1. Store norms and normalise
    norms = torch.linalg.norm(vectors_2d, dim=1)
    safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
    unit_vectors = vectors_2d / safe_norms.unsqueeze(1)

    # 2. MSE quantisation at (bits - 1)
    mse_bits = bits - 1
    Pi = generate_rotation_matrix(d, seed=seed, device=dev, dtype=dt)
    centroids, boundaries = get_codebook(d, mse_bits, device=dev, dtype=dt)

    rotated = rotate(unit_vectors, Pi)
    indices = _scalar_quantize(rotated, centroids, boundaries)
    packed_mse = pack_indices(indices.reshape(-1), mse_bits)

    mse_result = QuantizedMSE(
        packed_indices=packed_mse,
        norms=norms,
        rotation_matrix=Pi,
        bits=mse_bits,
        dim=d,
        original_shape=original_shape,
        num_vectors=n,
        centroids=centroids,
        boundaries=boundaries,
    )

    # 3. Compute residual on unit sphere
    reconstructed_rotated = _scalar_dequantize(indices, centroids)
    reconstructed_unit = inverse_rotate(reconstructed_rotated, Pi)
    residuals = unit_vectors - reconstructed_unit

    # 4. Residual norms (γ)
    residual_norms = torch.linalg.norm(residuals, dim=1)

    # 5. Normalise residuals for QJL
    safe_res = torch.where(
        residual_norms > 0, residual_norms, torch.ones_like(residual_norms)
    )
    normalised_residuals = residuals / safe_res.unsqueeze(1)

    # 6. QJL quantisation
    qjl_seed = (seed + 1_000_003) if seed is not None else None
    S = generate_qjl_matrix(d, seed=qjl_seed, device=dev, dtype=dt)
    signs = qjl_quantize(normalised_residuals, S)

    # 7. Pack sign bits
    packed_parts = []
    for i in range(n):
        packed_parts.append(pack_signs(signs[i]))
    packed_signs = torch.cat(packed_parts)

    return QuantizedProd(
        mse_component=mse_result,
        packed_signs=packed_signs,
        residual_norms=residual_norms,
        qjl_matrix=S,
        bits=bits,
    )


@torch.no_grad()
def dequantize_prod(quantized: QuantizedProd) -> torch.Tensor:
    """Reconstruct vectors from a TurboQuant_Prod quantisation result.

    Parameters
    ----------
    quantized : QuantizedProd

    Returns
    -------
    Tensor
        Reconstructed vectors with the same shape as the original input.
    """
    mse = quantized.mse_component
    n = mse.num_vectors
    d = mse.dim

    # 1. MSE reconstruction (unit sphere)
    total = n * d
    indices = unpack_indices(mse.packed_indices, mse.bits, total)
    indices = indices.reshape(n, d)
    reconstructed_rotated = _scalar_dequantize(indices, mse.centroids)
    mse_unit = inverse_rotate(reconstructed_rotated, mse.rotation_matrix)

    # 2. QJL dequantisation
    bytes_per_vector = (d + 7) // 8
    qjl_parts = []
    for i in range(n):
        start = i * bytes_per_vector
        end = start + bytes_per_vector
        signs_i = unpack_signs(quantized.packed_signs[start:end], d)
        qjl_vec = qjl_dequantize(signs_i, quantized.qjl_matrix)
        qjl_parts.append(quantized.residual_norms[i] * qjl_vec)
    qjl_reconstruction = torch.stack(qjl_parts)

    # 3. Combine
    unit_reconstruction = mse_unit + qjl_reconstruction

    # 4. Rescale
    reconstructed = unit_reconstruction * mse.norms.unsqueeze(1)

    return reconstructed.reshape(mse.original_shape)
