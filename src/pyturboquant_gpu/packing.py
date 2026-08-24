"""
Bit-packing and unpacking utilities using PyTorch operations.
"""

from __future__ import annotations

import torch


def pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack integer indices into a compact ``uint8`` byte tensor.

    Parameters
    ----------
    indices : Tensor of long/int
        Flat tensor of index values in ``[0, 2**bits)``.
    bits : int
        Number of bits per index (1–8).

    Returns
    -------
    packed : Tensor of uint8
    """
    indices = indices.reshape(-1).to(torch.int32)
    device = indices.device

    if bits == 8:
        return indices.to(torch.uint8)

    if bits == 4:
        n = indices.shape[0]
        if n % 2 != 0:
            indices = torch.cat(
                [indices, torch.zeros(1, dtype=torch.int32, device=device)]
            )
        low = indices[0::2] & 0x0F
        high = (indices[1::2] & 0x0F) << 4
        return (low | high).to(torch.uint8)

    if bits == 2:
        n = indices.shape[0]
        pad_len = (4 - n % 4) % 4
        if pad_len:
            indices = torch.cat(
                [indices, torch.zeros(pad_len, dtype=torch.int32, device=device)]
            )
        a = indices[0::4] & 0x03
        b = (indices[1::4] & 0x03) << 2
        c = (indices[2::4] & 0x03) << 4
        d = (indices[3::4] & 0x03) << 6
        return (a | b | c | d).to(torch.uint8)

    if bits == 1:
        return pack_signs_from_bits(indices.to(torch.uint8))

    # General fallback: bit-by-bit
    n = indices.shape[0]
    total_bits = n * bits
    total_bytes = (total_bits + 7) // 8
    packed = torch.zeros(total_bytes, dtype=torch.uint8, device=device)
    for i in range(n):
        val = indices[i].item()
        for b in range(bits):
            flat_pos = i * bits + b
            byte_idx = flat_pos // 8
            bit_idx = flat_pos % 8
            packed[byte_idx] |= ((val >> b) & 1) << bit_idx
    return packed


def unpack_indices(
    packed: torch.Tensor, bits: int, count: int
) -> torch.Tensor:
    """Unpack integer indices from a packed ``uint8`` byte tensor.

    Parameters
    ----------
    packed : Tensor of uint8
    bits : int
    count : int

    Returns
    -------
    indices : Tensor of int64
    """
    packed = packed.reshape(-1)
    device = packed.device

    if bits == 8:
        return packed[:count].to(torch.int64)

    if bits == 4:
        low = (packed & 0x0F).to(torch.int64)
        high = ((packed >> 4) & 0x0F).to(torch.int64)
        interleaved = torch.empty(
            packed.shape[0] * 2, dtype=torch.int64, device=device
        )
        interleaved[0::2] = low
        interleaved[1::2] = high
        return interleaved[:count]

    if bits == 2:
        a = (packed & 0x03).to(torch.int64)
        b = ((packed >> 2) & 0x03).to(torch.int64)
        c = ((packed >> 4) & 0x03).to(torch.int64)
        d = ((packed >> 6) & 0x03).to(torch.int64)
        interleaved = torch.empty(
            packed.shape[0] * 4, dtype=torch.int64, device=device
        )
        interleaved[0::4] = a
        interleaved[1::4] = b
        interleaved[2::4] = c
        interleaved[3::4] = d
        return interleaved[:count]

    if bits == 1:
        return unpack_signs_to_bits(packed, count).to(torch.int64)

    # General fallback
    indices = torch.zeros(count, dtype=torch.int64, device=device)
    for i in range(count):
        val = 0
        for b_idx in range(bits):
            flat_pos = i * bits + b_idx
            byte_idx = flat_pos // 8
            bit_idx = flat_pos % 8
            val |= ((packed[byte_idx].item() >> bit_idx) & 1) << b_idx
        indices[i] = val
    return indices


def pack_signs_from_bits(bits_tensor: torch.Tensor) -> torch.Tensor:
    """Pack a {0,1} bit tensor into bytes.

    Parameters
    ----------
    bits_tensor : Tensor of uint8
        Tensor of 0/1 values.

    Returns
    -------
    packed : Tensor of uint8
    """
    bits_tensor = bits_tensor.reshape(-1).to(torch.uint8)
    device = bits_tensor.device
    n = bits_tensor.shape[0]
    pad_len = (8 - n % 8) % 8
    if pad_len:
        bits_tensor = torch.cat(
            [bits_tensor, torch.zeros(pad_len, dtype=torch.uint8, device=device)]
        )
    reshaped = bits_tensor.reshape(-1, 8)
    multipliers = (1 << torch.arange(8, device=device)).to(torch.uint8)
    return (reshaped * multipliers).sum(dim=1).to(torch.uint8)


def unpack_signs_to_bits(
    packed: torch.Tensor, count: int
) -> torch.Tensor:
    """Unpack bytes into a {0,1} bit tensor.

    Parameters
    ----------
    packed : Tensor of uint8
    count : int

    Returns
    -------
    bits : Tensor of uint8
    """
    packed = packed.reshape(-1)
    device = packed.device
    shifts = torch.arange(8, device=device)
    # packed[:, None] >> shifts[None, :] & 1
    unpacked = ((packed.unsqueeze(1) >> shifts.unsqueeze(0)) & 1).to(
        torch.uint8
    )
    return unpacked.reshape(-1)[:count]


def pack_signs(signs: torch.Tensor) -> torch.Tensor:
    """Pack a {-1, +1} sign tensor into bytes.

    +1 → bit 1, -1 → bit 0.

    Parameters
    ----------
    signs : Tensor

    Returns
    -------
    packed : Tensor of uint8
    """
    bits = ((signs.reshape(-1).to(torch.int8) + 1) // 2).to(torch.uint8)
    return pack_signs_from_bits(bits)


def unpack_signs(packed: torch.Tensor, count: int) -> torch.Tensor:
    """Unpack bytes into a {-1, +1} sign tensor.

    Parameters
    ----------
    packed : Tensor of uint8
    count : int

    Returns
    -------
    signs : Tensor of float32
    """
    bits = unpack_signs_to_bits(packed, count)
    return bits.to(torch.float32) * 2.0 - 1.0
