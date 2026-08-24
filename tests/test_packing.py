"""Tests for bit-packing utilities (GPU package)."""

import pytest
import torch

from pyturboquant_gpu.packing import (
    pack_indices,
    pack_signs,
    unpack_indices,
    unpack_signs,
)


class TestPackUnpackIndices:
    @pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_roundtrip(self, bits):
        torch.manual_seed(42)
        n = 100
        max_val = 1 << bits
        indices = torch.randint(0, max_val, (n,), dtype=torch.int64)
        packed = pack_indices(indices, bits)
        recovered = unpack_indices(packed, bits, n)
        assert torch.equal(recovered, indices)


class TestPackUnpackSigns:
    def test_roundtrip(self):
        signs = torch.tensor(
            [1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1],
            dtype=torch.float32,
        )
        packed = pack_signs(signs)
        recovered = unpack_signs(packed, 16)
        torch.testing.assert_close(recovered, signs)

    def test_roundtrip_odd_count(self):
        signs = torch.tensor([1, -1, 1, 1, -1], dtype=torch.float32)
        packed = pack_signs(signs)
        recovered = unpack_signs(packed, 5)
        torch.testing.assert_close(recovered, signs)
