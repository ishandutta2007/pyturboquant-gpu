"""Tests for the Lloyd-Max codebook (GPU package)."""

import numpy as np
import pytest
import torch

from pyturboquant_gpu.codebook import get_codebook, lloyd_max_codebook_np


class TestLloydMaxCodebook:
    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_correct_number_of_centroids(self, bits):
        centroids, boundaries = lloyd_max_codebook_np(128, bits, max_iter=100)
        assert centroids.shape == (2**bits,)
        assert boundaries.shape == (2**bits - 1,)

    @pytest.mark.parametrize("bits", [1, 2, 3])
    def test_centroids_symmetric(self, bits):
        centroids, _ = lloyd_max_codebook_np(128, bits, max_iter=200)
        n = len(centroids)
        for i in range(n // 2):
            assert abs(centroids[i] + centroids[n - 1 - i]) < 1e-6

    def test_centroids_sorted(self):
        centroids, _ = lloyd_max_codebook_np(64, 3)
        assert np.all(np.diff(centroids) > 0)


class TestGetCodebook:
    def test_returns_tensors(self):
        c, b = get_codebook(64, 2)
        assert isinstance(c, torch.Tensor)
        assert isinstance(b, torch.Tensor)

    def test_correct_device(self):
        c, b = get_codebook(64, 2, device=torch.device("cpu"))
        assert c.device == torch.device("cpu")

    def test_correct_dtype(self):
        c, b = get_codebook(64, 2, dtype=torch.float64)
        assert c.dtype == torch.float64
