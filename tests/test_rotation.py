"""Tests for random rotation (GPU package)."""

import pytest
import torch

from pyturboquant_gpu.rotation import (
    generate_rotation_matrix,
    inverse_rotate,
    rotate,
)


class TestGenerateRotationMatrix:
    @pytest.mark.parametrize("dim", [4, 16, 64])
    def test_orthogonality(self, dim):
        Q = generate_rotation_matrix(dim, seed=42)
        identity = Q.T @ Q
        torch.testing.assert_close(
            identity, torch.eye(dim, dtype=Q.dtype), atol=1e-5, rtol=1e-5
        )

    @pytest.mark.parametrize("dim", [4, 16, 64])
    def test_determinant(self, dim):
        Q = generate_rotation_matrix(dim, seed=42)
        det = torch.linalg.det(Q)
        assert abs(abs(det.item()) - 1.0) < 1e-4

    def test_reproducibility(self):
        Q1 = generate_rotation_matrix(32, seed=123)
        Q2 = generate_rotation_matrix(32, seed=123)
        torch.testing.assert_close(Q1, Q2)


class TestRotateInverseRotate:
    def test_roundtrip(self):
        d = 64
        torch.manual_seed(0)
        X = torch.randn(20, d)
        Q = generate_rotation_matrix(d, seed=7)
        Y = rotate(X, Q)
        X_rec = inverse_rotate(Y, Q)
        torch.testing.assert_close(X_rec, X, atol=1e-5, rtol=1e-5)

    def test_norm_preservation(self):
        d = 64
        torch.manual_seed(0)
        X = torch.randn(20, d)
        Q = generate_rotation_matrix(d, seed=5)
        Y = rotate(X, Q)
        orig = torch.linalg.norm(X, dim=1)
        rot = torch.linalg.norm(Y, dim=1)
        torch.testing.assert_close(rot, orig, atol=1e-5, rtol=1e-5)
