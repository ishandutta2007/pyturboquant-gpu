"""End-to-end roundtrip tests for TurboQuant (GPU package)."""

import pytest
import torch

from pyturboquant_gpu import (
    dequantize_mse,
    dequantize_prod,
    quantize_mse,
    quantize_prod,
)


class TestEndToEnd:
    def test_mse_roundtrip_quality(self):
        d = 128
        torch.manual_seed(0)
        X = torch.randn(100, d)
        q = quantize_mse(X, bits=4, seed=0)
        X_hat = dequantize_mse(q)

        norms = torch.linalg.norm(X, dim=1)
        errors = torch.linalg.norm(X - X_hat, dim=1)
        rel = errors / torch.clamp(norms, min=1e-10)
        assert rel.mean().item() < 0.55

    def test_prod_roundtrip_quality(self):
        d = 128
        torch.manual_seed(0)
        X = torch.randn(50, d)
        q = quantize_prod(X, bits=4, seed=0)
        X_hat = dequantize_prod(q)

        norms = torch.linalg.norm(X, dim=1)
        errors = torch.linalg.norm(X - X_hat, dim=1)
        rel = errors / torch.clamp(norms, min=1e-10)
        assert rel.mean().item() < 0.85

    def test_3d_input_shape(self):
        shape = (4, 8, 64)
        X = torch.randn(*shape)
        q = quantize_mse(X, bits=3, seed=0)
        X_hat = dequantize_mse(q)
        assert X_hat.shape == shape

    def test_compression_ratio(self):
        d, n = 128, 100
        X = torch.randn(n, d)
        q = quantize_mse(X, bits=2, seed=0)

        original_bytes = X.nelement() * X.element_size()
        packed_bytes = (
            q.packed_indices.nelement() * q.packed_indices.element_size()
            + q.norms.nelement() * q.norms.element_size()
        )
        assert packed_bytes < original_bytes / 5

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_mse_theoretical_values(self, bits):
        theoretical = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}

        d = 256
        torch.manual_seed(42)
        n = 300
        X = torch.randn(n, d, dtype=torch.float64)
        X = X / torch.linalg.norm(X, dim=1, keepdim=True)

        q = quantize_mse(X, bits=bits, seed=0)
        X_hat = dequantize_mse(q)

        mse = torch.mean(torch.sum((X - X_hat) ** 2, dim=1)).item()
        assert mse < theoretical[bits] * 2.5, (
            f"bits={bits}: MSE={mse:.4f}, bound={theoretical[bits] * 2.5:.4f}"
        )

    def test_import_api(self):
        from pyturboquant_gpu import __version__
        assert isinstance(__version__, str)
