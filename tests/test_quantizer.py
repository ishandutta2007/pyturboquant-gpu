"""Tests for the core TurboQuant quantiser (GPU package)."""

import pytest
import torch

from pyturboquant_gpu.quantizer import (
    QuantizedMSE,
    QuantizedProd,
    dequantize_mse,
    dequantize_prod,
    quantize_mse,
    quantize_prod,
)


class TestQuantizeMSE:
    def test_output_type(self):
        x = torch.randn(10, 64)
        result = quantize_mse(x, bits=2, seed=42)
        assert isinstance(result, QuantizedMSE)

    def test_reconstructed_shape(self):
        x = torch.randn(5, 128)
        q = quantize_mse(x, bits=3, seed=0)
        x_hat = dequantize_mse(q)
        assert x_hat.shape == x.shape

    def test_single_vector(self):
        x = torch.randn(64)
        q = quantize_mse(x, bits=4, seed=0)
        x_hat = dequantize_mse(q)
        assert x_hat.shape == x.shape

    @pytest.mark.parametrize("bits", [1, 2, 3, 4])
    def test_mse_decreases_with_bits(self, bits):
        d = 128
        torch.manual_seed(42)
        X = torch.randn(50, d)
        X = X / torch.linalg.norm(X, dim=1, keepdim=True)

        mses = []
        for b in range(1, 5):
            q = quantize_mse(X, bits=b, seed=0)
            X_hat = dequantize_mse(q)
            mse = torch.mean(torch.sum((X - X_hat) ** 2, dim=1)).item()
            mses.append(mse)

        for i in range(len(mses) - 1):
            assert mses[i] > mses[i + 1]

    def test_reproducibility(self):
        x = torch.randn(10, 64)
        q1 = quantize_mse(x, bits=3, seed=99)
        q2 = quantize_mse(x, bits=3, seed=99)
        x_hat1 = dequantize_mse(q1)
        x_hat2 = dequantize_mse(q2)
        torch.testing.assert_close(x_hat1, x_hat2)

    def test_zero_vector(self):
        x = torch.zeros(3, 64)
        q = quantize_mse(x, bits=2, seed=0)
        x_hat = dequantize_mse(q)
        torch.testing.assert_close(x_hat, x, atol=1e-6, rtol=1e-6)

    def test_invalid_bits(self):
        x = torch.randn(10, 32)
        with pytest.raises(ValueError):
            quantize_mse(x, bits=0)
        with pytest.raises(ValueError):
            quantize_mse(x, bits=9)


class TestQuantizeProd:
    def test_output_type(self):
        x = torch.randn(10, 64)
        result = quantize_prod(x, bits=3, seed=42)
        assert isinstance(result, QuantizedProd)

    def test_reconstructed_shape(self):
        x = torch.randn(5, 128)
        q = quantize_prod(x, bits=3, seed=0)
        x_hat = dequantize_prod(q)
        assert x_hat.shape == x.shape

    def test_mse_component_uses_fewer_bits(self):
        x = torch.randn(5, 64)
        q = quantize_prod(x, bits=4, seed=0)
        assert q.mse_component.bits == 3

    def test_invalid_bits(self):
        x = torch.randn(10, 32)
        with pytest.raises(ValueError):
            quantize_prod(x, bits=1)

    def test_unbiased_inner_product(self):
        d = 128
        torch.manual_seed(0)
        x = torch.randn(d, dtype=torch.float64)
        x = x / torch.linalg.norm(x)
        y = torch.randn(d, dtype=torch.float64)
        true_ip = torch.dot(y, x).item()

        n_trials = 200
        estimates = []
        for t in range(n_trials):
            q = quantize_prod(x, bits=3, seed=t)
            x_hat = dequantize_prod(q)
            estimates.append(torch.dot(y, x_hat.reshape(-1).to(torch.float64)).item())

        import math
        mean_est = sum(estimates) / len(estimates)
        std_est = (sum((e - mean_est)**2 for e in estimates) / len(estimates))**0.5
        se = std_est / math.sqrt(n_trials)

        assert abs(mean_est - true_ip) < 4 * se, (
            f"Bias: E[est]={mean_est:.4f}, true={true_ip:.4f}, SE={se:.4f}"
        )
