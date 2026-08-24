"""Tests for the QJL module (GPU package)."""

import math
import pytest
import torch

from pyturboquant_gpu.qjl import (
    generate_qjl_matrix,
    qjl_dequantize,
    qjl_quantize,
)


class TestQJLMatrix:
    def test_shape(self):
        S = generate_qjl_matrix(64, seed=0)
        assert S.shape == (64, 64)

    def test_reproducibility(self):
        S1 = generate_qjl_matrix(32, seed=42)
        S2 = generate_qjl_matrix(32, seed=42)
        torch.testing.assert_close(S1, S2)


class TestQJLUnbiasedness:
    def test_unbiased_inner_product(self):
        d = 128
        torch.manual_seed(0)
        x = torch.randn(d, dtype=torch.float64)
        x = x / torch.linalg.norm(x)
        y = torch.randn(d, dtype=torch.float64)
        true_ip = torch.dot(y, x).item()

        n_trials = 500
        estimates = []
        for t in range(n_trials):
            S = generate_qjl_matrix(d, seed=t, dtype=torch.float64)
            signs = qjl_quantize(x.unsqueeze(0), S)[0]
            x_hat = qjl_dequantize(signs, S)
            estimates.append(torch.dot(y, x_hat).item())

        mean_est = sum(estimates) / len(estimates)
        std_est = (sum((e - mean_est) ** 2 for e in estimates) / len(estimates)) ** 0.5
        se = std_est / math.sqrt(n_trials)

        assert abs(mean_est - true_ip) < 4 * se, (
            f"Bias: E[est]={mean_est:.4f}, true={true_ip:.4f}, SE={se:.4f}"
        )
