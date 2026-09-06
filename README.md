<div align="center">
  <img src="./assets/banner.svg" alt="pyturboquant-gpu banner" width="100%">
</div>

<p align="center">
  <a href="https://github.com/ishandutta2007/Awesome-Awesome-Awesome"><img src="https://img.shields.io/badge/Awesome-%E2%9C%94-blueviolet?style=flat-square&logo=github" alt="Awesome"/></a><a href="https://discord.gg/jc4xtF58Ve"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord" /></a>
  <a href="https://pypi.org/project/pyturboquant-gpu/"><img src="https://img.shields.io/pypi/v/pyturboquant-gpu.svg?color=blue&style=flat-square" alt="PyPI version" /></a>
  <a href="https://pypi.org/project/pyturboquant-gpu/"><img src="https://img.shields.io/pypi/pyversions/pyturboquant-gpu.svg?style=flat-square" alt="Python Versions" /></a>
  <a href="https://github.com/ishandutta2007/pyturboquant-gpu/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square" alt="License" /></a>
  <a href="https://arxiv.org/abs/2504.19874"><img src="https://img.shields.io/badge/ICLR%202026-ArXiv%3A2504.19874-B31B1B.svg?style=flat-square" alt="ArXiv Paper" /></a>
  <a href="https://github.com/ishandutta2007"><img alt="GitHub followers" src="https://img.shields.io/github/followers/ishandutta2007?label=Follow" /></a>
</p>

# ⚡ pyturboquant-gpu

🚀 GPU-accelerated implementation of **TurboQuant**, a data-oblivious vector quantization algorithm for compressing high-dimensional vectors with near-optimal distortion. Built on **PyTorch** for seamless CUDA GPU acceleration.

📄 Based on the paper: [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) (Zandieh et al., ICLR 2026).

---

## 📦 Installation

```bash
pip install pyturboquant-gpu
```

Requires PyTorch ≥ 2.0. For development:

```bash
git clone https://github.com/pyturboquant/pyturboquant-gpu.git
cd pyturboquant-gpu
pip install -e ".[dev]"
```

---

## 🚀 Quick Start

### 🎯 MSE-Optimal Quantization

```python
import torch
from pyturboquant_gpu import quantize_mse, dequantize_mse

# Random vectors (e.g., KV cache embeddings) — works on CPU or GPU
vectors = torch.randn(100, 128)  # or vectors.cuda()

# Quantize at 3 bits per coordinate
quantized = quantize_mse(vectors, bits=3, seed=42)

# Reconstruct
reconstructed = dequantize_mse(quantized)

mse = torch.mean(torch.sum((vectors - reconstructed) ** 2, dim=1))
print(f"MSE: {mse:.4f}")
```

### 📐 Inner-Product-Optimal Quantization

Provides **unbiased** inner product estimates — essential for attention and nearest-neighbor search:

```python
from pyturboquant_gpu import quantize_prod, dequantize_prod

# 4 bits total (3 bits MSE + 1 bit QJL correction)
quantized = quantize_prod(vectors, bits=4, seed=42)
reconstructed = dequantize_prod(quantized)

# Inner products are unbiased: E[⟨y, x̃⟩] = ⟨y, x⟩
query = torch.randn(128)
true_ip = vectors @ query
approx_ip = reconstructed @ query
print(f"Mean IP error: {torch.mean(torch.abs(true_ip - approx_ip)):.4f}")
```

### ⚡ GPU Usage

```python
# Move to GPU — all operations automatically use CUDA
vectors_gpu = vectors.cuda()
quantized = quantize_mse(vectors_gpu, bits=3, seed=42)
reconstructed = dequantize_mse(quantized)  # result is on GPU
```

---

## 🧠 How It Works

TurboQuant is a **data-oblivious** algorithm — no training or calibration needed:

1. 🔄 **Random Rotation**: Multiply by a random orthogonal matrix → coordinates follow a known Beta distribution
2. 📊 **Lloyd-Max Scalar Quantization**: Each coordinate independently quantized with a precomputed optimal codebook
3. ✨ **QJL Residual Correction** (Prod mode): 1-bit Quantized Johnson-Lindenstrauss sketch removes inner-product bias

### 📉 Theoretical Distortion Bounds (unit vectors)

| Bits | MSE Distortion | Inner Product Distortion |
|:----:|:--------------:|:------------------------:|
| 1    | ≈ 0.36        | ≈ 1.57/d                |
| 2    | ≈ 0.117       | ≈ 0.56/d                |
| 3    | ≈ 0.03        | ≈ 0.18/d                |
| 4    | ≈ 0.009       | ≈ 0.047/d               |

---

## 📚 API Reference

### 🔹 `quantize_mse(vectors, bits, dim=None, seed=None, device=None)`

- **vectors**: Tensor of shape `(..., d)`
- **bits**: int in `[1, 8]`
- **seed**: int or None
- **device**: torch.device or None
- **Returns**: `QuantizedMSE`

### 🔹 `dequantize_mse(quantized)` → Tensor

### 🔹 `quantize_prod(vectors, bits, dim=None, seed=None, device=None)`

- **bits**: int in `[2, 8]`
- **Returns**: `QuantizedProd`

### 🔹 `dequantize_prod(quantized)` → Tensor

---

## 💻 CPU Version

For a NumPy/SciPy-based CPU-only package:

```bash
pip install pyturboquant-cpu
```

---

## 📜 Citation

```bibtex
@article{zandieh2025turboquant,
  title={TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate},
  author={Zandieh, Amir and Daliri, Majid and Hadian, Majid and Mirrokni, Vahab},
  journal={arXiv preprint arXiv:2504.19874},
  year={2025}
}
```

---

##  Star History
[![Star History Chart](https://star-history.dera.page/svg?repos=ishandutta2007/pyturboquant-gpu&type=date&legend=top-left)](https://star-history.dera.page/#ishandutta2007/pyturboquant-gpu&type=date&legend=top-left)

---

## 📄 License

Apache 2.0

