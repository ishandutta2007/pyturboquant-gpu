"""
pyturboquant-gpu: GPU-accelerated TurboQuant vector quantization.

TurboQuant is a data-oblivious vector quantization algorithm from
"TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"
(Zandieh et al., arXiv:2504.19874).

This package uses PyTorch as the backend, enabling GPU acceleration
via CUDA while also supporting CPU-only operation.

Two quantization modes are provided:

- **MSE-optimal** (``quantize_mse`` / ``dequantize_mse``):
  Minimises mean-squared reconstruction error.
- **Inner-product-optimal** (``quantize_prod`` / ``dequantize_prod``):
  Gives an *unbiased* inner-product estimator by combining an MSE
  quantizer with a 1-bit QJL residual correction.
"""

from pyturboquant_gpu.quantizer import (
    QuantizedMSE,
    QuantizedProd,
    quantize_mse,
    dequantize_mse,
    quantize_prod,
    dequantize_prod,
)

__version__ = "0.1.0"

__all__ = [
    "quantize_mse",
    "dequantize_mse",
    "quantize_prod",
    "dequantize_prod",
    "QuantizedMSE",
    "QuantizedProd",
    "__version__",
]
