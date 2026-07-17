"""CPU-friendly decoder-FFN fake-quantization experiments."""

from .quantization import QuantizationStats, fake_quantize_weight

__all__ = ["QuantizationStats", "fake_quantize_weight"]

