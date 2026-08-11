"""Paper-aligned quantization experiments for Qwen2.5-VL.

This package is intentionally independent from the earlier FFN/full-model
experiments.  It separates paper-faithful decoder experiments from extensions
that quantize the visual stack as well.
"""

from .config import ExperimentConfig, MethodName, QuantScope

__all__ = ["ExperimentConfig", "MethodName", "QuantScope"]

