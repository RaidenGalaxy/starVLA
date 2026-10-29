"""Matched CPU protocol for InfiniteVL/Qwen FFN quantization causality."""

from .config import (
    EXPECTED_FFN_LINEAR_COUNT,
    EXPECTED_HIDDEN_LAYERS,
    FFN_PROJECTIONS,
    CausalConfig,
    ModelSpec,
)

__all__ = [
    "CausalConfig",
    "EXPECTED_FFN_LINEAR_COUNT",
    "EXPECTED_HIDDEN_LAYERS",
    "FFN_PROJECTIONS",
    "ModelSpec",
]
