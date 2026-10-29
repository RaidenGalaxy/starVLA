"""Strong-causal FFN quantization mechanism protocol.

The package is deliberately separate from :mod:`experiments.ffn_quant_causal`
so the completed v3 evidence bundle remains immutable and content-addressable.
"""

from .config import MechanismConfig

__all__ = ["MechanismConfig"]
