"""Memory-safe public method entry point.

The compatibility implementation in :mod:`methods` stores rich tensor results.
This adapter prevents ``dataclasses.asdict`` from deep-copying multi-gigabyte
weights when serializing RTN audit metadata.
"""
from __future__ import annotations

from torch import nn

from . import methods as _implementation
from .config import ExperimentConfig
from .quantization import quantize_weight_groupwise


def _memory_safe_rtn(source: nn.Linear, config: ExperimentConfig):
    result = quantize_weight_groupwise(
        source.weight,
        bits=config.weight_bits,
        group_size=config.group_size,
        zero_point=config.zero_point,
    )
    if config.activation_bits is None:
        wrapper = _implementation.FrozenLinear(source, result.dequantized)
    else:
        wrapper = _implementation.W8A8Linear(source, result.dequantized)
    audit = {
        "bits": result.bits,
        "group_size": result.group_size,
        "mse": result.mse,
        "nmse": result.nmse,
        "sqnr_db": result.sqnr_db,
        "max_abs_error": result.max_abs_error,
        "scale_dtype": str(result.scales.dtype),
        "zero_point": result.zero_points is not None,
    }
    return wrapper, audit


# apply_method resolves _rtn from its defining module at call time.
_implementation._rtn = _memory_safe_rtn
apply_method = _implementation.apply_method
AppliedQuantization = _implementation.AppliedQuantization
