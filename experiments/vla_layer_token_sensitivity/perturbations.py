"""Reversible layer-skip and fake-quant perturbations for CPU diagnostics."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn

from experiments.vlm_quant_benchmark.quantization import quantize_weight_groupwise


@dataclass(frozen=True)
class QuantizationAudit:
    linears: int
    parameters: int
    weighted_mse: float


def _identity_layer_output(_module: nn.Module, args: tuple, output):
    if not args or not isinstance(args[0], torch.Tensor):
        raise TypeError("cannot identify the hidden-state input of the skipped layer")
    hidden = args[0]
    if isinstance(output, torch.Tensor):
        return hidden
    if isinstance(output, tuple):
        if not output:
            raise TypeError("cannot replace an empty layer output tuple")
        return (hidden, *output[1:])
    if isinstance(output, list):
        if not output:
            raise TypeError("cannot replace an empty layer output list")
        return [hidden, *output[1:]]
    raise TypeError(f"unsupported layer output type: {type(output).__name__}")


@contextmanager
def temporary_layer_skip(layer: nn.Module) -> Iterator[None]:
    """Patch one residual transformer block to an identity for one context."""

    handle = layer.register_forward_hook(_identity_layer_output)
    try:
        yield
    finally:
        handle.remove()


def quantize_module_inplace(
    module: nn.Module,
    *,
    bits: int,
    group_size: int = 128,
    zero_point: bool = True,
) -> QuantizationAudit:
    """Fake-quantize every Linear weight in ``module`` and return an audit."""

    linears = 0
    parameters = 0
    squared_error_sum = 0.0
    with torch.no_grad():
        for child in module.modules():
            if not isinstance(child, nn.Linear):
                continue
            reference = child.weight.detach()
            result = quantize_weight_groupwise(
                reference,
                bits=bits,
                group_size=group_size,
                zero_point=zero_point,
            )
            count = reference.numel()
            child.weight.copy_(result.dequantized.to(reference.dtype))
            linears += 1
            parameters += count
            squared_error_sum += result.mse * count
    if not linears:
        raise ValueError("module contains no Linear weights to quantize")
    return QuantizationAudit(
        linears=linears,
        parameters=parameters,
        weighted_mse=squared_error_sum / parameters,
    )


@contextmanager
def temporary_groupwise_weight_quantization(
    module: nn.Module,
    *,
    bits: int,
    group_size: int = 128,
    zero_point: bool = True,
) -> Iterator[QuantizationAudit]:
    """Apply fake quantization and restore the exact original weights."""

    backups = [
        (child, child.weight.detach().clone())
        for child in module.modules()
        if isinstance(child, nn.Linear)
    ]
    if not backups:
        raise ValueError("module contains no Linear weights to quantize")
    try:
        audit = quantize_module_inplace(
            module,
            bits=bits,
            group_size=group_size,
            zero_point=zero_point,
        )
        yield audit
    finally:
        with torch.no_grad():
            for child, weight in backups:
                child.weight.copy_(weight)
