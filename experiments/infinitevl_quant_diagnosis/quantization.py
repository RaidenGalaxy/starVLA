from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch
from torch import nn

from experiments.infinitevl_quant.inventory import LinearTarget
from experiments.infinitevl_quant.quantization import FakeQuantLinear, quantize_weight

from .config import DiagnosticScheme


def _parent_and_key(model: nn.Module, path: str) -> tuple[nn.Module, str]:
    if "." not in path:
        return model, path
    parent_path, key = path.rsplit(".", 1)
    return model.get_submodule(parent_path), key


@contextlib.contextmanager
def diagnostic_qdq_linears(
    model: nn.Module,
    targets: list[LinearTarget],
    scheme: DiagnosticScheme,
) -> Iterator[dict]:
    originals: dict[str, nn.Linear] = {}
    audit = {}
    try:
        for target in targets:
            source = target.module
            if scheme.weight_bits is None:
                restored = source.weight.detach().float()
                weight_audit = {
                    "mse": 0.0, "nmse": 0.0, "max_abs_error": 0.0, "groups": 0,
                }
                group_size = source.in_features
            else:
                group_size = (
                    source.in_features if scheme.per_output_channel else scheme.group_size
                )
                quantized = quantize_weight(
                    source.weight,
                    bits=scheme.weight_bits,
                    group_size=group_size,
                    zero_point=scheme.zero_point,
                )
                restored = quantized.dequantized
                weight_audit = {
                    "mse": quantized.mse,
                    "nmse": quantized.nmse,
                    "max_abs_error": quantized.max_abs_error,
                    "groups": quantized.groups,
                }
            replacement = FakeQuantLinear(
                source,
                restored,
                activation_bits=scheme.activation_bits,
                input_scale=None,
            )
            parent, key = _parent_and_key(model, target.name)
            originals[target.name] = source
            setattr(parent, key, replacement)
            audit[target.name] = {
                "scope": target.scope,
                "layer": target.layer,
                "group_size": group_size,
                "weight_bits": scheme.weight_bits,
                "activation_bits": scheme.activation_bits,
                **weight_audit,
            }
        yield {
            "scheme": scheme.to_dict(),
            "module_count": len(targets),
            "modules": audit,
            "fake_quantized": True,
            "deployment_kernel_claim": False,
        }
    finally:
        for name, source in originals.items():
            parent, key = _parent_and_key(model, name)
            setattr(parent, key, source)


def select_targets(
    targets: list[LinearTarget],
    *,
    scopes: set[str] | None = None,
    layers: set[int] | None = None,
    projections: set[str] | None = None,
) -> list[LinearTarget]:
    selected = []
    for target in targets:
        if scopes is not None and target.scope not in scopes:
            continue
        if layers is not None and target.layer not in layers:
            continue
        leaf = target.name.rsplit(".", 1)[-1]
        if projections is not None and leaf not in projections:
            continue
        selected.append(target)
    return selected

