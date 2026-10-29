from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from experiments.infinitevl_quant.inventory import (
    LinearTarget,
    build_inventory as build_infinitevl_inventory,
    decoder_layers,
    select_scope as select_infinitevl_scope,
)
from experiments.infinitevl_quant.loader import load_infinitevl_cpu
from experiments.vlm_quant_benchmark.runner import load_reference as load_qwen_reference


MODEL_KINDS = ("infinitevl", "qwen")
FFN_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
EXPECTED_FFN_COUNT = 36 * len(FFN_PROJECTIONS)
EXPECTED_FFN_WEIGHTS = 2_434_793_472


@dataclass
class ModelBundle:
    kind: str
    model: nn.Module
    processor: Any
    ffn_targets: list[LinearTarget]
    load_audit: dict
    decoder_prefix: str

    def close(self) -> None:
        del self.model
        del self.processor
        gc.collect()


def _decoder_prefix(model: nn.Module, layers: nn.ModuleList) -> str:
    for name, module in model.named_modules():
        if module is layers:
            return name
    raise RuntimeError("decoder ModuleList is not present in named_modules")


def build_ffn_targets(model: nn.Module, *, strict: bool = True) -> tuple[list[LinearTarget], dict]:
    layers = decoder_layers(model)
    prefix = _decoder_prefix(model, layers)
    targets: list[LinearTarget] = []
    for index, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            raise RuntimeError(f"decoder layer {index} has no mlp")
        for projection in FFN_PROJECTIONS:
            module = getattr(mlp, projection, None)
            if not isinstance(module, nn.Linear):
                raise RuntimeError(f"decoder layer {index} missing Linear mlp.{projection}")
            targets.append(LinearTarget(
                name=f"{prefix}.{index}.mlp.{projection}",
                module=module,
                scope="decoder_ffn",
                layer=index,
            ))
    weight_count = sum(target.module.weight.numel() for target in targets)
    if strict and (len(targets) != EXPECTED_FFN_COUNT or weight_count != EXPECTED_FFN_WEIGHTS):
        raise RuntimeError(
            "FFN inventory mismatch: "
            f"linears={len(targets)} expected={EXPECTED_FFN_COUNT}, "
            f"weights={weight_count} expected={EXPECTED_FFN_WEIGHTS}"
        )
    inventory = {
        "decoder_prefix": prefix,
        "linear_count": len(targets),
        "weight_count": weight_count,
        "modules": [{
            "name": target.name,
            "layer": target.layer,
            "projection": target.name.rsplit(".", 1)[-1],
            "shape": list(target.module.weight.shape),
            "parameters": target.module.weight.numel(),
            "dtype": str(target.module.weight.dtype),
            "bias": target.module.bias is not None,
        } for target in targets],
    }
    return targets, inventory


def _finite_parameter_audit(model: nn.Module) -> dict:
    dtypes: dict[str, int] = {}
    nonfinite: list[str] = []
    for name, parameter in model.named_parameters():
        dtypes[str(parameter.dtype)] = dtypes.get(str(parameter.dtype), 0) + parameter.numel()
        if not torch.isfinite(parameter).all():
            nonfinite.append(name)
    if nonfinite:
        raise FloatingPointError(f"model contains non-finite parameters: {nonfinite[:8]}")
    return {"parameter_dtypes": dtypes, "nonfinite_parameters": nonfinite}


def load_model_bundle(kind: str, model_path: Path) -> ModelBundle:
    if kind not in MODEL_KINDS:
        raise ValueError(f"unknown model kind: {kind}")
    resolved = model_path.resolve()
    if kind == "infinitevl":
        model, processor, load_audit = load_infinitevl_cpu(resolved)
    else:
        model, processor, load_audit = load_qwen_reference(str(resolved))
    model.eval()
    targets, inventory = build_ffn_targets(model)
    layers = decoder_layers(model)
    prefix = _decoder_prefix(model, layers)
    load_audit = {
        **load_audit,
        **_finite_parameter_audit(model),
        "model_kind": kind,
        "model_path": str(resolved),
        "ffn_inventory": inventory,
    }
    return ModelBundle(kind, model, processor, targets, load_audit, prefix)


def infinitevl_attention_targets(model: nn.Module, *, include_dynamics: bool) -> list[LinearTarget]:
    targets, _inventory = build_infinitevl_inventory(model)
    return select_infinitevl_scope(targets, "attn_all" if include_dynamics else "attn_main")

