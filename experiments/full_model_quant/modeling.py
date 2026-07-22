"""Strict Qwen2.5-VL module discovery and reversible FP16-weight reference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn as nn

GROUPS = (
    "decoder_ffn",
    "decoder_attention",
    "vision_ffn",
    "vision_attention",
    "vision_merger",
)

EXPECTED_COUNTS = {
    "decoder_ffn": 108,
    "decoder_attention": 144,
    "vision_ffn": 96,
    "vision_attention": 64,
    "vision_merger": 2,
}
EXPECTED_PARAMETERS = {
    "decoder_ffn": 2_434_793_472,
    "decoder_attention": 339_738_624,
    "vision_ffn": 420_249_600,
    "vision_attention": 209_715_200,
    "vision_merger": 36_700_160,
}


@dataclass(frozen=True)
class LinearTarget:
    name: str
    group: str
    module: nn.Linear
    layer: int | None


@dataclass(frozen=True)
class FFNTarget:
    domain: str
    index: int
    name: str
    norm_name: str
    mlp: nn.Module
    norm: nn.Module

    @property
    def gate(self) -> nn.Linear:
        return self.mlp.gate_proj

    @property
    def up(self) -> nn.Linear:
        return self.mlp.up_proj

    @property
    def down(self) -> nn.Linear:
        return self.mlp.down_proj

    @property
    def linears(self) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
        return self.gate, self.up, self.down

    def named_scale_parameters(self) -> Iterable[tuple[str, nn.Parameter]]:
        yield f"{self.norm_name}.weight", self.norm.weight
        if self.up.bias is not None:
            yield f"{self.name}.up_proj.bias", self.up.bias


def _language_layers(model: nn.Module) -> nn.ModuleList:
    language = getattr(getattr(model, "model", None), "language_model", None)
    layers = getattr(language, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise RuntimeError("Qwen language decoder layers were not found")
    return layers


def _vision(model: nn.Module) -> nn.Module:
    visual = getattr(getattr(model, "model", None), "visual", None)
    if not isinstance(visual, nn.Module):
        raise RuntimeError("Qwen visual encoder was not found")
    return visual


def discover_full_model(
    model: nn.Module, *, strict_qwen3b: bool = True
) -> tuple[list[LinearTarget], list[FFNTarget]]:
    """Discover supported Linear weights by structure, never by a broad name glob."""

    names = {id(module): name for name, module in model.named_modules()}
    targets: list[LinearTarget] = []
    ffns: list[FFNTarget] = []
    language_layers = _language_layers(model)
    visual = _vision(model)
    vision_blocks = getattr(visual, "blocks", None)
    if not isinstance(vision_blocks, nn.ModuleList):
        raise RuntimeError("Qwen visual blocks were not found")

    for index, layer in enumerate(language_layers):
        mlp = getattr(layer, "mlp", None)
        norm = getattr(layer, "post_attention_layernorm", None)
        attention = getattr(layer, "self_attn", None)
        if not isinstance(mlp, nn.Module) or not isinstance(norm, nn.Module):
            raise RuntimeError(f"decoder layer {index} has no supported SwiGLU FFN")
        ffn = FFNTarget("decoder", index, names[id(mlp)], names[id(norm)], mlp, norm)
        ffns.append(ffn)
        for projection in ("gate_proj", "up_proj", "down_proj"):
            linear = getattr(mlp, projection, None)
            if not isinstance(linear, nn.Linear):
                raise RuntimeError(f"decoder layer {index} missing {projection}")
            targets.append(LinearTarget(names[id(linear)], "decoder_ffn", linear, index))
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            linear = getattr(attention, projection, None)
            if not isinstance(linear, nn.Linear):
                raise RuntimeError(f"decoder layer {index} missing attention {projection}")
            targets.append(LinearTarget(names[id(linear)], "decoder_attention", linear, index))

    for index, block in enumerate(vision_blocks):
        mlp = getattr(block, "mlp", None)
        norm = getattr(block, "norm2", None)
        attention = getattr(block, "attn", None)
        if not isinstance(mlp, nn.Module) or not isinstance(norm, nn.Module):
            raise RuntimeError(f"vision block {index} has no supported SwiGLU FFN")
        ffn = FFNTarget("vision", index, names[id(mlp)], names[id(norm)], mlp, norm)
        ffns.append(ffn)
        for projection in ("gate_proj", "up_proj", "down_proj"):
            linear = getattr(mlp, projection, None)
            if not isinstance(linear, nn.Linear):
                raise RuntimeError(f"vision block {index} missing {projection}")
            targets.append(LinearTarget(names[id(linear)], "vision_ffn", linear, index))
        for projection in ("qkv", "proj"):
            linear = getattr(attention, projection, None)
            if not isinstance(linear, nn.Linear):
                raise RuntimeError(f"vision block {index} missing attention {projection}")
            targets.append(LinearTarget(names[id(linear)], "vision_attention", linear, index))

    merger = getattr(visual, "merger", None)
    merger_mlp = getattr(merger, "mlp", None)
    if not isinstance(merger_mlp, nn.Sequential):
        raise RuntimeError("Qwen visual merger MLP was not found")
    merger_linears = [module for module in merger_mlp if isinstance(module, nn.Linear)]
    if len(merger_linears) != 2:
        raise RuntimeError(f"expected two merger Linear modules, found {len(merger_linears)}")
    targets.extend(
        LinearTarget(names[id(linear)], "vision_merger", linear, None) for linear in merger_linears
    )

    if len({id(target.module) for target in targets}) != len(targets):
        raise RuntimeError("duplicate Linear target discovered")
    forbidden = {id(getattr(model, "lm_head", None)), id(model.get_input_embeddings())}
    if any(id(target.module) in forbidden for target in targets):
        raise RuntimeError("tied embedding/LM head was selected")

    if strict_qwen3b:
        if len(language_layers) != 36 or len(vision_blocks) != 32:
            raise RuntimeError(
                f"expected 36 decoder/32 vision layers, found {len(language_layers)}/{len(vision_blocks)}"
            )
        counts = {group: sum(target.group == group for target in targets) for group in GROUPS}
        parameters = {
            group: sum(target.module.weight.numel() for target in targets if target.group == group)
            for group in GROUPS
        }
        if counts != EXPECTED_COUNTS or parameters != EXPECTED_PARAMETERS:
            raise RuntimeError(f"unexpected Qwen2.5-VL-3B inventory: {counts}, {parameters}")
    return targets, ffns


def build_inventory(model: nn.Module, targets: list[LinearTarget], ffns: list[FFNTarget]) -> dict[str, Any]:
    total = sum(parameter.numel() for parameter in model.parameters())
    groups: dict[str, dict[str, Any]] = {}
    modules: list[dict[str, Any]] = []
    for group in GROUPS:
        selected = [target for target in targets if target.group == group]
        count = sum(target.module.weight.numel() for target in selected)
        groups[group] = {"linear_modules": len(selected), "parameters": count, "fraction": count / total}
    for target in targets:
        modules.append(
            {
                "name": target.name,
                "group": target.group,
                "layer": target.layer,
                "shape": list(target.module.weight.shape),
                "parameters": target.module.weight.numel(),
                "dtype": str(target.module.weight.dtype),
            }
        )
    quantized = sum(item["parameters"] for item in modules)
    return {
        "decoder_layers": sum(ffn.domain == "decoder" for ffn in ffns),
        "vision_layers": sum(ffn.domain == "vision" for ffn in ffns),
        "linear_modules": len(targets),
        "target_parameters": quantized,
        "total_parameters": total,
        "target_fraction": quantized / total,
        "groups": groups,
        "modules": modules,
    }


@torch.no_grad()
def make_fp16_weight_reference_(model: nn.Module) -> dict[str, Any]:
    """Round every floating parameter through FP16, then retain FP32 compute."""

    changed = 0
    values = 0
    max_error = 0.0
    per_dtype: dict[str, int] = {}
    for name, parameter in model.named_parameters():
        if not parameter.is_floating_point():
            continue
        original = parameter.detach().float()
        rounded_half = original.half()
        if not torch.isfinite(rounded_half).all():
            raise ValueError(f"FP16 reference conversion overflowed for {name}")
        rounded = rounded_half.float()
        error = (rounded - original).abs()
        changed += int((rounded != original).sum().item())
        values += parameter.numel()
        max_error = max(max_error, float(error.max().item()) if error.numel() else 0.0)
        per_dtype[str(parameter.dtype)] = per_dtype.get(str(parameter.dtype), 0) + parameter.numel()
        parameter.data = rounded
    model.float()
    if any(parameter.is_floating_point() and parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("reference model contains a non-FP32 floating parameter after conversion")
    return {
        "reference": "fp16_rounded_fp32_compute",
        "floating_values": values,
        "changed_values": changed,
        "changed_fraction": changed / max(values, 1),
        "max_abs_rounding_error": max_error,
        "loaded_parameter_dtypes": per_dtype,
        "compute_dtype": "torch.float32",
    }


CONFIGS = (
    "fp16_weight_reference",
    "rtn_w8_all_linear",
    "rtn_w4_all_linear",
    "hybrid_awq_w4_all_linear",
    "rtn_w4_all_ffn_only",
    "awq_w4_all_ffn_only",
    "rtn_w4_vision_only",
    "rtn_w4_decoder_attention_only",
)


def config_assignment(config: str, targets: list[LinearTarget]) -> dict[str, str]:
    """Map module name to reference/rtn/awq for one fixed ablation."""

    if config not in CONFIGS:
        raise ValueError(f"unsupported config {config!r}")
    assignment = {target.name: "reference" for target in targets}
    if config == "fp16_weight_reference":
        return assignment
    bits = "rtn_w8" if config == "rtn_w8_all_linear" else "rtn_w4"
    if config in ("rtn_w8_all_linear", "rtn_w4_all_linear"):
        return {target.name: bits for target in targets}
    if config == "hybrid_awq_w4_all_linear":
        return {
            target.name: "awq_w4" if target.group in ("decoder_ffn", "vision_ffn") else "rtn_w4"
            for target in targets
        }
    if config in ("rtn_w4_all_ffn_only", "awq_w4_all_ffn_only"):
        mode = "awq_w4" if config.startswith("awq") else "rtn_w4"
        for target in targets:
            if target.group in ("decoder_ffn", "vision_ffn"):
                assignment[target.name] = mode
        return assignment
    if config == "rtn_w4_vision_only":
        for target in targets:
            if target.group.startswith("vision_"):
                assignment[target.name] = "rtn_w4"
        return assignment
    for target in targets:
        if target.group == "decoder_attention":
            assignment[target.name] = "rtn_w4"
    return assignment

