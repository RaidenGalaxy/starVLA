from __future__ import annotations

import dataclasses

from torch import nn

from .config import QuantScope


@dataclasses.dataclass(frozen=True)
class SelectedLinear:
    name: str
    module: nn.Linear
    subsystem: str
    parameter_count: int


def _classify(name: str) -> str | None:
    lowered = name.lower()
    if lowered.endswith("lm_head") or "embed_tokens" in lowered:
        return None
    if "visual" in lowered or lowered.startswith("vision"):
        if "merger" in lowered:
            return "vision_merger"
        if any(part in lowered for part in ("qkv", "q_proj", "k_proj", "v_proj", "attn.proj")):
            return "vision_attention"
        return "vision_ffn"
    if "model.layers" in lowered or "language_model.layers" in lowered:
        if any(part in lowered for part in ("gate_proj", "up_proj", "down_proj", ".mlp.")):
            return "decoder_ffn"
        return "decoder_attention"
    return None


def select_linears(model: nn.Module, scope: QuantScope, *, strict_qwen_3b: bool = True) -> list[SelectedLinear]:
    selected: list[SelectedLinear] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        subsystem = _classify(name)
        if subsystem is None:
            continue
        if scope == QuantScope.DECODER_CANONICAL and not subsystem.startswith("decoder_"):
            continue
        selected.append(SelectedLinear(name, module, subsystem, module.weight.numel()))

    if strict_qwen_3b:
        expected = 252 if scope == QuantScope.DECODER_CANONICAL else 414
        if len(selected) != expected:
            sample = ", ".join(item.name for item in selected[:8])
            raise RuntimeError(
                f"Qwen2.5-VL-3B selector expected {expected} Linear modules, got {len(selected)}. "
                f"First selected modules: {sample}"
            )
    return selected


def inventory(selected: list[SelectedLinear]) -> dict:
    by_subsystem: dict[str, dict[str, int]] = {}
    modules = []
    for item in selected:
        shape = list(item.module.weight.shape)
        modules.append(
            {
                "name": item.name,
                "subsystem": item.subsystem,
                "shape": shape,
                "parameters": item.parameter_count,
                "dtype": str(item.module.weight.dtype),
            }
        )
        summary = by_subsystem.setdefault(item.subsystem, {"modules": 0, "parameters": 0})
        summary["modules"] += 1
        summary["parameters"] += item.parameter_count
    return {
        "linear_count": len(selected),
        "weight_count": sum(x.parameter_count for x in selected),
        "by_subsystem": by_subsystem,
        "modules": modules,
    }
