from __future__ import annotations

import dataclasses
from collections import Counter

from torch import nn

from .config import EXPECTED_COUNTS


@dataclasses.dataclass(frozen=True)
class LinearTarget:
    name: str
    module: nn.Linear
    scope: str
    layer: int

    @property
    def parameter_count(self) -> int:
        return self.module.weight.numel() + (
            0 if self.module.bias is None else self.module.bias.numel()
        )


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    for path in (
        "model.language_model.layers", "language_model.layers", "model.layers",
    ):
        try:
            value = model.get_submodule(path)
        except AttributeError:
            continue
        if isinstance(value, nn.ModuleList):
            return value
    raise TypeError("could not locate InfiniteVL decoder layers")


def _classify(local_name: str, layer_type: str) -> str | None:
    if local_name in {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"}:
        return "decoder_ffn"
    if not local_name.startswith("self_attn."):
        return None
    leaf = local_name.rsplit(".", 1)[-1]
    if layer_type == "sliding_attention" and leaf in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return "swa_attention"
    if layer_type == "linear_attention":
        if leaf in {"q_proj", "k_proj", "v_proj", "g_proj", "o_proj"}:
            return "gdn_main"
        if leaf in {"a_proj", "b_proj"}:
            return "gdn_dynamics"
    return None


def build_inventory(model: nn.Module, *, strict: bool = True) -> tuple[list[LinearTarget], dict]:
    layers = decoder_layers(model)
    targets: list[LinearTarget] = []
    for index, layer in enumerate(layers):
        layer_type = getattr(layer, "layer_type", None)
        if layer_type not in {"sliding_attention", "linear_attention"}:
            raise RuntimeError(f"unexpected InfiniteVL layer type at {index}: {layer_type}")
        prefix = next(
            name for name, module in model.named_modules() if module is layer
        )
        for local_name, module in layer.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            scope = _classify(local_name, layer_type)
            if scope is None:
                raise RuntimeError(
                    f"unclassified decoder Linear at {prefix}.{local_name} ({layer_type})"
                )
            targets.append(LinearTarget(
                f"{prefix}.{local_name}", module, scope, index,
            ))
    counts = Counter(target.scope for target in targets)
    counts["decoder_total"] = len(targets)
    counts["decoder_safe"] = sum(
        target.scope != "gdn_dynamics" for target in targets
    )
    if strict:
        mismatches = {
            key: (counts[key], expected) for key, expected in EXPECTED_COUNTS.items()
            if counts[key] != expected
        }
        if mismatches:
            raise RuntimeError(f"InfiniteVL decoder inventory mismatch: {mismatches}")
    total_model = sum(parameter.numel() for parameter in model.parameters())
    by_scope = {}
    for scope in ("decoder_ffn", "swa_attention", "gdn_main", "gdn_dynamics"):
        members = [target for target in targets if target.scope == scope]
        parameters = sum(target.parameter_count for target in members)
        by_scope[scope] = {
            "linears": len(members), "parameters": parameters,
            "model_parameter_fraction": parameters / total_model,
        }
    audit = {
        "counts": dict(counts),
        "by_scope": by_scope,
        "total_model_parameters": total_model,
        "modules": [{
            "name": target.name,
            "scope": target.scope,
            "layer": target.layer,
            "weight_shape": list(target.module.weight.shape),
            "parameters": target.parameter_count,
            "dtype": str(target.module.weight.dtype),
        } for target in targets],
    }
    return targets, audit


def select_scope(targets: list[LinearTarget], scope: str) -> list[LinearTarget]:
    aliases = {
        "decoder_safe": {"decoder_ffn", "swa_attention", "gdn_main"},
        "decoder_all": {"decoder_ffn", "swa_attention", "gdn_main", "gdn_dynamics"},
        "attn_main": {"swa_attention", "gdn_main"},
        "attn_all": {"swa_attention", "gdn_main", "gdn_dynamics"},
        "ffn": {"decoder_ffn"},
        "swa": {"swa_attention"},
        "gdn_main": {"gdn_main"},
        "gdn_dynamics": {"gdn_dynamics"},
    }
    if scope not in aliases:
        raise ValueError(f"unknown quantization scope: {scope}")
    return [target for target in targets if target.scope in aliases[scope]]
