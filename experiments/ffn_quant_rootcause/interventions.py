from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from experiments.infinitevl_quant.quantization import quantize_weight

from .numerics import finite, matching_noise


@dataclasses.dataclass(frozen=True)
class Intervention:
    bits: int | None = 4
    layer_bits: dict[int, int | None] = dataclasses.field(default_factory=dict)
    projection_bits: dict[str, int | None] = dataclasses.field(default_factory=dict)
    gamma: float = 1.0
    layer_gammas: dict[int, float] = dataclasses.field(default_factory=dict)
    direction: str = "real"
    direction_layers: tuple[int, ...] = ()
    seed: int = 42
    channel_repairs: dict[int, tuple[int, ...]] = dataclasses.field(default_factory=dict)
    gdn_groups: tuple[str, ...] = ()
    carry: bool = False
    recurrent: bool = False

    def __post_init__(self):
        if any(v not in (None, 3, 4, 8) for v in [self.bits, *self.layer_bits.values(), *self.projection_bits.values()]):
            raise ValueError("unsupported projection precision")
        if self.direction not in ("real", "permuted", "gaussian"):
            raise ValueError("unsupported perturbation direction")
        if not all(math.isfinite(v) and v >= 0 for v in [self.gamma, *self.layer_gammas.values()]):
            raise ValueError("gammas must be finite and nonnegative")

    def payload(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_payload(cls, value):
        value = dict(value)
        for key in ("layer_bits", "layer_gammas", "channel_repairs"):
            value[key] = {int(k): tuple(v) if key == "channel_repairs" else v for k, v in value.get(key, {}).items()}
        for key in ("gdn_groups", "direction_layers"):
            value[key] = tuple(value.get(key, ()))
        return cls(**value)


class QuantBank:
    """Quantize once per model/projection/bit width, never change original weights."""

    def __init__(self, layers, group_size=128):
        self.sources = [layer.mlp for layer in layers]
        self.group_size = group_size
        self.cache = {}

    def weight(self, layer, projection, bits):
        source = getattr(self.sources[layer], projection + "_proj")
        if bits is None:
            return source.weight
        key = layer, projection, bits
        if key not in self.cache:
            self.cache[key] = quantize_weight(
                source.weight, bits=bits, group_size=self.group_size, zero_point=True
            ).dequantized
        return self.cache[key]

    def linear(self, layer, projection, bits, value):
        source = getattr(self.sources[layer], projection + "_proj")
        return F.linear(value, self.weight(layer, projection, bits), source.bias)


class ControlledFFN(nn.Module):
    def __init__(self, bank, layer, spec, observer, sample_id):
        super().__init__()
        self.source = bank.sources[layer]
        self.bank, self.layer, self.spec = bank, layer, spec
        self.observer, self.sample_id = observer, sample_id
        self.residual = None
        self.invocation = 0

    def _precision(self, projection):
        return self.spec.projection_bits.get(
            f"{self.layer}.{projection}", self.spec.layer_bits.get(self.layer, self.spec.bits)
        )

    def _reference(self, value):
        gate = self.source.gate_proj(value)
        up = self.source.up_proj(value)
        product = self.source.act_fn(gate) * up
        return {"gate": gate, "up": up, "product": product, "down_output": self.source.down_proj(product)}

    def forward(self, value):
        invocation = self.invocation
        self.invocation += 1
        residual, self.residual = self.residual, None
        if residual is None:
            raise RuntimeError("FFN called without its pre-norm residual")
        gamma = self.spec.layer_gammas.get(self.layer, self.spec.gamma)
        precisions = {p: self._precision(p) for p in ("gate", "up", "down")}
        need_reference = (
            self.observer is not None
            or gamma != 1
            or self.spec.direction != "real"
            or self.layer in self.spec.channel_repairs
        )
        if gamma == 0 and self.observer is None:
            return self.source(value)
        if all(v is None for v in precisions.values()) and self.observer is None:
            return self.source(value)
        reference = self._reference(value) if need_reference else None
        gate = self.bank.linear(self.layer, "gate", precisions["gate"], value)
        up = self.bank.linear(self.layer, "up", precisions["up"], value)
        product = self.source.act_fn(gate) * up
        if self.layer in self.spec.channel_repairs:
            indices = list(self.spec.channel_repairs[self.layer])
            product = product.clone()
            product[..., indices] = reference["product"][..., indices]
        candidate = {
            "gate": gate,
            "up": up,
            "product": product,
            "down_output": self.bank.linear(self.layer, "down", precisions["down"], product),
        }
        if reference is None:
            output = candidate["down_output"]
        else:
            delta = candidate["down_output"] - reference["down_output"]
            direction = (
                self.spec.direction
                if not self.spec.direction_layers or self.layer in self.spec.direction_layers
                else "real"
            )
            if direction != "real":
                digest = hashlib.sha256(
                    f"{self.spec.seed}:{self.sample_id}:{self.layer}:{invocation}:{tuple(delta.shape)}".encode()
                ).digest()
                delta = matching_noise(
                    delta, direction=direction, seed=int.from_bytes(digest[:8], "little") % (2**63 - 1)
                )
            if gamma == 0:
                output = reference["down_output"]
            elif gamma == 1 and direction == "real":
                output = candidate["down_output"]
            else:
                output = reference["down_output"] + gamma * delta
        finite(output)
        if self.observer is not None:
            self.observer(self.layer, residual, value, reference, candidate, output)
        return output


@contextlib.contextmanager
def intervene(layers, bank, spec: Intervention, *, observer: Callable | None = None, sample_id=""):
    originals, handles = {}, []
    try:
        for index, layer in enumerate(layers):
            original = layer.mlp
            if original is not bank.sources[index]:
                raise RuntimeError("nested interventions or changed source FFN")
            wrapper = ControlledFFN(bank, index, spec, observer, sample_id)
            originals[index] = original
            layer.mlp = wrapper

            def bind(_module, args, target=wrapper):
                if target.residual is not None:
                    raise RuntimeError("unbalanced FFN residual hooks")
                if not args or not isinstance(args[0], torch.Tensor):
                    raise TypeError("norm pre-hook has no residual tensor")
                target.residual = args[0]

            handles.append(layer.post_attention_layernorm.register_forward_pre_hook(bind))
        yield
    finally:
        for handle in handles:
            handle.remove()
        for index, source in originals.items():
            layers[index].mlp = source


@contextlib.contextmanager
def capture_path(layers):
    """One live teacher-forced pass; full token alignment, no subsampling."""
    rows, handles = {}, []

    def save(index, key, tensor):
        if isinstance(tensor, (tuple, list)):
            tensor = tensor[0]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"cannot capture {key}")
        row = rows.setdefault(index, {})
        if key in row:
            raise RuntimeError("path capture only supports one invocation per layer")
        row[key] = finite(tensor).detach().float().cpu().clone()

    try:
        for i, layer in enumerate(layers):
            handles.append(
                layer.post_attention_layernorm.register_forward_pre_hook(lambda _m, a, i=i: save(i, "residual", a[0]))
            )
            handles.append(
                layer.post_attention_layernorm.register_forward_hook(lambda _m, _a, o, i=i: save(i, "normalized", o))
            )
            handles.append(layer.mlp.register_forward_hook(lambda _m, _a, o, i=i: save(i, "ffn", o)))
            handles.append(layer.register_forward_hook(lambda _m, _a, o, i=i: save(i, "merged", o)))
        yield rows
    finally:
        for h in handles:
            h.remove()
