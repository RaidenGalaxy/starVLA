from __future__ import annotations

import contextlib
import dataclasses
import math
from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import nn

from .config import QuantScheme
from .inventory import LinearTarget


@dataclasses.dataclass(frozen=True)
class QuantizedWeight:
    dequantized: torch.Tensor
    mse: float
    nmse: float
    max_abs_error: float
    groups: int


def quantize_weight(
    weight: torch.Tensor, *, bits: int, group_size: int, zero_point: bool,
) -> QuantizedWeight:
    if weight.ndim != 2 or not torch.is_floating_point(weight):
        raise TypeError("weight must be a floating [out_features, in_features] tensor")
    if bits not in (2, 3, 4, 8):
        raise ValueError("weight bits must be 2, 3, 4, or 8")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if not torch.isfinite(weight).all():
        raise ValueError("weight contains NaN or Inf")
    rows, columns = weight.shape
    groups = (columns + group_size - 1) // group_size
    value = weight.detach().float()
    # Quantize the tail at its true width. Padding with zeros would change the
    # min/max of an all-positive or all-negative final group and therefore no
    # longer be the stated group-wise quantizer.
    restored_groups = []
    for start in range(0, columns, group_size):
        grouped = value[:, start:min(start + group_size, columns)]
        if zero_point:
            qmin, qmax = 0, (1 << bits) - 1
            lower = grouped.amin(dim=-1, keepdim=True)
            upper = grouped.amax(dim=-1, keepdim=True)
            constant = upper == lower
            scale = (upper - lower) / float(qmax - qmin)
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            zeros = torch.round(qmin - lower / scale).clamp(qmin, qmax)
            restored = (
                torch.round(grouped / scale + zeros).clamp(qmin, qmax) - zeros
            ) * scale
            restored = torch.where(constant, grouped, restored)
        else:
            qmax = (1 << (bits - 1)) - 1
            maximum = grouped.abs().amax(dim=-1, keepdim=True)
            scale = maximum / float(qmax)
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            restored = torch.round(grouped / scale).clamp(-qmax, qmax) * scale
        restored_groups.append(restored)
    restored = torch.cat(restored_groups, dim=1)
    error = restored - weight.float()
    mse = float(error.square().mean()) if error.numel() else 0.0
    power = float(weight.float().square().mean()) if weight.numel() else 0.0
    return QuantizedWeight(
        restored, mse, mse / max(power, torch.finfo(torch.float32).tiny),
        float(error.abs().max()) if error.numel() else 0.0, groups,
    )


def activation_qdq(value: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(value).all():
        raise ValueError("activation contains NaN or Inf")
    maximum = value.float().abs().amax(dim=-1, keepdim=True)
    scale = torch.where(maximum > 0, maximum / 127.0, torch.ones_like(maximum))
    return torch.round(value.float() / scale).clamp(-127, 127) * scale


class FakeQuantLinear(nn.Module):
    def __init__(
        self, source: nn.Linear, weight: torch.Tensor, *,
        activation_bits: int | None, input_scale: torch.Tensor | None,
    ) -> None:
        super().__init__()
        self.in_features, self.out_features = source.in_features, source.out_features
        self.activation_bits = activation_bits
        self.register_buffer("weight", weight.detach().float().clone())
        self.register_buffer(
            "bias", None if source.bias is None else source.bias.detach().float().clone(),
        )
        self.register_buffer(
            "input_scale", None if input_scale is None else input_scale.detach().float().clone(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        compute = value.float()
        if self.input_scale is not None:
            compute = compute / self.input_scale
        if self.activation_bits == 8:
            compute = activation_qdq(compute)
        return F.linear(compute, self.weight, self.bias)


def _parent_and_key(model: nn.Module, path: str) -> tuple[nn.Module, str]:
    if "." not in path:
        return model, path
    parent_path, key = path.rsplit(".", 1)
    return model.get_submodule(parent_path), key


@contextlib.contextmanager
def quantized_linears(
    model: nn.Module,
    targets: list[LinearTarget],
    scheme: QuantScheme,
    *,
    scales: dict[str, torch.Tensor] | None = None,
) -> Iterator[dict]:
    if scheme.weight_bits is None:
        raise ValueError("quantized_linears requires a low-bit scheme")
    expected = {target.name for target in targets}
    if scales is not None and set(scales) != expected:
        missing, extra = expected - set(scales), set(scales) - expected
        raise ValueError(f"scale names do not match targets; missing={len(missing)} extra={len(extra)}")
    originals: dict[str, nn.Linear] = {}
    audit = {}
    try:
        for target in targets:
            source = target.module
            scale = None if scales is None else scales[target.name].float()
            if scale is not None:
                if scale.ndim != 1 or scale.numel() != source.in_features:
                    raise ValueError(f"invalid input scale at {target.name}")
                if not torch.isfinite(scale).all() or (scale <= 0).any():
                    raise ValueError(f"non-positive/non-finite input scale at {target.name}")
                weight = source.weight.detach().float() * scale.unsqueeze(0)
            else:
                weight = source.weight.detach().float()
            group_size = source.in_features if scheme.per_output_channel else scheme.group_size
            quantized = quantize_weight(
                weight, bits=scheme.weight_bits, group_size=group_size,
                zero_point=scheme.zero_point,
            )
            replacement = FakeQuantLinear(
                source, quantized.dequantized, activation_bits=scheme.activation_bits,
                input_scale=scale,
            )
            parent, key = _parent_and_key(model, target.name)
            originals[target.name] = source
            setattr(parent, key, replacement)
            audit[target.name] = {
                "scope": target.scope, "layer": target.layer,
                "weight_mse": quantized.mse, "weight_nmse": quantized.nmse,
                "max_abs_error": quantized.max_abs_error,
                "group_size": group_size, "groups_per_row": quantized.groups,
                "activation_bits": scheme.activation_bits,
                "input_scale": scale is not None,
            }
        yield {
            "scheme": scheme.to_dict(), "module_count": len(targets),
            "modules": audit, "fake_quantized": True,
            "deployment_kernel_claim": False,
        }
    finally:
        for name, source in originals.items():
            parent, key = _parent_and_key(model, name)
            setattr(parent, key, source)


@contextlib.contextmanager
def fp16_rounded_parameters(model: nn.Module) -> Iterator[dict]:
    originals: list[tuple[nn.Parameter, torch.Tensor]] = []
    nonfinite = []
    try:
        for name, parameter in model.named_parameters():
            if not parameter.is_floating_point():
                continue
            rounded = parameter.detach().to(torch.float16).to(torch.float32)
            if not torch.isfinite(rounded).all():
                nonfinite.append(name)
            originals.append((parameter, parameter.data))
            parameter.data = rounded
        if nonfinite:
            raise ValueError(f"FP16 rounding produced non-finite parameters: {nonfinite[:8]}")
        yield {
            "definition": "official BF16 checkpoint values rounded through FP16 then widened to FP32",
            "parameter_tensors": len(originals), "compute_dtype": "torch.float32",
        }
    finally:
        for parameter, original in originals:
            parameter.data = original


def logical_bits_per_weight(scheme: QuantScheme, in_features: int) -> float:
    if scheme.weight_bits is None:
        return 16.0
    group = in_features if scheme.per_output_channel else scheme.group_size
    metadata = 16 + (scheme.weight_bits if scheme.zero_point else 0)
    return scheme.weight_bits + metadata / group
