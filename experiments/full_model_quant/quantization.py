"""FP32 group-wise RTN and exact reversible snapshots."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Iterable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class WeightStats:
    bits: int
    group_size: int
    groups: int
    mse: float
    normalized_mse: float
    sqnr_db: float
    max_abs_error: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@torch.no_grad()
def fake_quantize_weight(weight: torch.Tensor, *, bits: int, group_size: int) -> tuple[torch.Tensor, WeightStats]:
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("weight must be a floating 2-D tensor")
    if bits not in (4, 8) or group_size <= 0:
        raise ValueError("only positive-group W4/W8 quantization is supported")
    source = weight.detach().float()
    if not torch.isfinite(source).all():
        raise ValueError("weight contains NaN or Inf")
    qmax = (1 << (bits - 1)) - 1
    output = torch.empty_like(source)
    for start in range(0, source.shape[1], group_size):
        stop = min(start + group_size, source.shape[1])
        group = source[:, start:stop]
        maximum = group.abs().amax(dim=1, keepdim=True)
        scale = torch.where(maximum > 0, maximum / qmax, torch.ones_like(maximum))
        output[:, start:stop] = torch.round(group / scale).clamp(-qmax, qmax) * scale
    error = (output - source).reshape(-1)
    signal = source.reshape(-1)
    error_power = float(torch.dot(error, error).item())
    signal_power = float(torch.dot(signal, signal).item())
    mse = error_power / max(error.numel(), 1)
    nmse = error_power / max(signal_power, 1e-30)
    sqnr = math.inf if error_power == 0 else 10.0 * math.log10(max(signal_power, 1e-30) / error_power)
    return output, WeightStats(
        bits=bits,
        group_size=group_size,
        groups=source.shape[0] * math.ceil(source.shape[1] / group_size),
        mse=mse,
        normalized_mse=nmse,
        sqnr_db=sqnr,
        max_abs_error=float(error.abs().max().item()) if error.numel() else 0.0,
    )


@torch.no_grad()
def quantize_linear_(linear: nn.Linear, *, bits: int, group_size: int) -> WeightStats:
    value, stats = fake_quantize_weight(linear.weight, bits=bits, group_size=group_size)
    linear.weight.copy_(value.to(linear.weight.dtype))
    return stats


class ParameterSnapshot:
    def __init__(self, parameters: Iterable[tuple[str, nn.Parameter]]) -> None:
        unique: dict[int, tuple[str, nn.Parameter]] = {}
        for name, parameter in parameters:
            unique.setdefault(id(parameter), (name, parameter))
        self.parameters = list(unique.values())
        self.values = {name: parameter.detach().cpu().clone() for name, parameter in self.parameters}

    @torch.no_grad()
    def restore(self) -> None:
        for name, parameter in self.parameters:
            parameter.copy_(self.values[name].to(parameter.device, parameter.dtype))

    def assert_restored(self) -> None:
        for name, parameter in self.parameters:
            if not torch.equal(parameter.detach().cpu(), self.values[name]):
                raise RuntimeError(f"parameter restoration failed for {name}")


@torch.no_grad()
def fingerprint(parameters: Iterable[tuple[str, nn.Parameter]], excluded: set[int] | None = None) -> str:
    excluded = excluded or set()
    digest = hashlib.sha256()
    for name, parameter in parameters:
        if id(parameter) in excluded:
            continue
        value = parameter.detach()
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(f"{value.double().sum().item():.17g}".encode())
        digest.update(f"{value.double().abs().sum().item():.17g}".encode())
    return digest.hexdigest()
