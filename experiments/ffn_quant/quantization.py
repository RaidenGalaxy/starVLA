"""Group-wise fake quantization and reversible parameter snapshots."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Iterable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class QuantizationStats:
    bits: int
    group_size: int
    num_groups: int
    weight_mse: float
    weight_sqnr_db: float
    max_abs_error: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _validate_weight(weight: torch.Tensor, bits: int, group_size: int) -> None:
    if not weight.is_floating_point():
        raise TypeError("fake quantization requires a floating-point tensor")
    if weight.ndim != 2:
        raise ValueError(f"expected a 2-D Linear weight, got shape {tuple(weight.shape)}")
    if bits not in (4, 8):
        raise ValueError(f"only W4 and W8 are supported, got W{bits}")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if not torch.isfinite(weight).all():
        raise ValueError("weight contains NaN or Inf")


@torch.no_grad()
def fake_quantize_weight(
    weight: torch.Tensor,
    *,
    bits: int,
    group_size: int = 128,
) -> tuple[torch.Tensor, QuantizationStats]:
    """Symmetric per-row, per-input-group quantize/dequantize.

    W4 uses [-7, 7] and W8 uses [-127, 127]. A zero group maps exactly to
    zero. The returned tensor has the same shape, dtype and device as input.
    """

    _validate_weight(weight, bits, group_size)
    qmax = (1 << (bits - 1)) - 1
    dequantized = torch.empty_like(weight)
    num_groups = math.ceil(weight.shape[1] / group_size)

    for start in range(0, weight.shape[1], group_size):
        stop = min(start + group_size, weight.shape[1])
        group = weight[:, start:stop]
        max_abs = group.abs().amax(dim=1, keepdim=True)
        scale = torch.where(max_abs > 0, max_abs / qmax, torch.ones_like(max_abs))
        quantized = torch.round(group / scale).clamp(-qmax, qmax)
        dequantized[:, start:stop] = quantized * scale

    error = (dequantized.float() - weight.float()).reshape(-1)
    reference = weight.float().reshape(-1)
    error_power = float(torch.dot(error, error).item())
    signal_power = float(torch.dot(reference, reference).item())
    mse = error_power / max(error.numel(), 1)
    if error_power == 0:
        sqnr = math.inf
    elif signal_power == 0:
        sqnr = -math.inf
    else:
        sqnr = 10.0 * math.log10(signal_power / error_power)
    stats = QuantizationStats(
        bits=bits,
        group_size=group_size,
        num_groups=weight.shape[0] * num_groups,
        weight_mse=mse,
        weight_sqnr_db=sqnr,
        max_abs_error=float(error.abs().max().item()) if error.numel() else 0.0,
    )
    return dequantized, stats


@torch.no_grad()
def quantize_linear_(linear: nn.Linear, *, bits: int, group_size: int) -> QuantizationStats:
    quantized, stats = fake_quantize_weight(linear.weight, bits=bits, group_size=group_size)
    linear.weight.copy_(quantized)
    return stats


class ParameterSnapshot:
    """Exact CPU snapshot of a bounded set of parameters."""

    def __init__(self, parameters: Iterable[tuple[str, nn.Parameter]]) -> None:
        unique: dict[int, tuple[str, nn.Parameter]] = {}
        for name, parameter in parameters:
            unique.setdefault(id(parameter), (name, parameter))
        self._parameters = list(unique.values())
        self._values = {name: parameter.detach().cpu().clone() for name, parameter in self._parameters}

    @torch.no_grad()
    def restore(self) -> None:
        for name, parameter in self._parameters:
            parameter.copy_(self._values[name].to(device=parameter.device, dtype=parameter.dtype))

    def assert_restored(self) -> None:
        for name, parameter in self._parameters:
            current = parameter.detach().cpu()
            if not torch.equal(current, self._values[name]):
                raise RuntimeError(f"parameter restoration failed for {name}")


@torch.no_grad()
def parameter_fingerprint(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
    *,
    excluded_ids: set[int] | None = None,
) -> str:
    """Stable mutation detector without materializing a second model copy."""

    excluded_ids = excluded_ids or set()
    digest = hashlib.sha256()
    for name, parameter in named_parameters:
        if id(parameter) in excluded_ids:
            continue
        value = parameter.detach()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        # Two FP64 reductions make accidental collisions unlikely while keeping
        # the check practical for a multi-billion parameter CPU model.
        digest.update(f"{value.double().sum().item():.17g}".encode("ascii"))
        digest.update(f"{value.double().abs().sum().item():.17g}".encode("ascii"))
    return digest.hexdigest()
