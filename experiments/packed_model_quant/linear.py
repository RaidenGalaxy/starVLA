"""Weight-only Linear backed by packed INT4 or INT8 buffers."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .packing import PackedWeight, dequantize_packed

COMPUTE_MODES = ("dequant_fp32", "a16_round_fp32_accum")


@dataclass
class RuntimeStats:
    calls: int = 0
    unpack_dequant_seconds: float = 0.0
    linear_seconds: float = 0.0
    max_temporary_bytes: int = 0


def _round_fp16(value: torch.Tensor, label: str) -> torch.Tensor:
    rounded_half = value.half()
    if not torch.isfinite(rounded_half).all():
        raise FloatingPointError(f"FP16 rounding overflowed for {label}")
    return rounded_half.float()


class PackedWeightOnlyLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bits: int,
        group_size: int,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        bias: torch.Tensor | None = None,
        compute_mode: str = "dequant_fp32",
    ) -> None:
        super().__init__()
        if compute_mode not in COMPUTE_MODES:
            raise ValueError(f"unsupported compute mode {compute_mode!r}")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.bits = int(bits)
        self.group_size = int(group_size)
        self.compute_mode = compute_mode
        self.register_buffer("qweight", qweight)
        self.register_buffer("scales", scales)
        self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False) if bias is not None else None
        self.runtime = RuntimeStats()

    @classmethod
    def from_packed(
        cls, packed: PackedWeight, *, bias: torch.Tensor | None = None, compute_mode: str = "dequant_fp32"
    ) -> "PackedWeightOnlyLinear":
        return cls(
            packed.in_features,
            packed.out_features,
            bits=packed.bits,
            group_size=packed.group_size,
            qweight=packed.qweight,
            scales=packed.scales,
            bias=bias,
            compute_mode=compute_mode,
        )

    def reset_runtime_stats(self) -> None:
        self.runtime = RuntimeStats()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        started = time.perf_counter()
        weight = dequantize_packed(
            self.qweight,
            self.scales,
            bits=self.bits,
            group_size=self.group_size,
            in_features=self.in_features,
        )
        self.runtime.unpack_dequant_seconds += time.perf_counter() - started
        self.runtime.max_temporary_bytes = max(
            self.runtime.max_temporary_bytes, weight.numel() * weight.element_size()
        )
        bias = self.bias
        if self.compute_mode == "a16_round_fp32_accum":
            value = _round_fp16(value.float(), "Linear input")
            weight = _round_fp16(weight, "dequantized weight")
            bias = _round_fp16(bias.float(), "Linear bias") if bias is not None else None
        elif value.dtype != torch.float32:
            value = value.float()
        started = time.perf_counter()
        output = F.linear(value, weight, bias)
        self.runtime.linear_seconds += time.perf_counter() - started
        self.runtime.calls += 1
        if self.compute_mode == "a16_round_fp32_accum":
            output = _round_fp16(output, "Linear output")
        return output

