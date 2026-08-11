"""Streaming-dequant and AVX-512 W4A8 packed Linear implementations."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from experiments.packed_model_quant.linear import PackedWeightOnlyLinear
from experiments.packed_model_quant.packing import dequantize_codes, unpack_int4

from . import BACKENDS
from .extension import load_native_extension, require_avx512_vnni


@dataclass
class NativeRuntimeStats:
    calls: int = 0
    unpack_seconds: float = 0.0
    dequant_seconds: float = 0.0
    unpack_dequant_seconds: float = 0.0
    linear_seconds: float = 0.0
    activation_quant_seconds: float = 0.0
    native_gemm_seconds: float = 0.0
    max_temporary_bytes: int = 0
    max_float_weight_tile_bytes: int = 0
    full_weight_bytes: int = 0
    tiles: int = 0
    integer_macs: int = 0
    fallback_calls: int = 0
    activation_error_power: float = 0.0
    activation_signal_power: float = 0.0
    activation_values: int = 0
    activation_max_abs_error: float = 0.0
    activation_metrics_seconds: float = 0.0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _round_fp16(value: torch.Tensor, label: str) -> torch.Tensor:
    rounded = value.half()
    if not torch.isfinite(rounded).all():
        raise FloatingPointError(f"FP16 rounding overflowed for {label}")
    return rounded.float()


@torch.no_grad()
def quantize_activation_reference(value: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    if value.ndim != 2 or value.dtype != torch.float32 or group_size <= 0:
        raise ValueError("reference activation quantization expects a 2D FP32 tensor")
    if not torch.isfinite(value).all():
        raise ValueError("activation contains NaN or Inf")
    rows, width = value.shape
    groups = math.ceil(width / group_size)
    codes = torch.empty((rows, width), dtype=torch.int8)
    scales = torch.empty((rows, groups), dtype=torch.float32)
    for group, start in enumerate(range(0, width, group_size)):
        stop = min(start + group_size, width)
        block = value[:, start:stop]
        maximum = block.abs().amax(dim=1)
        scale = torch.where(maximum > 0, maximum / 127.0, torch.ones_like(maximum))
        scales[:, group] = scale
        codes[:, start:stop] = torch.round(block / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return codes, scales


@torch.no_grad()
def w4a8_reference(
    qactivation: torch.Tensor,
    activation_scales: torch.Tensor,
    qweight: torch.Tensor,
    weight_scales: torch.Tensor,
    *,
    in_features: int,
    group_size: int = 128,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    codes = unpack_int4(qweight, in_features).to(torch.int32)
    output = torch.zeros((qactivation.shape[0], qweight.shape[0]), dtype=torch.float32)
    for group, start in enumerate(range(0, in_features, group_size)):
        stop = min(start + group_size, in_features)
        dot = qactivation[:, start:stop].to(torch.int32) @ codes[:, start:stop].T
        output += dot.float() * activation_scales[:, group, None] * weight_scales[:, group].float()[None, :]
    if bias is not None:
        output += bias.float()
    return output


class NativePackedLinear(PackedWeightOnlyLinear):
    """Packed Linear that never caches or exposes a floating target weight."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bits: int,
        group_size: int,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        bias: torch.Tensor | None,
        backend: str,
        output_tile_size: int = 256,
    ) -> None:
        if backend not in BACKENDS:
            raise ValueError(f"unsupported native backend {backend!r}")
        if output_tile_size <= 0:
            raise ValueError("output tile size must be positive")
        if backend == "avx512_w4a8" and (bits != 4 or group_size != 128):
            raise ValueError("avx512_w4a8 requires W4 group-128 weights")
        super().__init__(
            in_features,
            out_features,
            bits=bits,
            group_size=group_size,
            qweight=qweight,
            scales=scales,
            bias=bias,
            compute_mode="dequant_fp32",
        )
        self.backend = backend
        self.output_tile_size = int(output_tile_size)
        self.runtime = NativeRuntimeStats(full_weight_bytes=self.out_features * self.in_features * 4)

    def reset_runtime_stats(self) -> None:
        self.runtime = NativeRuntimeStats(full_weight_bytes=self.out_features * self.in_features * 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != self.in_features:
            raise ValueError(f"expected input width {self.in_features}, found {value.shape[-1]}")
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise FloatingPointError("Linear input must contain finite floating-point values")
        original_shape = value.shape[:-1]
        flattened = value.float().reshape(-1, self.in_features).contiguous()
        if self.backend == "avx512_w4a8":
            output = self._forward_w4a8(flattened)
        else:
            output = self._forward_streaming(flattened)
        self.runtime.calls += 1
        return output.reshape(*original_shape, self.out_features)

    def _forward_streaming(self, value: torch.Tensor) -> torch.Tensor:
        simulate_a16 = self.backend == "streaming_a16_sim"
        if simulate_a16:
            value = _round_fp16(value, "Linear input")
        output = torch.empty((value.shape[0], self.out_features), dtype=torch.float32, device=value.device)
        tile_size = min(self.output_tile_size, max(1, self.out_features - 1))
        for start in range(0, self.out_features, tile_size):
            stop = min(start + tile_size, self.out_features)
            started = time.perf_counter()
            if self.bits == 4:
                codes = unpack_int4(self.qweight[start:stop], self.in_features)
            else:
                codes = self.qweight[start:stop]
            unpack_elapsed = time.perf_counter() - started
            started = time.perf_counter()
            weight = dequantize_codes(codes, self.scales[start:stop], group_size=self.group_size)
            dequant_elapsed = time.perf_counter() - started
            bias = self.bias[start:stop].float() if self.bias is not None else None
            if simulate_a16:
                weight = _round_fp16(weight, "dequantized weight tile")
                bias = _round_fp16(bias, "Linear bias tile") if bias is not None else None
            started = time.perf_counter()
            output[:, start:stop] = F.linear(value, weight, bias)
            linear_elapsed = time.perf_counter() - started
            float_bytes = weight.numel() * weight.element_size()
            temporary_bytes = float_bytes + codes.numel() * codes.element_size()
            self.runtime.unpack_seconds += unpack_elapsed
            self.runtime.dequant_seconds += dequant_elapsed
            self.runtime.unpack_dequant_seconds += unpack_elapsed + dequant_elapsed
            self.runtime.linear_seconds += linear_elapsed
            self.runtime.max_float_weight_tile_bytes = max(self.runtime.max_float_weight_tile_bytes, float_bytes)
            self.runtime.max_temporary_bytes = max(self.runtime.max_temporary_bytes, temporary_bytes)
            self.runtime.tiles += 1
            del codes, weight
        if simulate_a16:
            output = _round_fp16(output, "Linear output")
        return output

    def _forward_w4a8(self, value: torch.Tensor) -> torch.Tensor:
        extension = load_native_extension()
        require_avx512_vnni(extension)
        started = time.perf_counter()
        qactivation, activation_scales = extension.quantize_activation_groupwise(value, self.group_size)
        self.runtime.activation_quant_seconds += time.perf_counter() - started
        started = time.perf_counter()
        restored = torch.empty_like(value)
        for group, offset in enumerate(range(0, self.in_features, self.group_size)):
            stop = min(offset + self.group_size, self.in_features)
            restored[:, offset:stop] = qactivation[:, offset:stop].float() * activation_scales[:, group, None]
        error = restored - value
        self.runtime.activation_error_power += float(error.double().pow(2).sum().item())
        self.runtime.activation_signal_power += float(value.double().pow(2).sum().item())
        self.runtime.activation_values += value.numel()
        self.runtime.activation_max_abs_error = max(
            self.runtime.activation_max_abs_error, float(error.abs().max().item()) if error.numel() else 0.0
        )
        self.runtime.activation_metrics_seconds += time.perf_counter() - started
        started = time.perf_counter()
        output = extension.packed_w4a8_linear_vnni(
            qactivation,
            activation_scales,
            self.qweight,
            self.scales,
            self.bias,
            self.in_features,
        )
        self.runtime.native_gemm_seconds += time.perf_counter() - started
        activation_bytes = qactivation.numel() + activation_scales.numel() * 4 + restored.numel() * 4
        self.runtime.max_temporary_bytes = max(self.runtime.max_temporary_bytes, activation_bytes)
        self.runtime.integer_macs += value.shape[0] * self.out_features * self.in_features
        return output


def make_linear_factory(backend: str, output_tile_size: int = 256) -> Callable:
    def factory(row, qweight, scales, bias):
        return NativePackedLinear(
            row["shape"][1],
            row["shape"][0],
            bits=row["bits"],
            group_size=row["group_size"],
            qweight=qweight,
            scales=scales,
            bias=bias,
            backend=backend,
            output_tile_size=output_tile_size,
        )

    return factory
