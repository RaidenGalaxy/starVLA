"""Signed group-wise quantization and deterministic INT4 nibble packing."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class PackedWeightStats:
    bits: int
    group_size: int
    groups: int
    parameters: int
    qweight_bytes: int
    scale_bytes: int
    mse: float
    normalized_mse: float
    sqnr_db: float
    max_abs_error: float
    scale_rounding_mse: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class PackedWeight:
    qweight: torch.Tensor
    scales: torch.Tensor
    out_features: int
    in_features: int
    bits: int
    group_size: int
    stats: PackedWeightStats


def _scale_dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError("scale dtype must be fp16 or fp32")


def pack_int4(codes: torch.Tensor) -> torch.Tensor:
    """Pack signed [-7, 7] int8 values, low nibble first."""

    if codes.ndim != 2 or codes.dtype != torch.int8:
        raise ValueError("INT4 codes must be a two-dimensional int8 tensor")
    if codes.numel() and (int(codes.min()) < -7 or int(codes.max()) > 7):
        raise ValueError("INT4 codes must be in [-7, 7]")
    width = codes.shape[1]
    if width % 2:
        codes = torch.nn.functional.pad(codes, (0, 1))
    unsigned = torch.bitwise_and(codes.to(torch.int16), 0xF).to(torch.uint8)
    return unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)


def unpack_int4(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    if packed.ndim != 2 or packed.dtype != torch.uint8 or in_features <= 0:
        raise ValueError("packed INT4 must be a two-dimensional uint8 tensor with positive width")
    if packed.shape[1] != math.ceil(in_features / 2):
        raise ValueError("packed INT4 shape does not match in_features")
    low = torch.bitwise_and(packed, 0xF).to(torch.int8)
    high = torch.bitwise_and(packed >> 4, 0xF).to(torch.int8)
    values = torch.empty((packed.shape[0], packed.shape[1] * 2), dtype=torch.int8, device=packed.device)
    values[:, 0::2], values[:, 1::2] = low, high
    values = torch.where(values >= 8, values - 16, values)
    return values[:, :in_features].contiguous()


@torch.no_grad()
def quantize_packed_weight(
    weight: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    scale_dtype: str = "fp16",
) -> PackedWeight:
    if weight.ndim != 2 or not weight.is_floating_point():
        raise ValueError("weight must be a floating two-dimensional tensor")
    if bits not in (4, 8) or group_size <= 0:
        raise ValueError("only positive-group W4/W8 quantization is supported")
    source = weight.detach().float().cpu()
    if not torch.isfinite(source).all():
        raise ValueError("weight contains NaN or Inf")
    out_features, in_features = source.shape
    group_count = math.ceil(in_features / group_size)
    qmax = 7 if bits == 4 else 127
    codes = torch.empty_like(source, dtype=torch.int8)
    scales_fp32 = torch.empty((out_features, group_count), dtype=torch.float32)
    for index, start in enumerate(range(0, in_features, group_size)):
        stop = min(start + group_size, in_features)
        group = source[:, start:stop]
        maximum = group.abs().amax(dim=1)
        scale = torch.where(maximum > 0, maximum / qmax, torch.ones_like(maximum))
        scales_fp32[:, index] = scale
        codes[:, start:stop] = torch.round(group / scale[:, None]).clamp(-qmax, qmax).to(torch.int8)
    stored_scales = scales_fp32.to(_scale_dtype(scale_dtype))
    restored = dequantize_codes(codes, stored_scales, group_size=group_size)
    error = restored - source
    error_power = float(error.double().pow(2).sum().item())
    signal_power = float(source.double().pow(2).sum().item())
    scale_error = stored_scales.float() - scales_fp32
    qweight = pack_int4(codes) if bits == 4 else codes
    stats = PackedWeightStats(
        bits=bits,
        group_size=group_size,
        groups=out_features * group_count,
        parameters=source.numel(),
        qweight_bytes=qweight.numel() * qweight.element_size(),
        scale_bytes=stored_scales.numel() * stored_scales.element_size(),
        mse=error_power / max(source.numel(), 1),
        normalized_mse=error_power / max(signal_power, 1e-30),
        sqnr_db=math.inf if error_power == 0 else 10.0 * math.log10(max(signal_power, 1e-30) / error_power),
        max_abs_error=float(error.abs().max().item()) if error.numel() else 0.0,
        scale_rounding_mse=float(scale_error.double().pow(2).mean().item()) if scale_error.numel() else 0.0,
    )
    return PackedWeight(qweight, stored_scales, out_features, in_features, bits, group_size, stats)


def dequantize_codes(codes: torch.Tensor, scales: torch.Tensor, *, group_size: int) -> torch.Tensor:
    if codes.ndim != 2 or codes.dtype != torch.int8 or scales.ndim != 2 or group_size <= 0:
        raise ValueError("invalid integer codes, scales, or group size")
    if codes.shape[0] != scales.shape[0] or scales.shape[1] != math.ceil(codes.shape[1] / group_size):
        raise ValueError("scale shape does not match integer codes")
    output = torch.empty(codes.shape, dtype=torch.float32, device=codes.device)
    scale_values = scales.float().to(codes.device)
    for index, start in enumerate(range(0, codes.shape[1], group_size)):
        stop = min(start + group_size, codes.shape[1])
        output[:, start:stop] = codes[:, start:stop].float() * scale_values[:, index, None]
    return output


def dequantize_packed(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    in_features: int,
) -> torch.Tensor:
    if bits == 4:
        codes = unpack_int4(qweight, in_features)
    elif bits == 8 and qweight.dtype == torch.int8 and qweight.shape[1] == in_features:
        codes = qweight
    else:
        raise ValueError("invalid packed weight for requested bit width")
    return dequantize_codes(codes, scales, group_size=group_size)

