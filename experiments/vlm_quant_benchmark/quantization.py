from __future__ import annotations

import dataclasses
import math

import torch


@dataclasses.dataclass
class QuantizedTensor:
    dequantized: torch.Tensor
    scales: torch.Tensor
    zero_points: torch.Tensor | None
    qvalues: torch.Tensor
    bits: int
    group_size: int
    mse: float
    nmse: float
    sqnr_db: float
    max_abs_error: float


def _require_finite(value: torch.Tensor, name: str) -> None:
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be floating point")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or Inf")


def _stats(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float, float, float]:
    error = (candidate.float() - reference.float()).reshape(-1)
    ref = reference.float().reshape(-1)
    mse = error.square().mean().item() if error.numel() else 0.0
    power = ref.square().mean().item() if ref.numel() else 0.0
    nmse = mse / max(power, torch.finfo(torch.float32).tiny)
    sqnr = math.inf if mse == 0 else 10.0 * math.log10(max(power, 1e-30) / mse)
    max_error = error.abs().max().item() if error.numel() else 0.0
    return mse, nmse, sqnr, max_error


def quantize_weight_groupwise(
    weight: torch.Tensor,
    *,
    bits: int,
    group_size: int = 128,
    zero_point: bool = True,
    clip_ratio: float = 1.0,
) -> QuantizedTensor:
    """Per-output-row, input-channel-group fake quantization.

    AWQ/GPTQ comparisons use the same asymmetric quantizer by default.  A
    symmetric path is retained for Tender and compatibility experiments.
    """
    _require_finite(weight, "weight")
    if weight.ndim != 2:
        raise ValueError("weight must have [out_features, in_features] shape")
    if bits not in (2, 3, 4, 8):
        raise ValueError("only W2, W3, W4, and W8 are supported")
    if not (0 < clip_ratio <= 1):
        raise ValueError("clip_ratio must be in (0, 1]")

    original_shape = weight.shape
    rows, columns = original_shape
    groups = (columns + group_size - 1) // group_size
    padded_columns = groups * group_size
    w = weight.detach().float()
    if padded_columns != columns:
        w = torch.nn.functional.pad(w, (0, padded_columns - columns))
    grouped = w.reshape(rows, groups, group_size)

    if zero_point:
        qmin, qmax = 0, (1 << bits) - 1
        lower = grouped.amin(dim=-1, keepdim=True)
        upper = grouped.amax(dim=-1, keepdim=True)
        midpoint = (lower + upper) * 0.5
        half_range = (upper - lower) * 0.5 * clip_ratio
        lower, upper = midpoint - half_range, midpoint + half_range
        scales = (upper - lower) / float(qmax - qmin)
        scales = torch.where(scales > 0, scales, torch.ones_like(scales))
        zeros = torch.round(qmin - lower / scales).clamp(qmin, qmax)
        qvalues = torch.round(grouped / scales + zeros).clamp(qmin, qmax)
        restored = (qvalues - zeros) * scales
    else:
        qmax = (1 << (bits - 1)) - 1
        qmin = -qmax
        absolute = grouped.abs().amax(dim=-1, keepdim=True) * clip_ratio
        scales = absolute / float(qmax)
        scales = torch.where(scales > 0, scales, torch.ones_like(scales))
        zeros = None
        qvalues = torch.round(grouped / scales).clamp(qmin, qmax)
        restored = qvalues * scales

    restored = restored.reshape(rows, padded_columns)[:, :columns]
    qvalues = qvalues.reshape(rows, padded_columns)[:, :columns]
    mse, nmse, sqnr, max_error = _stats(weight, restored)
    return QuantizedTensor(
        dequantized=restored,
        scales=scales.squeeze(-1),
        zero_points=None if zeros is None else zeros.squeeze(-1),
        qvalues=qvalues,
        bits=bits,
        group_size=group_size,
        mse=mse,
        nmse=nmse,
        sqnr_db=sqnr,
        max_abs_error=max_error,
    )


def dynamic_activation_qdq(value: torch.Tensor, bits: int, *, per_token: bool = True) -> torch.Tensor:
    _require_finite(value, "activation")
    if bits not in (4, 8):
        raise ValueError("activation bits must be 4 or 8")
    qmax = (1 << (bits - 1)) - 1
    if per_token:
        maximum = value.float().abs().amax(dim=-1, keepdim=True)
    else:
        maximum = value.float().abs().amax().reshape([1] * value.ndim)
    scale = maximum / float(qmax)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    return torch.round(value.float() / scale).clamp(-qmax, qmax) * scale


def normalized_mse(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    return _stats(reference, candidate)[1]
