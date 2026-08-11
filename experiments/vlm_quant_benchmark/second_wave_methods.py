"""CPU research implementations for the second method wave.

These implementations preserve each paper's defining mechanism, but are not
packed deployment kernels: OmniQuant uses layer-local LET/LWC optimization,
SpQR stores sparse high-precision corrections, and SqueezeLLM uses a weighted
non-uniform codebook plus sparse correction.
"""
from __future__ import annotations

import time
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from . import methods as base
from .calibration import ActivationStats
from .config import ExperimentConfig, MethodName
from .quantization import normalized_mse, quantize_weight_groupwise
from .selection import SelectedLinear


class SparseCorrectionLinear(base.FrozenLinear):
    def __init__(self, source: nn.Linear, dense_weight: torch.Tensor,
                 sparse_indices: torch.Tensor, sparse_values: torch.Tensor) -> None:
        super().__init__(source, dense_weight)
        self.register_buffer("sparse_indices", sparse_indices.to(torch.long))
        self.register_buffer("sparse_values", sparse_values.float())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        x = value.float()
        dense = F.linear(x, self.weight, self.bias)
        if self.sparse_values.numel() == 0:
            return dense
        sparse = torch.sparse_coo_tensor(
            self.sparse_indices, self.sparse_values,
            (self.out_features, self.in_features), device=x.device,
        ).coalesce()
        flattened = x.reshape(-1, self.in_features)
        correction = torch.sparse.mm(sparse, flattened.T).T
        return dense + correction.reshape(*x.shape[:-1], self.out_features)


def _ste_qdq(weight: torch.Tensor, bits: int, group_size: int,
             zero_point: bool, clip_ratio: torch.Tensor) -> torch.Tensor:
    rows, columns = weight.shape
    groups = (columns + group_size - 1) // group_size
    padded = groups * group_size
    value = F.pad(weight, (0, padded - columns)).reshape(rows, groups, group_size)
    if zero_point:
        qmin, qmax = 0, (1 << bits) - 1
        lower, upper = value.amin(-1, keepdim=True), value.amax(-1, keepdim=True)
        midpoint, half = (lower + upper) / 2, (upper - lower) / 2 * clip_ratio
        lower, upper = midpoint - half, midpoint + half
        scale = ((upper - lower) / (qmax - qmin)).clamp_min(1e-8)
        zero = torch.round(qmin - lower / scale).clamp(qmin, qmax)
        dequant = (torch.round(value / scale + zero).clamp(qmin, qmax) - zero) * scale
    else:
        qmax = (1 << (bits - 1)) - 1
        scale = (value.abs().amax(-1, keepdim=True) * clip_ratio / qmax).clamp_min(1e-8)
        dequant = torch.round(value / scale).clamp(-qmax, qmax) * scale
    dequant = dequant.reshape(rows, padded)[:, :columns]
    return weight + (dequant - weight).detach()


def omniquant_linear(source: nn.Linear, stats: ActivationStats, config: ExperimentConfig):
    weight = source.weight.detach().float()
    sample = stats.reservoir.float()
    reference = F.linear(sample, weight, None)
    activation = stats.abs_max.clamp_min(1e-6)
    weight_max = weight.abs().amax(0).clamp_min(1e-6)
    initial = (activation.pow(0.5) / weight_max.pow(0.5)).clamp_min(1e-6)
    log_scale = nn.Parameter(initial.log())
    clip_logit = nn.Parameter(torch.tensor(5.0))
    optimizer = torch.optim.Adam((log_scale, clip_logit), lr=2e-2)
    curve = []
    for _ in range(20):
        scale = log_scale.exp().clamp(1e-4, 1e4)
        clip = 0.5 + 0.5 * torch.sigmoid(clip_logit)
        candidate_weight = _ste_qdq(
            weight * scale.unsqueeze(0), config.weight_bits,
            config.group_size, config.zero_point, clip,
        )
        candidate = F.linear(sample / scale, candidate_weight, None)
        loss = (candidate - reference).square().mean() / reference.square().mean().clamp_min(1e-8)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        curve.append(float(loss.detach()))
    scale = log_scale.detach().exp().clamp(1e-4, 1e4)
    clip = float((0.5 + 0.5 * torch.sigmoid(clip_logit.detach())).item())
    quantized = quantize_weight_groupwise(
        weight * scale.unsqueeze(0), bits=config.weight_bits,
        group_size=config.group_size, zero_point=config.zero_point,
        clip_ratio=clip,
    )
    wrapper = base.ScaledQuantLinear(source, quantized.dequantized, scale)
    return wrapper, {"algorithm": "layer_local_LET_LWC", "steps": 20,
                     "clip_ratio": clip, "loss_curve": curve,
                     "calibration_nmse": normalized_mse(reference, wrapper(sample))}


def _sparse_residual(source: nn.Linear, dense_weight: torch.Tensor,
                     stats: ActivationStats, fraction: float):
    residual = source.weight.detach().float() - dense_weight
    saliency = residual.abs() * stats.abs_mean.clamp_min(1e-8).sqrt().unsqueeze(0)
    count = max(1, int(saliency.numel() * fraction))
    flat_indices = torch.topk(saliency.reshape(-1), count, sorted=False).indices
    rows = torch.div(flat_indices, source.in_features, rounding_mode="floor")
    columns = flat_indices.remainder(source.in_features)
    indices = torch.stack((rows, columns))
    values = residual[rows, columns]
    return indices, values


def spqr_linear(source: nn.Linear, stats: ActivationStats, config: ExperimentConfig):
    quantized = quantize_weight_groupwise(
        source.weight, bits=config.weight_bits, group_size=config.group_size,
        zero_point=config.zero_point)
    indices, values = _sparse_residual(source, quantized.dequantized, stats, 0.01)
    wrapper = SparseCorrectionLinear(source, quantized.dequantized, indices, values)
    index_bits = max(1, (source.weight.numel() - 1).bit_length())
    bpw = config.weight_bits + values.numel() * (16 + index_bits) / source.weight.numel()
    return wrapper, {"algorithm": "sparse_quantized_representation",
                     "outlier_fraction": values.numel() / source.weight.numel(),
                     "estimated_bits_per_weight": bpw}


def _weighted_codebook(weight: torch.Tensor, sensitivity: torch.Tensor, levels: int) -> torch.Tensor:
    flat = weight.reshape(-1)
    weights = sensitivity.unsqueeze(0).expand(weight.shape[0], -1).reshape(-1).clamp_min(1e-8)
    centers = torch.quantile(flat, torch.linspace(0, 1, levels, device=flat.device))
    for _ in range(5):
        assignment = (flat[:, None] - centers[None, :]).abs().argmin(1)
        updated = centers.clone()
        for index in range(levels):
            mask = assignment == index
            if mask.any():
                updated[index] = (flat[mask] * weights[mask]).sum() / weights[mask].sum()
        centers = updated
    return centers[assignment].reshape_as(weight)


def squeezellm_linear(source: nn.Linear, stats: ActivationStats, config: ExperimentConfig):
    weight = source.weight.detach().float()
    restored = torch.empty_like(weight)
    levels = 1 << config.weight_bits
    sensitivity = stats.abs_mean.square().clamp_min(1e-8)
    for start in range(0, source.in_features, config.group_size):
        stop = min(start + config.group_size, source.in_features)
        restored[:, start:stop] = _weighted_codebook(
            weight[:, start:stop], sensitivity[start:stop], levels)
    indices, values = _sparse_residual(source, restored, stats, 0.0045)
    wrapper = SparseCorrectionLinear(source, restored, indices, values)
    return wrapper, {"algorithm": "sensitivity_weighted_nonuniform_dense_sparse",
                     "centroids": levels,
                     "outlier_fraction": values.numel() / source.weight.numel()}


def apply_second_wave(model: nn.Module, selected: list[SelectedLinear], config: ExperimentConfig,
                      calibration: Mapping[str, ActivationStats] | None):
    if calibration is None:
        raise ValueError(f"{config.method.value} requires calibration statistics")
    factories = {
        MethodName.OMNIQUANT: omniquant_linear,
        MethodName.SPQR: spqr_linear,
        MethodName.SQUEEZELLM: squeezellm_linear,
    }
    factory = factories[config.method]
    originals, audits, started = {}, {}, time.perf_counter()
    try:
        for item in selected:
            replacement, audit = factory(item.module, calibration[item.name], config)
            parent, key = base._parent_and_key(model, item.name)
            originals[item.name] = item.module
            setattr(parent, key, replacement)
            audits[item.name] = audit | {"subsystem": item.subsystem}
    except Exception:
        for name, original in originals.items():
            parent, key = base._parent_and_key(model, name); setattr(parent, key, original)
        raise
    return base.AppliedQuantization(model, originals, {
        "method": config.method.value, "scope": config.scope.value,
        "module_count": len(originals), "elapsed_seconds": time.perf_counter() - started,
        "implementation_class": "cpu_research_not_deployment_kernel", "modules": audits})
