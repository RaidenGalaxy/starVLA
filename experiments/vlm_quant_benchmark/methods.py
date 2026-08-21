from __future__ import annotations

import dataclasses
import time
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .calibration import ActivationStats
from .config import ExperimentConfig, MethodName
from .quantization import dynamic_activation_qdq, normalized_mse, quantize_weight_groupwise
from .selection import SelectedLinear


def _parent_and_key(model: nn.Module, path: str) -> tuple[nn.Module, str]:
    parent: nn.Module = model
    parts = path.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


class FrozenLinear(nn.Module):
    def __init__(self, source: nn.Linear, weight: torch.Tensor) -> None:
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.register_buffer("weight", weight.detach().float().contiguous())
        if source.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", source.bias.detach().float().contiguous())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value.float(), self.weight, self.bias)


class ScaledQuantLinear(FrozenLinear):
    """Equivalent input/weight scaling followed by activation-aware weight QDQ."""

    def __init__(self, source: nn.Linear, weight: torch.Tensor, input_scale: torch.Tensor) -> None:
        super().__init__(source, weight)
        self.register_buffer("input_scale", input_scale.detach().float().contiguous())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.linear(value.float() / self.input_scale, self.weight, self.bias)


class W8A8Linear(FrozenLinear):
    def __init__(self, source: nn.Linear, weight: torch.Tensor,
                 input_scale: torch.Tensor | None = None,
                 activation_bits: int = 8) -> None:
        super().__init__(source, weight)
        self.register_buffer("input_scale", input_scale)
        self.activation_bits = activation_bits

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.float()
        if self.input_scale is not None:
            value = value / self.input_scale
        value = dynamic_activation_qdq(value, self.activation_bits, per_token=True)
        return F.linear(value, self.weight, self.bias)


class LLMInt8Linear(FrozenLinear):
    """Mixed-precision decomposition with a dynamic activation-outlier branch."""

    def __init__(self, source: nn.Linear, quant_weight: torch.Tensor, threshold: float) -> None:
        super().__init__(source, quant_weight)
        self.register_buffer("original_weight", source.weight.detach().float().contiguous())
        self.threshold = float(threshold)
        self.outlier_values = 0
        self.total_values = 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        x = value.float()
        outlier = x.abs() > self.threshold
        self.outlier_values += int(outlier.sum().item())
        self.total_values += outlier.numel()
        regular = x.masked_fill(outlier, 0)
        high_precision = x.masked_fill(~outlier, 0)
        regular = dynamic_activation_qdq(regular, 8, per_token=True)
        return F.linear(regular, self.weight, self.bias) + F.linear(high_precision, self.original_weight, None)


class TenderLinear(FrozenLinear):
    """Software simulation of channel decomposition with power-of-two scales."""

    def __init__(
        self,
        source: nn.Linear,
        quant_weight: torch.Tensor,
        channel_group: torch.Tensor,
        activation_bits: int,
    ) -> None:
        super().__init__(source, quant_weight)
        self.register_buffer("channel_group", channel_group.to(torch.long))
        self.activation_bits = activation_bits
        self.group_calls = torch.zeros(int(channel_group.max().item()) + 1, dtype=torch.long)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        x = value.float()
        output = torch.zeros(*x.shape[:-1], self.out_features, dtype=torch.float32, device=x.device)
        for group in range(self.group_calls.numel()):
            mask = self.channel_group == group
            if not mask.any():
                continue
            self.group_calls[group] += 1
            partial_x = dynamic_activation_qdq(x[..., mask], self.activation_bits, per_token=True)
            output.add_(F.linear(partial_x, self.weight[:, mask], None))
        if self.bias is not None:
            output.add_(self.bias)
        return output


@dataclasses.dataclass
class AppliedQuantization:
    model: nn.Module
    originals: dict[str, nn.Module]
    audit: dict

    def restore(self) -> None:
        for name, module in self.originals.items():
            parent, key = _parent_and_key(self.model, name)
            setattr(parent, key, module)
        self.originals.clear()

    def __enter__(self) -> "AppliedQuantization":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.restore()


def _rtn(source: nn.Linear, config: ExperimentConfig) -> tuple[nn.Module, dict]:
    group_size = (
        source.in_features
        if config.weight_granularity == "per_output_channel"
        else config.group_size
    )
    result = quantize_weight_groupwise(
        source.weight,
        bits=config.weight_bits,
        group_size=group_size,
        zero_point=config.zero_point,
    )
    wrapper: nn.Module
    if config.activation_bits is None:
        wrapper = FrozenLinear(source, result.dequantized)
    else:
        wrapper = W8A8Linear(
            source, result.dequantized, activation_bits=config.activation_bits or 8)
    return wrapper, dataclasses.asdict(result) | {"dequantized": None, "qvalues": None, "scales": None, "zero_points": None}


def _awq(
    source: nn.Linear,
    stats: ActivationStats,
    config: ExperimentConfig,
) -> tuple[nn.Module, dict]:
    weight = source.weight.detach().float()
    weight_stat = weight.abs().mean(dim=0).clamp_min(1e-6)
    activation_stat = stats.abs_mean.clamp_min(1e-6)
    sample = stats.reservoir.float()
    reference = F.linear(sample, weight, None)
    best: tuple[float, float, float, torch.Tensor, torch.Tensor, object] | None = None
    curve = []
    for alpha in config.awq_alpha_grid:
        scale = activation_stat.pow(alpha) / weight_stat.pow(1.0 - alpha)
        scale = scale.clamp_min(1e-6)
        scale = scale / torch.sqrt(scale.max() * scale.min())
        scaled_weight = weight * scale.unsqueeze(0)
        for clip_ratio in config.awq_clip_grid:
            quantized = quantize_weight_groupwise(
                scaled_weight,
                bits=config.weight_bits,
                group_size=config.group_size,
                zero_point=config.zero_point,
                clip_ratio=clip_ratio,
            )
            candidate = F.linear(sample / scale, quantized.dequantized, None)
            error = normalized_mse(reference, candidate)
            curve.append({"alpha": alpha, "clip_ratio": clip_ratio, "nmse": error})
            if best is None or error < best[0]:
                best = (error, alpha, clip_ratio, scale.clone(), quantized.dequantized.clone(), quantized)
    assert best is not None
    error, alpha, clip_ratio, scale, quantized_weight, quantized = best
    wrapper = ScaledQuantLinear(source, quantized_weight, scale)
    return wrapper, {
        "calibration_nmse": error,
        "alpha": alpha,
        "clip_ratio": clip_ratio,
        "curve": curve,
        "weight_mse": quantized.mse,
        "weight_nmse": quantized.nmse,
        "weight_sqnr_db": quantized.sqnr_db,
    }


def _smoothquant(source: nn.Linear, stats: ActivationStats, config: ExperimentConfig) -> tuple[nn.Module, dict]:
    weight = source.weight.detach().float()
    activation_stat = stats.abs_max.clamp_min(1e-6)
    weight_stat = weight.abs().amax(dim=0).clamp_min(1e-6)
    alpha = config.smoothquant_alpha
    scale = activation_stat.pow(alpha) / weight_stat.pow(1.0 - alpha)
    scale = scale.clamp_min(1e-6)
    quantized = quantize_weight_groupwise(
        weight * scale.unsqueeze(0),
        bits=config.weight_bits,
        group_size=(
            source.in_features
            if config.weight_granularity == "per_output_channel"
            else config.group_size
        ),
        zero_point=False,
    )
    return W8A8Linear(
        source, quantized.dequantized, scale,
        activation_bits=config.activation_bits or 8,
    ), {
        "alpha": alpha,
        "weight_mse": quantized.mse,
        "weight_nmse": quantized.nmse,
    }


def _vectorwise_w8(source: nn.Linear, config: ExperimentConfig) -> tuple[nn.Module, dict]:
    """Per-output-channel W8 plus per-token A8, without a dense outlier branch."""
    quantized = quantize_weight_groupwise(
        source.weight,
        bits=8,
        group_size=source.in_features,
        zero_point=False,
    )
    return W8A8Linear(source, quantized.dequantized, activation_bits=8), {
        "algorithm": "vectorwise_w8_baseline",
        "weight_mse": quantized.mse,
        "weight_nmse": quantized.nmse,
        "logical_bits_per_weight": 8 + 16 / source.in_features,
    }


def _llm_int8(source: nn.Linear, config: ExperimentConfig) -> tuple[nn.Module, dict]:
    quantized = quantize_weight_groupwise(
        source.weight, bits=8, group_size=source.in_features, zero_point=False,
    )
    return LLMInt8Linear(source, quantized.dequantized, config.llm_int8_threshold), {
        "algorithm": "vectorwise_int8_with_dynamic_fp32_outlier_branch",
        "threshold": config.llm_int8_threshold,
        "weight_mse": quantized.mse,
        "logical_bits_per_weight": 8 + 16 / source.in_features,
        "quality_eval_storage": "int8_qdq_plus_full_fp32_reference",
        "deployment_kernel_claim": False,
    }


def _tender(source: nn.Linear, stats: ActivationStats, config: ExperimentConfig) -> tuple[nn.Module, dict]:
    maximum = stats.abs_max.clamp_min(1e-12)
    top = maximum.max()
    ratio = (top / maximum).log2().floor().clamp(0, config.tender_groups - 1)
    channel_group = ratio.to(torch.long)
    quantized = quantize_weight_groupwise(
        source.weight,
        bits=config.weight_bits,
        group_size=config.group_size,
        zero_point=False,
    )
    counts = torch.bincount(channel_group, minlength=config.tender_groups).tolist()
    return TenderLinear(source, quantized.dequantized, channel_group, config.activation_bits or 8), {
        "channel_counts": counts,
        "weight_mse": quantized.mse,
    }


def _gptq_groupwise(source: nn.Linear, stats: ActivationStats, config: ExperimentConfig) -> tuple[nn.Module, dict]:
    """Blockwise sequential GPTQ with Hessian-guided error propagation."""
    reference = source.weight.detach().float()
    restored = torch.empty_like(reference)
    sample = stats.reservoir.float()
    columns = reference.shape[1]
    damp = 0.01
    fallback_blocks = 0
    for start in range(0, columns, config.group_size):
        stop = min(start + config.group_size, columns)
        x = sample[:, start:stop]
        hessian = (x.T @ x) / max(x.shape[0], 1)
        block = reference[:, start:stop].clone()
        dead = hessian.diag() == 0
        hessian[dead, dead] = 1
        block[:, dead] = 0
        diagonal_mean = hessian.diag().mean().clamp_min(1e-8)
        hessian.diagonal().add_(damp * diagonal_mean)
        try:
            inverse = torch.cholesky_inverse(torch.linalg.cholesky(hessian))
            inverse_factor = torch.linalg.cholesky(inverse, upper=True)
        except RuntimeError:
            fallback_blocks += 1
            inverse = torch.linalg.pinv(hessian)
            inverse.diagonal().clamp_min_(1e-8)
            inverse_factor = inverse

        quantizer = quantize_weight_groupwise(
            block,
            bits=config.weight_bits,
            group_size=stop - start,
            zero_point=config.zero_point,
        )
        scales = quantizer.scales[:, 0].clamp_min(1e-12)
        zeros = None if quantizer.zero_points is None else quantizer.zero_points[:, 0]
        qmin = 0 if config.zero_point else -(1 << (config.weight_bits - 1)) + 1
        qmax = (1 << config.weight_bits) - 1 if config.zero_point else (1 << (config.weight_bits - 1)) - 1
        quantized_block = torch.empty_like(block)
        work = block.clone()
        for column in range(stop - start):
            value = work[:, column]
            if zeros is None:
                quantized = torch.round(value / scales).clamp(qmin, qmax) * scales
            else:
                quantized = (
                    torch.round(value / scales + zeros).clamp(qmin, qmax) - zeros
                ) * scales
            quantized_block[:, column] = quantized
            diagonal = inverse_factor[column, column].clamp_min(1e-8)
            error = (value - quantized) / diagonal
            if column + 1 < stop - start:
                work[:, column + 1:] -= (
                    error[:, None] * inverse_factor[column, column + 1:][None, :]
                )
        restored[:, start:stop] = quantized_block
    return FrozenLinear(source, restored), {
        "algorithm": "blockwise_sequential_gptq",
        "block_size": config.group_size,
        "damp_percent": damp,
        "fallback_blocks": fallback_blocks,
        "weight_nmse": normalized_mse(reference, restored),
    }


def apply_method(
    model: nn.Module,
    selected: list[SelectedLinear],
    config: ExperimentConfig,
    calibration: Mapping[str, ActivationStats] | None = None,
) -> AppliedQuantization:
    if config.method == MethodName.BF16_REFERENCE:
        return AppliedQuantization(model, {}, {"method": config.method.value, "modules": {}})
    calibration_required = {
        MethodName.AWQ,
        MethodName.GPTQ,
        MethodName.SMOOTHQUANT,
        MethodName.TENDER,
    }
    if config.method in calibration_required and calibration is None:
        raise ValueError(f"{config.method.value} requires calibration statistics")
    if config.method in {MethodName.OMNIQUANT, MethodName.SPQR, MethodName.SQUEEZELLM}:
        raise NotImplementedError(
            f"{config.method.value} belongs to the second implementation wave; "
            "it is registered but deliberately cannot silently fall back to RTN"
        )

    originals: dict[str, nn.Module] = {}
    module_audit: dict[str, dict] = {}
    started = time.perf_counter()
    try:
        for item in selected:
            source = item.module
            if config.method == MethodName.RTN:
                replacement, audit = _rtn(source, config)
            elif config.method == MethodName.AWQ:
                replacement, audit = _awq(source, calibration[item.name], config)  # type: ignore[index]
            elif config.method == MethodName.GPTQ:
                replacement, audit = _gptq_groupwise(source, calibration[item.name], config)  # type: ignore[index]
            elif config.method == MethodName.SMOOTHQUANT:
                replacement, audit = _smoothquant(source, calibration[item.name], config)  # type: ignore[index]
            elif config.method == MethodName.LLM_INT8:
                replacement, audit = _llm_int8(source, config)
            elif config.method == MethodName.VECTORWISE_W8:
                replacement, audit = _vectorwise_w8(source, config)
            elif config.method == MethodName.TENDER:
                replacement, audit = _tender(source, calibration[item.name], config)  # type: ignore[index]
            else:
                raise ValueError(f"unsupported method: {config.method.value}")
            parent, key = _parent_and_key(model, item.name)
            originals[item.name] = source
            setattr(parent, key, replacement)
            module_audit[item.name] = audit | {"subsystem": item.subsystem}
            module_audit[item.name]["weight_count"] = source.weight.numel()
    except Exception:
        for name, original in originals.items():
            parent, key = _parent_and_key(model, name)
            setattr(parent, key, original)
        raise

    scale_bits = 16
    zero_bits = config.weight_bits if config.zero_point else 0
    if config.method in {MethodName.VECTORWISE_W8, MethodName.LLM_INT8}:
        logical_bpw = sum(
            value["logical_bits_per_weight"] * value["weight_count"]
            for value in module_audit.values()
        ) / sum(value["weight_count"] for value in module_audit.values())
    else:
        if config.weight_granularity == "per_output_channel":
            logical_bpw = sum(
                (
                    config.weight_bits
                    + (scale_bits + zero_bits) / model.get_submodule(name).in_features
                ) * value["weight_count"]
                for name, value in module_audit.items()
            ) / sum(value["weight_count"] for value in module_audit.values())
        else:
            logical_bpw = config.weight_bits + (scale_bits + zero_bits) / config.group_size
    return AppliedQuantization(
        model,
        originals,
        {
            "method": config.method.value,
            "scope": config.scope.value,
            "module_count": len(originals),
            "elapsed_seconds": time.perf_counter() - started,
            "logical_bits_per_weight": logical_bpw,
            "modules": module_audit,
        },
    )
