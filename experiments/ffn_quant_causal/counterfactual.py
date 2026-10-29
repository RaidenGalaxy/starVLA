"""Offline FFN weight-by-trace replay for causal quantization analysis."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from experiments.infinitevl_quant.quantization import quantize_weight

from .trace import FFNTrace, tensor_distribution_summary


@dataclasses.dataclass(frozen=True)
class FFNWeights:
    """Detached CPU weights for a standard gate/up/down SwiGLU FFN."""

    gate_weight: torch.Tensor
    up_weight: torch.Tensor
    down_weight: torch.Tensor
    gate_bias: torch.Tensor | None = None
    up_bias: torch.Tensor | None = None
    down_bias: torch.Tensor | None = None

    @classmethod
    def from_mlp(cls, mlp: nn.Module) -> "FFNWeights":
        values: dict[str, torch.Tensor | None] = {}
        for projection in ("gate", "up", "down"):
            module = getattr(mlp, f"{projection}_proj", None)
            weight = getattr(module, "weight", None)
            if not isinstance(weight, torch.Tensor):
                raise TypeError(f"mlp.{projection}_proj must expose a tensor weight")
            bias = getattr(module, "bias", None)
            if bias is not None and not isinstance(bias, torch.Tensor):
                raise TypeError(f"mlp.{projection}_proj.bias must be a tensor or None")
            values[f"{projection}_weight"] = weight.detach().float().cpu().clone()
            values[f"{projection}_bias"] = (
                None if bias is None else bias.detach().float().cpu().clone()
            )
        result = cls(**values)  # type: ignore[arg-type]
        result.validate()
        return result

    def validate(self) -> None:
        weights = (self.gate_weight, self.up_weight, self.down_weight)
        if any(value.ndim != 2 for value in weights):
            raise ValueError("all FFN weights must be two dimensional")
        if self.gate_weight.shape != self.up_weight.shape:
            raise ValueError("gate and up projection weights must have identical shapes")
        intermediate, hidden = self.gate_weight.shape
        if self.down_weight.shape != (hidden, intermediate):
            raise ValueError(
                "down projection shape must be [hidden, intermediate], got "
                f"{tuple(self.down_weight.shape)}"
            )
        for name, value in self.named_tensors().items():
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite and floating point")
        for name, bias, size in (
            ("gate_bias", self.gate_bias, intermediate),
            ("up_bias", self.up_bias, intermediate),
            ("down_bias", self.down_bias, hidden),
        ):
            if bias is not None and (bias.ndim != 1 or bias.numel() != size):
                raise ValueError(f"{name} has an incompatible shape")

    def named_tensors(self) -> dict[str, torch.Tensor]:
        values = {
            "gate_weight": self.gate_weight,
            "up_weight": self.up_weight,
            "down_weight": self.down_weight,
        }
        for name in ("gate_bias", "up_bias", "down_bias"):
            value = getattr(self, name)
            if value is not None:
                values[name] = value
        return values


@dataclasses.dataclass(frozen=True)
class FFNReplay:
    """Intermediate tensors from one offline SwiGLU replay."""

    mode: str
    bits: int | None
    group_size: int | None
    gate: torch.Tensor
    up: torch.Tensor
    product: torch.Tensor
    down_output: torch.Tensor
    weight_audit: dict[str, dict[str, int | float]]

    def stage_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "gate": self.gate,
            "up": self.up,
            "product": self.product,
            "down_output": self.down_output,
        }


@dataclasses.dataclass(frozen=True)
class CounterfactualPair:
    """Reference-value and RTN replays for one weight-by-trace pairing."""

    reference: FFNReplay
    rtn: FFNReplay
    summary: dict[str, Any]


def _reference_audit(weight: torch.Tensor) -> dict[str, int | float]:
    return {
        "mse": 0.0,
        "nmse": 0.0,
        "max_abs_error": 0.0,
        "groups_per_row": 0,
        "parameters": weight.numel(),
    }


@torch.no_grad()
def replay_ffn(
    ffn_input: torch.Tensor,
    weights: FFNWeights,
    *,
    bits: int | None = None,
    group_size: int = 128,
    zero_point: bool = True,
) -> FFNReplay:
    """Replay a SwiGLU FFN using reference values or asymmetric RTN QDQ.

    Both paths compute in FP32 on CPU.  With ``bits=None``, values are used as
    supplied (for loaded checkpoints these are BF16 values widened to FP32).
    """

    weights.validate()
    if not isinstance(ffn_input, torch.Tensor) or ffn_input.ndim == 0:
        raise TypeError("ffn_input must be a tensor with a channel dimension")
    if ffn_input.shape[-1] != weights.gate_weight.shape[1]:
        raise ValueError("ffn_input width does not match the projection weights")
    if not torch.isfinite(ffn_input).all():
        raise ValueError("ffn_input contains NaN or Inf")
    if bits is not None and bits not in (2, 3, 4, 8):
        raise ValueError("RTN bits must be 2, 3, 4, or 8")
    if group_size <= 0:
        raise ValueError("group_size must be positive")

    projection_weights: dict[str, torch.Tensor] = {}
    audit: dict[str, dict[str, int | float]] = {}
    for projection in ("gate", "up", "down"):
        source = getattr(weights, f"{projection}_weight").float()
        if bits is None:
            projection_weights[projection] = source
            audit[projection] = _reference_audit(source)
        else:
            quantized = quantize_weight(
                source,
                bits=bits,
                group_size=group_size,
                zero_point=zero_point,
            )
            projection_weights[projection] = quantized.dequantized
            audit[projection] = {
                "mse": quantized.mse,
                "nmse": quantized.nmse,
                "max_abs_error": quantized.max_abs_error,
                "groups_per_row": quantized.groups,
                "parameters": source.numel(),
            }

    value = ffn_input.detach().float().cpu()
    gate = F.linear(value, projection_weights["gate"], weights.gate_bias)
    up = F.linear(value, projection_weights["up"], weights.up_bias)
    product = F.silu(gate) * up
    down = F.linear(product, projection_weights["down"], weights.down_bias)
    return FFNReplay(
        mode=(
            "checkpoint_values_fp32_compute"
            if bits is None else f"asymmetric_groupwise_rtn_w{bits}_fp32_compute"
        ),
        bits=bits,
        group_size=None if bits is None else group_size,
        gate=gate,
        up=up,
        product=product,
        down_output=down,
        weight_audit=audit,
    )


@torch.no_grad()
def tensor_error_summary(
    reference: torch.Tensor, candidate: torch.Tensor,
) -> dict[str, int | float]:
    """Calculate stable, additive-style error statistics for two tensors."""

    if reference.shape != candidate.shape:
        raise ValueError(
            f"tensor shape mismatch: {tuple(reference.shape)} != {tuple(candidate.shape)}"
        )
    ref = reference.detach().float().reshape(-1).cpu()
    cand = candidate.detach().float().reshape(-1).cpu()
    if not torch.isfinite(ref).all() or not torch.isfinite(cand).all():
        raise ValueError("error metrics require finite tensors")
    error = cand - ref
    error_power = float(torch.dot(error, error))
    signal_power = float(torch.dot(ref, ref))
    candidate_power = float(torch.dot(cand, cand))
    denominator = math.sqrt(signal_power * candidate_power)
    if denominator:
        cosine = float(torch.dot(ref, cand)) / denominator
    else:
        cosine = 1.0 if signal_power == candidate_power else 0.0
    count = max(ref.numel(), 1)
    nmse = error_power / max(signal_power, torch.finfo(torch.float32).tiny)
    return {
        "numel": ref.numel(),
        "mse": error_power / count,
        "normalized_mse": nmse,
        "relative_rms_error": math.sqrt(nmse),
        "cosine_similarity": cosine,
        "sqnr_db": (
            math.inf if error_power == 0.0
            else 10.0 * math.log10(
                max(signal_power, torch.finfo(torch.float32).tiny) / error_power
            )
        ),
        "max_abs_error": float(error.abs().max()) if error.numel() else 0.0,
    }


def _captured_stage_tensors(trace: FFNTrace) -> dict[str, torch.Tensor]:
    return {
        "gate": trace.gate,
        "up": trace.up,
        "product": trace.product,
        "down_output": trace.down_output,
    }


@torch.no_grad()
def summarize_replays(
    reference: FFNReplay,
    candidate: FFNReplay,
    *,
    residual: torch.Tensor | None = None,
    captured: FFNTrace | None = None,
) -> dict[str, Any]:
    """Summarize RTN local errors, optional capture parity, and branch impact."""

    errors = {
        name: tensor_error_summary(reference.stage_tensors()[name], value)
        for name, value in candidate.stage_tensors().items()
    }
    result: dict[str, Any] = {
        "analysis_scope": "offline_ffn_local_replay",
        "end_to_end_model_score": False,
        "reference_mode": reference.mode,
        "candidate_mode": candidate.mode,
        "stage_error": errors,
        "candidate_weight_audit": candidate.weight_audit,
        "reference_distributions": {
            name: tensor_distribution_summary(value)
            for name, value in reference.stage_tensors().items()
        },
    }
    if captured is not None:
        result["reference_vs_captured"] = {
            name: tensor_error_summary(value, reference.stage_tensors()[name])
            for name, value in _captured_stage_tensors(captured).items()
        }
    if residual is not None:
        if residual.shape != reference.down_output.shape:
            raise ValueError("residual shape must match the replayed down output")
        residual_flat = residual.detach().float().reshape(-1).cpu()
        reference_down = reference.down_output.float().reshape(-1)
        candidate_down = candidate.down_output.float().reshape(-1)
        residual_rms = math.sqrt(float(residual_flat.square().mean()))
        reference_rms = math.sqrt(float(reference_down.square().mean()))
        candidate_rms = math.sqrt(float(candidate_down.square().mean()))
        error_rms = math.sqrt(float((candidate_down - reference_down).square().mean()))

        def branch_cosine(branch: torch.Tensor) -> float:
            branch_power = float(torch.dot(branch, branch))
            residual_power = float(torch.dot(residual_flat, residual_flat))
            denominator = math.sqrt(branch_power * residual_power)
            if denominator:
                return float(torch.dot(residual_flat, branch)) / denominator
            return 1.0 if branch_power == residual_power else 0.0

        tiny = torch.finfo(torch.float32).tiny
        result["residual_geometry"] = {
            "reference_ffn_to_residual_rms": reference_rms / max(residual_rms, tiny),
            "candidate_ffn_to_residual_rms": candidate_rms / max(residual_rms, tiny),
            "quant_error_to_residual_rms": error_rms / max(residual_rms, tiny),
            "reference_ffn_residual_cosine": branch_cosine(reference_down),
            "candidate_ffn_residual_cosine": branch_cosine(candidate_down),
            "reference_merged_rms": math.sqrt(
                float((residual_flat + reference_down).square().mean())
            ),
            "candidate_merged_rms": math.sqrt(
                float((residual_flat + candidate_down).square().mean())
            ),
        }
    return result


@torch.no_grad()
def replay_counterfactual_pair(
    trace: FFNTrace,
    weights: FFNWeights,
    *,
    bits: int,
    group_size: int = 128,
    zero_point: bool = True,
    compare_capture: bool = False,
) -> CounterfactualPair:
    """Run reference and RTN replays for one trace/weight combination."""

    reference = replay_ffn(trace.ffn_input, weights)
    rtn = replay_ffn(
        trace.ffn_input,
        weights,
        bits=bits,
        group_size=group_size,
        zero_point=zero_point,
    )
    summary = summarize_replays(
        reference,
        rtn,
        residual=trace.residual,
        captured=trace if compare_capture else None,
    )
    return CounterfactualPair(reference=reference, rtn=rtn, summary=summary)


@torch.no_grad()
def run_weight_trace_matrix(
    traces: Mapping[str, FFNTrace],
    weight_sets: Mapping[str, FFNWeights],
    *,
    bits: Sequence[int] = (4,),
    group_size: int = 128,
    zero_point: bool = True,
) -> dict[str, Any]:
    """Return JSON-ready rows for every trace × weight × RTN-bit pairing.

    Capture parity is reported only when trace and weight source names match;
    cross-source replays intentionally have no captured-output reference.
    """

    if not traces or not weight_sets:
        raise ValueError("at least one trace and one weight set are required")
    if not bits:
        raise ValueError("at least one RTN bit width is required")
    rows = []
    for trace_source, trace in traces.items():
        for weight_source, weights in weight_sets.items():
            for width in bits:
                pair = replay_counterfactual_pair(
                    trace,
                    weights,
                    bits=width,
                    group_size=group_size,
                    zero_point=zero_point,
                    compare_capture=trace_source == weight_source,
                )
                rows.append({
                    "trace_source": trace_source,
                    "weight_source": weight_source,
                    "bits": width,
                    "group_size": group_size,
                    "zero_point": zero_point,
                    "summary": pair.summary,
                })
    return {
        "schema_version": 1,
        "analysis_scope": "offline_ffn_local_replay",
        "end_to_end_model_score": False,
        "compute_dtype": "float32",
        "quantizer": "asymmetric_groupwise_rtn" if zero_point else "symmetric_groupwise_rtn",
        "rows": rows,
    }
