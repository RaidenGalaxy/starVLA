"""FFN-only activation-aware fake quantization for Qwen-style SwiGLU MLPs."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch

from .modeling import DecoderFFN
from .quantization import ParameterSnapshot, quantize_linear_


@dataclass
class AWQLayerResult:
    layer: int
    input_scale: torch.Tensor
    internal_scale: torch.Tensor
    rtn_normalized_mse: float
    awq_normalized_mse: float
    equivalence_max_error: float
    input_search: list[dict[str, float | str]]
    internal_search: list[dict[str, float | str]]
    elapsed_seconds: float

    def metrics_dict(self) -> dict[str, object]:
        return {
            "layer": self.layer,
            "rtn_normalized_mse": self.rtn_normalized_mse,
            "awq_normalized_mse": self.awq_normalized_mse,
            "equivalence_max_error": self.equivalence_max_error,
            "input_search": self.input_search,
            "internal_search": self.internal_search,
            "elapsed_seconds": self.elapsed_seconds,
        }


def _normalized_mse(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    error = (candidate.float() - reference.float()).pow(2).sum()
    signal = reference.float().pow(2).sum().clamp_min(1e-30)
    return float((error / signal).item())


def _candidate_scales(activation: torch.Tensor, weight: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    activation = activation.float().clamp_min(1e-8)
    weight = weight.float().clamp_min(1e-8)
    candidates: list[tuple[str, torch.Tensor]] = [("identity", torch.ones_like(activation))]
    for step in range(21):
        alpha = step / 20.0
        scale = activation.pow(alpha) / weight.pow(1.0 - alpha)
        scale = scale.clamp(1e-4, 1e4)
        scale = scale / torch.sqrt(scale.max() * scale.min()).clamp_min(1e-8)
        candidates.append((f"alpha={alpha:.2f}", scale))
    return candidates


@torch.no_grad()
def apply_input_scale_(entry: DecoderFFN, scale: torch.Tensor) -> None:
    """Equivalent transform: norm / s, gate and up input columns * s."""

    scale = scale.to(entry.norm.weight.device, entry.norm.weight.dtype)
    entry.norm.weight.div_(scale)
    entry.linears[0].weight.mul_(scale.unsqueeze(0))
    entry.linears[1].weight.mul_(scale.unsqueeze(0))


@torch.no_grad()
def apply_internal_scale_(entry: DecoderFFN, scale: torch.Tensor) -> None:
    """Equivalent transform: up output / s, down input columns * s."""

    up, down = entry.linears[1], entry.linears[2]
    scale = scale.to(up.weight.device, up.weight.dtype)
    up.weight.div_(scale.unsqueeze(1))
    if up.bias is not None:
        up.bias.div_(scale)
    down.weight.mul_(scale.unsqueeze(0))


@torch.no_grad()
def _quantize_mlp_(entry: DecoderFFN, group_size: int) -> None:
    for linear in entry.linears:
        quantize_linear_(linear, bits=4, group_size=group_size)


def _mutable_parameters(entry: DecoderFFN) -> list[tuple[str, torch.nn.Parameter]]:
    parameters = list(entry.named_mutable_parameters())
    if entry.linears[1].bias is not None:
        parameters.append((f"{entry.mlp_name}.up_proj.bias", entry.linears[1].bias))
    return parameters


@torch.no_grad()
def search_awq_layer(
    entry: DecoderFFN,
    reconstruction_inputs: torch.Tensor,
    input_mean_abs: torch.Tensor,
    down_mean_abs: torch.Tensor,
    *,
    group_size: int = 128,
    equivalence_tolerance: float = 1e-5,
) -> AWQLayerResult:
    """Search and leave one decoder MLP in its final AWQ fake-quantized state."""

    started = time.monotonic()
    device = entry.linears[0].weight.device
    dtype = entry.linears[0].weight.dtype
    inputs = reconstruction_inputs.to(device=device, dtype=dtype)
    reference = entry.mlp(inputs).detach().float()
    signal = reference.pow(2).sum().clamp_min(1e-30)

    original = ParameterSnapshot(_mutable_parameters(entry))
    _quantize_mlp_(entry, group_size)
    rtn_error = float(((entry.mlp(inputs).float() - reference).pow(2).sum() / signal).item())
    original.restore()

    gate, up, down = entry.linears
    input_weight_importance = torch.cat((gate.weight.float(), up.weight.float()), dim=0).abs().mean(0)
    input_candidates = _candidate_scales(input_mean_abs, input_weight_importance)
    input_curve: list[dict[str, float | str]] = []
    best_input_name = "identity"
    best_input_scale = torch.ones_like(input_mean_abs)
    best_input_error = math.inf
    for name, scale in input_candidates:
        original.restore()
        apply_input_scale_(entry, scale)
        transformed_inputs = inputs / scale.to(device=device, dtype=dtype)
        _quantize_mlp_(entry, group_size)
        error = _normalized_mse(reference, entry.mlp(transformed_inputs))
        input_curve.append({"candidate": name, "normalized_mse": error})
        if error < best_input_error:
            best_input_error = error
            best_input_name = name
            best_input_scale = scale.clone()

    original.restore()
    apply_input_scale_(entry, best_input_scale)
    scaled_inputs = inputs / best_input_scale.to(device=device, dtype=dtype)
    input_equivalence = float((entry.mlp(scaled_inputs).float() - reference).abs().max().item())
    if input_equivalence > equivalence_tolerance:
        original.restore()
        raise RuntimeError(
            f"layer {entry.index} input scaling is not equivalent: max error {input_equivalence:.3e}"
        )

    after_input = ParameterSnapshot(_mutable_parameters(entry))
    internal_weight_importance = down.weight.float().abs().mean(0)
    internal_candidates = _candidate_scales(down_mean_abs, internal_weight_importance)
    internal_curve: list[dict[str, float | str]] = []
    best_internal_name = "identity"
    best_internal_scale = torch.ones_like(down_mean_abs)
    best_internal_error = math.inf
    for name, scale in internal_candidates:
        after_input.restore()
        apply_internal_scale_(entry, scale)
        _quantize_mlp_(entry, group_size)
        error = _normalized_mse(reference, entry.mlp(scaled_inputs))
        internal_curve.append({"candidate": name, "normalized_mse": error})
        if error < best_internal_error:
            best_internal_error = error
            best_internal_name = name
            best_internal_scale = scale.clone()

    after_input.restore()
    apply_internal_scale_(entry, best_internal_scale)
    full_precision_output = entry.mlp(scaled_inputs).float()
    equivalence_error = float((full_precision_output - reference).abs().max().item())
    if equivalence_error > equivalence_tolerance:
        original.restore()
        raise RuntimeError(
            f"layer {entry.index} internal scaling is not equivalent: max error {equivalence_error:.3e}"
        )
    _quantize_mlp_(entry, group_size)
    awq_error = _normalized_mse(reference, entry.mlp(scaled_inputs))
    if awq_error > rtn_error + 1e-12:
        original.restore()
        _quantize_mlp_(entry, group_size)
        best_input_name = "identity"
        best_internal_name = "identity"
        best_input_scale = torch.ones_like(input_mean_abs)
        best_internal_scale = torch.ones_like(down_mean_abs)
        awq_error = rtn_error

    input_curve.append({"selected": best_input_name, "normalized_mse": best_input_error})
    internal_curve.append({"selected": best_internal_name, "normalized_mse": best_internal_error})
    return AWQLayerResult(
        layer=entry.index,
        input_scale=best_input_scale.cpu(),
        internal_scale=best_internal_scale.cpu(),
        rtn_normalized_mse=rtn_error,
        awq_normalized_mse=awq_error,
        equivalence_max_error=max(input_equivalence, equivalence_error),
        input_search=input_curve,
        internal_search=internal_curve,
        elapsed_seconds=time.monotonic() - started,
    )
