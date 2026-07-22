"""Activation-aware scaling for decoder and vision SwiGLU FFNs."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import torch

from .modeling import FFNTarget
from .quantization import ParameterSnapshot, WeightStats, quantize_linear_


@dataclass
class AWQResult:
    domain: str
    layer: int
    input_scale: torch.Tensor
    internal_scale: torch.Tensor
    rtn_nmse: float
    awq_nmse: float
    equivalence_max_error: float
    selected_input: str
    selected_internal: str
    fallback_reason: str | None
    input_curve: list[dict[str, Any]]
    internal_curve: list[dict[str, Any]]
    elapsed_seconds: float

    def metrics_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "layer": self.layer,
            "rtn_normalized_mse": self.rtn_nmse,
            "awq_normalized_mse": self.awq_nmse,
            "equivalence_max_error": self.equivalence_max_error,
            "selected_input": self.selected_input,
            "selected_internal": self.selected_internal,
            "fallback_reason": self.fallback_reason,
            "input_search": self.input_curve,
            "internal_search": self.internal_curve,
            "elapsed_seconds": self.elapsed_seconds,
        }


class ActivationCache:
    """Streaming channel statistics and bounded reconstruction tokens."""

    def __init__(self, ffns: list[FFNTarget], max_tokens: int = 256) -> None:
        self.ffns = ffns
        self.max_tokens = max_tokens
        self.abs_sum: dict[tuple[str, int, str], torch.Tensor] = {}
        self.count: dict[tuple[str, int, str], int] = {}
        self.samples: dict[tuple[str, int], list[torch.Tensor]] = {
            (ffn.domain, ffn.index): [] for ffn in ffns
        }
        self.handles: list[Any] = []

    def _record(self, ffn: FFNTarget, value: torch.Tensor, kind: str) -> None:
        flat = value.detach().float().reshape(-1, value.shape[-1]).cpu()
        key = (ffn.domain, ffn.index, kind)
        self.abs_sum[key] = self.abs_sum.get(key, torch.zeros(flat.shape[-1], dtype=torch.float64)) + flat.abs().double().sum(0)
        self.count[key] = self.count.get(key, 0) + flat.shape[0]
        if kind == "input":
            chunks = self.samples[(ffn.domain, ffn.index)]
            used = sum(chunk.shape[0] for chunk in chunks)
            if used < self.max_tokens:
                chunks.append(flat[: self.max_tokens - used].clone())

    def __enter__(self) -> "ActivationCache":
        for ffn in self.ffns:
            self.handles.append(
                ffn.mlp.register_forward_pre_hook(
                    lambda _module, args, target=ffn: self._record(target, args[0], "input")
                )
            )
            self.handles.append(
                ffn.down.register_forward_pre_hook(
                    lambda _module, args, target=ffn: self._record(target, args[0], "down")
                )
            )
        return self

    def __exit__(self, *_args: object) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def mean_abs(self, ffn: FFNTarget, kind: str) -> torch.Tensor:
        key = (ffn.domain, ffn.index, kind)
        if self.count.get(key, 0) == 0:
            raise RuntimeError(f"no {kind} activations for {ffn.domain} layer {ffn.index}")
        return (self.abs_sum[key] / self.count[key]).float()

    def reconstruction_inputs(self, ffn: FFNTarget) -> torch.Tensor:
        chunks = self.samples[(ffn.domain, ffn.index)]
        if not chunks:
            raise RuntimeError(f"no reconstruction inputs for {ffn.domain} layer {ffn.index}")
        return torch.cat(chunks)


def _nmse(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    error = (candidate.float() - reference.float()).pow(2).sum()
    signal = reference.float().pow(2).sum().clamp_min(1e-30)
    return float((error / signal).item())


def _scales(activation: torch.Tensor, weight: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    activation = activation.float().clamp_min(1e-8)
    weight = weight.float().clamp_min(1e-8)
    result = [("identity", torch.ones_like(activation))]
    for step in range(21):
        alpha = step / 20.0
        scale = activation.pow(alpha) / weight.pow(1.0 - alpha)
        scale = scale.clamp(1e-4, 1e4)
        scale /= torch.sqrt(scale.max() * scale.min()).clamp_min(1e-8)
        result.append((f"alpha={alpha:.2f}", scale))
    return result


@torch.no_grad()
def apply_input_scale_(ffn: FFNTarget, scale: torch.Tensor) -> None:
    scale = scale.to(ffn.norm.weight.device, ffn.norm.weight.dtype)
    ffn.norm.weight.div_(scale)
    ffn.gate.weight.mul_(scale.unsqueeze(0))
    ffn.up.weight.mul_(scale.unsqueeze(0))


@torch.no_grad()
def apply_internal_scale_(ffn: FFNTarget, scale: torch.Tensor) -> None:
    scale = scale.to(ffn.up.weight.device, ffn.up.weight.dtype)
    ffn.up.weight.div_(scale.unsqueeze(1))
    if ffn.up.bias is not None:
        ffn.up.bias.div_(scale)
    ffn.down.weight.mul_(scale.unsqueeze(0))


def _mutable(ffn: FFNTarget) -> list[tuple[str, torch.nn.Parameter]]:
    values = [(f"{ffn.name}.{name}.weight", linear.weight) for name, linear in zip(("gate_proj", "up_proj", "down_proj"), ffn.linears)]
    values.append((f"{ffn.norm_name}.weight", ffn.norm.weight))
    for name, linear in zip(("gate_proj", "up_proj", "down_proj"), ffn.linears):
        if linear.bias is not None:
            values.append((f"{ffn.name}.{name}.bias", linear.bias))
    return values


@torch.no_grad()
def quantize_ffn_(ffn: FFNTarget, group_size: int) -> list[WeightStats]:
    return [quantize_linear_(linear, bits=4, group_size=group_size) for linear in ffn.linears]


@torch.no_grad()
def search_awq(
    ffn: FFNTarget,
    inputs: torch.Tensor,
    input_mean_abs: torch.Tensor,
    down_mean_abs: torch.Tensor,
    *,
    group_size: int,
    tolerance: float = 1e-5,
) -> AWQResult:
    """Search one layer and restore it before returning reusable scales."""

    started = time.monotonic()
    inputs = inputs.to(ffn.gate.weight.device, ffn.gate.weight.dtype)
    snapshot = ParameterSnapshot(_mutable(ffn))
    reference = ffn.mlp(inputs).detach().float()
    try:
        quantize_ffn_(ffn, group_size)
        rtn = _nmse(reference, ffn.mlp(inputs))
        snapshot.restore()

        input_importance = torch.cat((ffn.gate.weight.float(), ffn.up.weight.float())).abs().mean(0)
        input_curve: list[dict[str, Any]] = []
        best_input = ("identity", torch.ones_like(input_mean_abs), math.inf)
        for name, scale in _scales(input_mean_abs, input_importance):
            snapshot.restore()
            apply_input_scale_(ffn, scale)
            scaled_inputs = inputs / scale.to(inputs.device, inputs.dtype)
            quantize_ffn_(ffn, group_size)
            error = _nmse(reference, ffn.mlp(scaled_inputs))
            input_curve.append({"candidate": name, "normalized_mse": error})
            if error < best_input[2]:
                best_input = (name, scale.clone(), error)

        snapshot.restore()
        apply_input_scale_(ffn, best_input[1])
        scaled_inputs = inputs / best_input[1].to(inputs.device, inputs.dtype)
        input_equivalence = float((ffn.mlp(scaled_inputs).float() - reference).abs().max().item())
        if input_equivalence > tolerance:
            return _fallback(ffn, input_mean_abs, down_mean_abs, rtn, input_equivalence, "input_equivalence", input_curve, started)

        after_input = ParameterSnapshot(_mutable(ffn))
        internal_importance = ffn.down.weight.float().abs().mean(0)
        internal_curve: list[dict[str, Any]] = []
        best_internal = ("identity", torch.ones_like(down_mean_abs), math.inf)
        for name, scale in _scales(down_mean_abs, internal_importance):
            after_input.restore()
            apply_internal_scale_(ffn, scale)
            quantize_ffn_(ffn, group_size)
            error = _nmse(reference, ffn.mlp(scaled_inputs))
            internal_curve.append({"candidate": name, "normalized_mse": error})
            if error < best_internal[2]:
                best_internal = (name, scale.clone(), error)

        after_input.restore()
        apply_internal_scale_(ffn, best_internal[1])
        equivalence = float((ffn.mlp(scaled_inputs).float() - reference).abs().max().item())
        if equivalence > tolerance:
            return _fallback(ffn, input_mean_abs, down_mean_abs, rtn, equivalence, "internal_equivalence", input_curve, started, internal_curve)
        quantize_ffn_(ffn, group_size)
        awq = _nmse(reference, ffn.mlp(scaled_inputs))
        if awq > rtn + 1e-12:
            return _fallback(ffn, input_mean_abs, down_mean_abs, rtn, equivalence, "not_better_than_rtn", input_curve, started, internal_curve)
        return AWQResult(
            ffn.domain, ffn.index, best_input[1].cpu(), best_internal[1].cpu(), rtn, awq,
            max(input_equivalence, equivalence), best_input[0], best_internal[0], None,
            input_curve, internal_curve, time.monotonic() - started,
        )
    finally:
        snapshot.restore()
        snapshot.assert_restored()


def _fallback(
    ffn: FFNTarget,
    input_mean: torch.Tensor,
    down_mean: torch.Tensor,
    rtn: float,
    equivalence: float,
    reason: str,
    input_curve: list[dict[str, Any]],
    started: float,
    internal_curve: list[dict[str, Any]] | None = None,
) -> AWQResult:
    return AWQResult(
        ffn.domain, ffn.index, torch.ones_like(input_mean).cpu(), torch.ones_like(down_mean).cpu(),
        rtn, rtn, equivalence, "identity", "identity", reason, input_curve,
        internal_curve or [], time.monotonic() - started,
    )


@torch.no_grad()
def apply_awq_(ffn: FFNTarget, result: AWQResult, *, group_size: int) -> list[WeightStats]:
    apply_input_scale_(ffn, result.input_scale)
    apply_internal_scale_(ffn, result.internal_scale)
    return quantize_ffn_(ffn, group_size)
