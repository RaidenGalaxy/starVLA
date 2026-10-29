from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from experiments.ffn_quant_causal.adapters import ModelBundle
from experiments.ffn_quant_causal.budget import WallclockBudget
from experiments.infinitevl_quant.evaluation import (
    _compare_logits,
    _target_metrics,
    _teacher_inputs,
)
from experiments.infinitevl_quant.inventory import decoder_layers
from experiments.infinitevl_quant.progress import StageProgress
from experiments.vlm_quant_benchmark.data import MultimodalSample

from .shadow import ShadowObservation, shadow_ffns, tensor_rms


def _hidden(output: Any, *, location: str) -> torch.Tensor:
    value = output[0] if isinstance(output, tuple) else output
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{location} did not produce a hidden-state tensor")
    return value


def _replace_hidden(output: Any, value: torch.Tensor) -> Any:
    return (value, *output[1:]) if isinstance(output, tuple) else value


class _ExactErrorCollector:
    def __init__(self, layers: Sequence[int]) -> None:
        self.requested = set(int(layer) for layer in layers)
        self.errors: dict[int, torch.Tensor] = {}

    def __call__(self, observation: ShadowObservation) -> None:
        if observation.layer in self.requested:
            self.errors[observation.layer] = (
                (observation.quantized["down_output"] - observation.reference["down_output"])
                .detach()
                .float()
                .cpu()
                .clone()
            )

    def validate(self) -> None:
        if set(self.errors) != self.requested:
            raise RuntimeError(f"exact FFN error capture mismatch: {sorted(self.errors)} != {sorted(self.requested)}")


@torch.no_grad()
def capture_exact_ffn_errors(
    bundle: ModelBundle,
    teacher: Mapping[str, torch.Tensor],
    *,
    predecessor_layers: Sequence[int],
    bits: int,
    group_size: int,
) -> dict[int, torch.Tensor]:
    layers = list(decoder_layers(bundle.model))
    collector = _ExactErrorCollector(predecessor_layers)
    with shadow_ffns(
        layers,
        decoder_prefix=bundle.decoder_prefix,
        gammas=0.0,
        bits=bits,
        group_size=group_size,
        observer=collector,
    ):
        bundle.model(**teacher, use_cache=False, return_dict=True)
    collector.validate()
    return collector.errors


@torch.no_grad()
def perturb_block_input(
    bundle: ModelBundle,
    teacher: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    positions: Sequence[int],
    *,
    target_layer: int,
    perturbation: torch.Tensor,
    bits: int,
    group_size: int,
) -> dict[str, Any]:
    layers = list(decoder_layers(bundle.model))
    if target_layer < 0 or target_layer >= len(layers):
        raise ValueError(f"target layer {target_layer} is out of range")
    layer = layers[target_layer]
    attention = getattr(layer, "self_attn", None)
    if not isinstance(attention, nn.Module):
        raise TypeError(f"layer {target_layer} has no attention module")
    captures: dict[str, dict[str, torch.Tensor]] = {
        "baseline": {},
        "candidate": {},
    }

    def run(key: str, delta: torch.Tensor | None):
        handles = []
        try:
            if delta is not None:

                def inject(_module, args, kwargs):
                    if args and isinstance(args[0], torch.Tensor):
                        if args[0].shape != delta.shape:
                            raise RuntimeError("block-input perturbation shape mismatch")
                        return (args[0] + delta.to(args[0].device, args[0].dtype), *args[1:]), kwargs
                    hidden = kwargs.get("hidden_states")
                    if not isinstance(hidden, torch.Tensor) or hidden.shape != delta.shape:
                        raise RuntimeError("cannot locate compatible block hidden_states")
                    return args, {**kwargs, "hidden_states": hidden + delta.to(hidden.device, hidden.dtype)}

                handles.append(layer.register_forward_pre_hook(inject, with_kwargs=True))

            def capture_attention(_module, _args, output):
                captures[key]["attention"] = (
                    _hidden(
                        output,
                        location="attention",
                    )
                    .detach()
                    .float()
                    .cpu()
                    .clone()
                )

            def capture_block(_module, _args, output):
                captures[key]["block"] = (
                    _hidden(
                        output,
                        location="decoder block",
                    )
                    .detach()
                    .float()
                    .cpu()
                    .clone()
                )

            handles.append(attention.register_forward_hook(capture_attention))
            handles.append(layer.register_forward_hook(capture_block))
            with shadow_ffns(
                layers,
                decoder_prefix=bundle.decoder_prefix,
                gammas=0.0,
                bits=bits,
                group_size=group_size,
            ):
                return bundle.model(**teacher, use_cache=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()

    baseline = run("baseline", None)
    candidate = run("candidate", perturbation)
    input_rms = tensor_rms(perturbation)
    if input_rms <= 0:
        raise ValueError("localization perturbation must be nonzero")
    attention_error = captures["candidate"]["attention"] - captures["baseline"]["attention"]
    block_error = captures["candidate"]["block"] - captures["baseline"]["block"]
    prediction_rows = torch.tensor(
        [position - 1 for position in positions],
        dtype=torch.long,
    )
    baseline_logits = baseline.logits[0, prediction_rows].detach().float().cpu()
    candidate_logits = candidate.logits[0, prediction_rows].detach().float().cpu()
    baseline_target = _target_metrics(baseline.logits, labels)
    candidate_target = _target_metrics(candidate.logits, labels)
    return {
        "perturbation_rms": input_rms,
        "attention_gain": tensor_rms(attention_error) / input_rms,
        "block_gain": tensor_rms(block_error) / input_rms,
        "baseline_target_nll": baseline_target["nll"],
        "candidate_target_nll": candidate_target["nll"],
        "delta_target_nll": candidate_target["nll"] - baseline_target["nll"],
        **_compare_logits(candidate_logits, baseline_logits),
    }


def _noise_like(
    error: torch.Tensor,
    *,
    sample_id: str,
    layer: int,
    seed: int,
) -> torch.Tensor:
    digest = hashlib.sha256(f"{seed}:{sample_id}:{layer}:gaussian".encode()).digest()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int.from_bytes(digest[:8], "little") % (2**63 - 1))
    noise = torch.randn(error.shape, generator=generator, dtype=torch.float32)
    noise_rms = tensor_rms(noise)
    return noise * (tensor_rms(error) / noise_rms)


def _row_id(
    sample_id: str,
    pair: tuple[int, int],
    target: int,
    direction: str,
    dose: float,
) -> str:
    return f"{sample_id}|pair{pair[0]:02d}-{pair[1]:02d}|layer{target:02d}|{direction}|dose{dose:g}"


def run_localization_grid(
    bundle: ModelBundle,
    samples: Sequence[MultimodalSample],
    *,
    progress: StageProgress,
    variant: str,
    pairs: Sequence[tuple[int, int]],
    directions: Sequence[str],
    doses: Sequence[float],
    min_pixels: int,
    max_pixels: int,
    bits: int,
    group_size: int,
    seed: int,
    wallclock_budget: WallclockBudget | None = None,
) -> list[dict[str, Any]]:
    if bundle.kind != "infinitevl":
        raise ValueError("GDN/SWA localization is defined on InfiniteVL")
    if any(direction not in {"exact", "gaussian"} for direction in directions):
        raise ValueError("localization directions are exact/gaussian")
    for sample in samples:
        if wallclock_budget is not None:
            wallclock_budget.check(
                stage=progress.root.name,
                boundary="before_localization_sample",
                variant=variant,
                sample_id=sample.id,
            )
        _prompt, teacher, labels, positions = _teacher_inputs(
            bundle.processor,
            sample,
            min_pixels,
            max_pixels,
        )
        predecessors = sorted({target - 1 for pair in pairs for target in pair})
        errors = capture_exact_ffn_errors(
            bundle,
            teacher,
            predecessor_layers=predecessors,
            bits=bits,
            group_size=group_size,
        )
        for pair in pairs:
            for target, structure in zip(pair, ("gdn", "swa"), strict=True):
                exact = errors[target - 1]
                for direction in directions:
                    base = (
                        exact
                        if direction == "exact"
                        else _noise_like(
                            exact,
                            sample_id=sample.id,
                            layer=target,
                            seed=seed,
                        )
                    )
                    for dose in doses:
                        identifier = _row_id(sample.id, pair, target, direction, dose)
                        if progress.completed_sample(variant, identifier) is not None:
                            continue
                        if wallclock_budget is not None:
                            wallclock_budget.check(
                                stage=progress.root.name,
                                boundary="before_localization_cell",
                                variant=variant,
                                sample_id=identifier,
                            )
                        started = time.perf_counter()
                        metrics = perturb_block_input(
                            bundle,
                            teacher,
                            labels,
                            positions,
                            target_layer=target,
                            perturbation=base * float(dose),
                            bits=bits,
                            group_size=group_size,
                        )
                        progress.complete_sample(
                            variant,
                            identifier,
                            {
                                "id": identifier,
                                "sample_id": sample.id,
                                "dataset": sample.dataset,
                                "pair": f"{pair[0]}-{pair[1]}",
                                "layer": target,
                                "structure": structure,
                                "direction": direction,
                                "dose": float(dose),
                                "seconds": time.perf_counter() - started,
                                **metrics,
                            },
                        )
                        if wallclock_budget is not None:
                            wallclock_budget.check(
                                stage=progress.root.name,
                                boundary="after_localization_cell_committed",
                                variant=variant,
                                sample_id=identifier,
                            )
    return progress.rows(variant)
