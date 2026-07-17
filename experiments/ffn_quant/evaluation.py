"""Reference caching and three-level VLM error evaluation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

from .data import QuantSample
from .metrics import LogitsAccumulator, TensorErrorAccumulator
from .modeling import ActivationCache, DecoderFFN, build_teacher_forced_inputs


def _capture_ffn_outputs(entries: list[DecoderFFN]) -> tuple[dict[int, torch.Tensor], list[Any]]:
    outputs: dict[int, torch.Tensor] = {}
    handles = [
        entry.mlp.register_forward_hook(
            lambda _module, _args, value, index=entry.index: outputs.__setitem__(index, value.detach().cpu())
        )
        for entry in entries
    ]
    return outputs, handles


def _prediction_rows(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_positions = torch.nonzero(target_mask[0], as_tuple=False).flatten()
    valid = target_positions > 0
    target_positions = target_positions[valid]
    prediction_positions = target_positions - 1
    if target_positions.numel() == 0:
        raise RuntimeError("teacher-forced sample has no predictable target token")
    return logits[0, prediction_positions], input_ids[0, target_positions]


@torch.inference_mode()
def create_reference_cache(
    model: torch.nn.Module,
    processor: Any,
    samples: list[QuantSample],
    entries: list[DecoderFFN],
    cache_dir: Path,
    *,
    reuse: bool,
) -> dict[str, float | int]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    logits_accumulator = LogitsAccumulator()
    for sample in samples:
        destination = cache_dir / f"{sample.id}.pt"
        if reuse and destination.is_file():
            artifact = torch.load(destination, map_location="cpu", weights_only=True)
            logits_accumulator.update(artifact["logits"], artifact["logits"], artifact["target_ids"])
            continue
        model_inputs, target_mask = build_teacher_forced_inputs(processor, sample)
        ffn_outputs, handles = _capture_ffn_outputs(entries)
        try:
            outputs = model(**model_inputs, use_cache=False, output_hidden_states=True, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        target_logits, target_ids = _prediction_rows(outputs.logits, model_inputs["input_ids"], target_mask)
        hidden_states = outputs.hidden_states[1 : len(entries) + 1]
        artifact = {
            "id": sample.id,
            "logits": target_logits.detach().half().cpu(),
            "target_ids": target_ids.detach().cpu(),
            "ffn": {index: value.half() for index, value in ffn_outputs.items()},
            "hidden": {index: value.detach().half().cpu() for index, value in enumerate(hidden_states)},
        }
        temporary = destination.with_suffix(".pt.tmp")
        torch.save(artifact, temporary)
        temporary.replace(destination)
        logits_accumulator.update(target_logits, target_logits, target_ids)
    return logits_accumulator.compute()


@torch.inference_mode()
def collect_calibration_activations(
    model: torch.nn.Module,
    processor: Any,
    samples: list[QuantSample],
    entries: list[DecoderFFN],
    *,
    max_tokens_per_layer: int = 256,
) -> ActivationCache:
    cache = ActivationCache(entries, max_tokens_per_layer=max_tokens_per_layer)
    with cache:
        for sample in samples:
            model_inputs, _target_mask = build_teacher_forced_inputs(processor, sample)
            model(**model_inputs, use_cache=False, output_hidden_states=False, return_dict=True)
    return cache


@torch.inference_mode()
def evaluate_variant(
    model: torch.nn.Module,
    processor: Any,
    samples: list[QuantSample],
    entries: list[DecoderFFN],
    reference_dir: Path,
) -> dict[str, Any]:
    ffn_accumulators = {entry.index: TensorErrorAccumulator() for entry in entries}
    hidden_accumulators = {entry.index: TensorErrorAccumulator() for entry in entries}
    final_hidden = TensorErrorAccumulator()
    logits_accumulator = LogitsAccumulator()

    for sample in samples:
        reference = torch.load(reference_dir / f"{sample.id}.pt", map_location="cpu", weights_only=True)
        model_inputs, target_mask = build_teacher_forced_inputs(processor, sample)
        current_ffn, handles = _capture_ffn_outputs(entries)
        try:
            outputs = model(**model_inputs, use_cache=False, output_hidden_states=True, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()

        for entry in entries:
            ffn_accumulators[entry.index].update(reference["ffn"][entry.index], current_ffn[entry.index])
        hidden_states = outputs.hidden_states[1 : len(entries) + 1]
        for index, hidden in enumerate(hidden_states):
            hidden_accumulators[index].update(reference["hidden"][index], hidden.cpu())
        final_hidden.update(reference["hidden"][len(entries) - 1], hidden_states[-1].cpu())
        candidate_logits, target_ids = _prediction_rows(outputs.logits, model_inputs["input_ids"], target_mask)
        if not torch.equal(target_ids.cpu(), reference["target_ids"]):
            raise RuntimeError(f"target tokenization changed for sample {sample.id}")
        logits_accumulator.update(reference["logits"], candidate_logits.cpu(), target_ids.cpu())

    ffn_per_layer = {str(index): accumulator.compute() for index, accumulator in ffn_accumulators.items()}
    hidden_per_layer = {str(index): accumulator.compute() for index, accumulator in hidden_accumulators.items()}
    ffn_overall = TensorErrorAccumulator()
    hidden_overall = TensorErrorAccumulator()
    # Accumulators are additive, so merge without retaining tensors.
    for source, destination in ((ffn_accumulators, ffn_overall), (hidden_accumulators, hidden_overall)):
        for accumulator in source.values():
            destination.error_power += accumulator.error_power
            destination.signal_power += accumulator.signal_power
            destination.candidate_power += accumulator.candidate_power
            destination.dot += accumulator.dot
            destination.max_abs_error = max(destination.max_abs_error, accumulator.max_abs_error)
            destination.count += accumulator.count
    return {
        "ffn": {"overall": ffn_overall.compute(), "per_layer": ffn_per_layer},
        "hidden_states": {
            "overall": hidden_overall.compute(),
            "final": final_hidden.compute(),
            "per_layer": hidden_per_layer,
        },
        "logits": logits_accumulator.compute(),
    }


def baseline_metrics(reference_logits: dict[str, float | int], entries: list[DecoderFFN]) -> dict[str, Any]:
    identity = {
        "mse": 0.0,
        "normalized_mse": 0.0,
        "cosine_similarity": 1.0,
        "sqnr_db": math.inf,
        "max_abs_error": 0.0,
        "numel": 0,
    }
    return {
        "ffn": {"overall": identity, "per_layer": {str(entry.index): identity for entry in entries}},
        "hidden_states": {
            "overall": identity,
            "final": identity,
            "per_layer": {str(entry.index): identity for entry in entries},
        },
        "logits": reference_logits,
    }
