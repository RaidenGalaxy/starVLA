"""End-to-end FP32 reference caching and multimodal error evaluation."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from experiments.ffn_quant.data import QuantSample
from experiments.ffn_quant.metrics import LogitsAccumulator, TensorErrorAccumulator
from experiments.ffn_quant.modeling import build_teacher_forced_inputs

from .modeling import FFNTarget


def _first_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
        return value[0]
    raise TypeError(f"hook output contains no tensor: {type(value)!r}")


class BoundaryCapture:
    def __init__(self, model: torch.nn.Module, ffns: list[FFNTarget]) -> None:
        self.values: dict[str, dict[int, torch.Tensor] | torch.Tensor] = {
            "vision_blocks": {},
            "decoder_attention": {},
            "vision_ffn": {},
            "decoder_ffn": {},
        }
        self.handles: list[Any] = []
        visual = model.model.visual
        language_layers = model.model.language_model.layers
        for index, block in enumerate(visual.blocks):
            self.handles.append(block.register_forward_hook(self._indexed("vision_blocks", index)))
        self.handles.append(visual.merger.register_forward_hook(self._single("merger")))
        for index, layer in enumerate(language_layers):
            self.handles.append(layer.self_attn.register_forward_hook(self._indexed("decoder_attention", index)))
        for ffn in ffns:
            self.handles.append(ffn.mlp.register_forward_hook(self._indexed(f"{ffn.domain}_ffn", ffn.index)))

    def _indexed(self, key: str, index: int):
        def hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: Any) -> None:
            values = self.values[key]
            assert isinstance(values, dict)
            values[index] = _first_tensor(output).detach().float().cpu()
        return hook

    def _single(self, key: str):
        def hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: Any) -> None:
            self.values[key] = _first_tensor(output).detach().float().cpu()
        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _prediction_rows(logits: torch.Tensor, input_ids: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.nonzero(mask[0], as_tuple=False).flatten()
    positions = positions[positions > 0]
    if not positions.numel():
        raise RuntimeError("teacher-forced sample has no predictable target token")
    return logits[0, positions - 1], input_ids[0, positions]


def _prompt_inputs(processor: Any, sample: QuantSample) -> dict[str, torch.Tensor]:
    with Image.open(sample.image) as source:
        image = source.convert("RGB")
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": sample.prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    return {key: value.to("cpu") for key, value in inputs.items() if torch.is_tensor(value)}


@torch.inference_mode()
def generate_answer(model: torch.nn.Module, processor: Any, sample: QuantSample, max_new_tokens: int) -> str:
    inputs = _prompt_inputs(processor, sample)
    length = inputs["input_ids"].shape[1]
    generated = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        pad_token_id=getattr(processor.tokenizer, "pad_token_id", None),
    )
    return processor.batch_decode(generated[:, length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


@torch.inference_mode()
def create_reference_cache(
    model: torch.nn.Module,
    processor: Any,
    samples: list[QuantSample],
    ffns: list[FFNTarget],
    cache_dir: Path,
    *,
    max_new_tokens: int,
    reuse: bool,
) -> list[dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    generations: list[dict[str, Any]] = []
    for sample in samples:
        destination = cache_dir / f"{sample.id}.pt"
        if not (reuse and destination.is_file()):
            inputs, mask = build_teacher_forced_inputs(processor, sample)
            capture = BoundaryCapture(model, ffns)
            try:
                output = model(**inputs, use_cache=False, output_hidden_states=True, return_dict=True)
            finally:
                capture.close()
            logits, target_ids = _prediction_rows(output.logits, inputs["input_ids"], mask)
            artifact = {
                "id": sample.id,
                "logits": logits.detach().float().cpu(),
                "target_ids": target_ids.detach().cpu(),
                "boundaries": capture.values,
                "hidden": {index: hidden.detach().float().cpu() for index, hidden in enumerate(output.hidden_states[1:37])},
            }
            temporary = destination.with_suffix(".pt.tmp")
            torch.save(artifact, temporary)
            temporary.replace(destination)
        answer_path = cache_dir / f"{sample.id}.generation.json"
        if reuse and answer_path.is_file():
            answer = json.loads(answer_path.read_text(encoding="utf-8"))["answer"]
        else:
            answer = generate_answer(model, processor, sample, max_new_tokens)
            temporary = answer_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps({"id": sample.id, "answer": answer}, ensure_ascii=False) + "\n", encoding="utf-8")
            temporary.replace(answer_path)
        generations.append({"id": sample.id, "target": sample.target, "reference": answer})
    return generations


def _normalise(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def _edit_distance(first: list[str], second: list[str]) -> int:
    previous = list(range(len(second) + 1))
    for i, left in enumerate(first, 1):
        current = [i]
        for j, right in enumerate(second, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def generation_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    reference_matches = target_matches = positional_matches = total_positions = 0
    similarities = 0.0
    for row in rows:
        candidate = _normalise(row["candidate"])
        reference = _normalise(row["reference"])
        target = _normalise(row["target"])
        reference_matches += candidate == reference
        target_matches += candidate == target
        candidate_tokens, reference_tokens = candidate.split(), reference.split()
        width = max(len(candidate_tokens), len(reference_tokens), 1)
        positional_matches += sum(a == b for a, b in zip(candidate_tokens, reference_tokens))
        total_positions += width
        similarities += 1.0 - _edit_distance(candidate_tokens, reference_tokens) / width
    count = max(len(rows), 1)
    return {
        "samples": len(rows),
        "reference_exact_agreement": reference_matches / count,
        "target_exact_accuracy": target_matches / count,
        "token_agreement": positional_matches / max(total_positions, 1),
        "normalized_edit_similarity": similarities / count,
    }


def _new_boundaries() -> dict[str, Any]:
    return {
        "vision_blocks": {index: TensorErrorAccumulator() for index in range(32)},
        "merger": TensorErrorAccumulator(),
        "decoder_attention": {index: TensorErrorAccumulator() for index in range(36)},
        "vision_ffn": {index: TensorErrorAccumulator() for index in range(32)},
        "decoder_ffn": {index: TensorErrorAccumulator() for index in range(36)},
    }


def _merge(accumulators: dict[int, TensorErrorAccumulator]) -> dict[str, Any]:
    overall = TensorErrorAccumulator()
    for accumulator in accumulators.values():
        overall.error_power += accumulator.error_power
        overall.signal_power += accumulator.signal_power
        overall.candidate_power += accumulator.candidate_power
        overall.dot += accumulator.dot
        overall.max_abs_error = max(overall.max_abs_error, accumulator.max_abs_error)
        overall.count += accumulator.count
    return {
        "overall": overall.compute(),
        "per_layer": {str(index): accumulator.compute() for index, accumulator in accumulators.items()},
    }


@torch.inference_mode()
def evaluate_variant(
    model: torch.nn.Module,
    processor: Any,
    samples: list[QuantSample],
    ffns: list[FFNTarget],
    reference_dir: Path,
    reference_generations: list[dict[str, Any]],
    *,
    max_new_tokens: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    boundaries = _new_boundaries()
    hidden = {index: TensorErrorAccumulator() for index in range(36)}
    logits_accumulator = LogitsAccumulator()
    generated_rows: list[dict[str, Any]] = []
    references = {row["id"]: row for row in reference_generations}

    for sample in samples:
        reference = torch.load(reference_dir / f"{sample.id}.pt", map_location="cpu", weights_only=True)
        inputs, mask = build_teacher_forced_inputs(processor, sample)
        capture = BoundaryCapture(model, ffns)
        try:
            output = model(**inputs, use_cache=False, output_hidden_states=True, return_dict=True)
        finally:
            capture.close()
        for key in ("vision_blocks", "decoder_attention", "vision_ffn", "decoder_ffn"):
            ref_values = reference["boundaries"][key]
            candidate_values = capture.values[key]
            assert isinstance(candidate_values, dict)
            for index, accumulator in boundaries[key].items():
                accumulator.update(ref_values[index], candidate_values[index])
        boundaries["merger"].update(reference["boundaries"]["merger"], capture.values["merger"])
        for index, value in enumerate(output.hidden_states[1:37]):
            hidden[index].update(reference["hidden"][index], value.detach().float().cpu())
        candidate_logits, target_ids = _prediction_rows(output.logits, inputs["input_ids"], mask)
        if not torch.equal(target_ids.cpu(), reference["target_ids"]):
            raise RuntimeError(f"target tokenization changed for {sample.id}")
        logits_accumulator.update(reference["logits"], candidate_logits.cpu(), target_ids.cpu())
        generated_rows.append(
            {
                "id": sample.id,
                "target": sample.target,
                "reference": references[sample.id]["reference"],
                "candidate": generate_answer(model, processor, sample, max_new_tokens),
            }
        )

    activation_metrics = {key: _merge(value) for key, value in boundaries.items() if isinstance(value, dict)}
    activation_metrics["merger"] = boundaries["merger"].compute()
    hidden_metrics = _merge(hidden)
    hidden_metrics["final"] = hidden[35].compute()
    return {
        "activations": activation_metrics,
        "hidden_states": hidden_metrics,
        "logits": logits_accumulator.compute(),
        "generation": generation_metrics(generated_rows),
    }, generated_rows


def reference_metrics(reference_generations: list[dict[str, Any]], target_logits: dict[str, Any]) -> dict[str, Any]:
    identity = {"mse": 0.0, "normalized_mse": 0.0, "cosine_similarity": 1.0, "sqnr_db": math.inf, "max_abs_error": 0.0, "numel": 0}
    generation_rows = [{**row, "candidate": row["reference"]} for row in reference_generations]
    layered_32 = {"overall": identity, "per_layer": {str(i): identity for i in range(32)}}
    layered_36 = {"overall": identity, "per_layer": {str(i): identity for i in range(36)}}
    return {
        "activations": {
            "vision_blocks": layered_32,
            "merger": identity,
            "decoder_attention": layered_36,
            "vision_ffn": layered_32,
            "decoder_ffn": layered_36,
        },
        "hidden_states": {**layered_36, "final": identity},
        "logits": target_logits,
        "generation": generation_metrics(generation_rows),
    }


def reference_logits_metrics(cache_dir: Path, samples: list[QuantSample]) -> dict[str, Any]:
    accumulator = LogitsAccumulator()
    for sample in samples:
        artifact = torch.load(cache_dir / f"{sample.id}.pt", map_location="cpu", weights_only=True)
        accumulator.update(artifact["logits"], artifact["logits"], artifact["target_ids"])
    return accumulator.compute()
