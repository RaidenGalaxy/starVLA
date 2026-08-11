from __future__ import annotations

import json
import math
import time
from collections.abc import Iterable
from pathlib import Path

import torch
from PIL import Image

from .data import MultimodalSample
from .metrics import TokenNLL, anls, exact_match, normalize_answer, normalized_edit_similarity


def _messages(sample: MultimodalSample, include_target: bool) -> list[dict]:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(sample.image)},
            {"type": "text", "text": sample.prompt},
        ],
    }]
    if include_target:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": sample.target}]})
    return messages


def processor_inputs(processor, sample: MultimodalSample, include_target: bool):
    text = processor.apply_chat_template(
        _messages(sample, include_target), tokenize=False,
        add_generation_prompt=not include_target,
    )
    with Image.open(sample.image) as image:
        return processor(text=[text], images=[image.convert("RGB")], return_tensors="pt")


@torch.inference_mode()
def evaluate_multimodal(model, processor, samples: Iterable[MultimodalSample], *,
                        max_new_tokens: int, output_jsonl: Path) -> dict:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_jsonl.with_suffix(output_jsonl.suffix + ".tmp")
    answer_nll, rows = TokenNLL(), []
    exact_values, anls_values, edit_values = [], [], []
    started = time.perf_counter()
    with temporary.open("w") as handle:
        for sample in samples:
            prompt_inputs = processor_inputs(processor, sample, False)
            prompt_length = prompt_inputs["input_ids"].shape[-1]
            generated = model.generate(
                **prompt_inputs, do_sample=False,
                max_new_tokens=max_new_tokens, use_cache=True,
            )
            prediction = processor.batch_decode(
                generated[:, prompt_length:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            teacher = processor_inputs(processor, sample, True)
            labels = teacher["input_ids"].clone()
            # Image/prompt tokens are excluded; only the complete gold answer is scored.
            labels[:, :prompt_length] = -100
            output = model(**teacher, use_cache=False, return_dict=True)
            answer_nll.update(output.logits, labels)

            exact = exact_match(prediction, sample.target)
            similarity = normalized_edit_similarity(prediction, sample.target)
            sample_anls = anls(prediction, sample.target)
            exact_values.append(exact)
            edit_values.append(similarity)
            anls_values.append(sample_anls)
            row = {
                "id": sample.id,
                "dataset": sample.dataset,
                "prediction": prediction,
                "target": sample.target,
                "normalized_prediction": normalize_answer(prediction),
                "exact_match": exact,
                "anls": sample_anls,
                "normalized_edit_similarity": similarity,
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output_jsonl)
    return {
        "samples": len(rows),
        "exact_accuracy": sum(exact_values) / len(exact_values) if exact_values else math.nan,
        "anls": sum(anls_values) / len(anls_values) if anls_values else math.nan,
        "normalized_edit_similarity": sum(edit_values) / len(edit_values) if edit_values else math.nan,
        "conditional_answer": answer_nll.to_dict(),
        "elapsed_seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def evaluate_token_ids_ppl(model, token_ids: torch.Tensor, *, sequence_length: int = 2048) -> dict:
    """Paper-style corpus PPL, distinct from conditional answer PPL."""
    token_ids = token_ids.reshape(-1)
    metric, windows = TokenNLL(), 0
    for start in range(0, token_ids.numel() - 1, sequence_length):
        window = token_ids[start:start + sequence_length].unsqueeze(0)
        if window.shape[-1] < 2:
            continue
        output = model(input_ids=window, use_cache=False, return_dict=True)
        metric.update(output.logits, window)
        windows += 1
    return metric.to_dict() | {"windows": windows, "sequence_length": sequence_length}
