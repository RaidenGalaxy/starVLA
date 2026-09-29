from __future__ import annotations

import csv
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.vlm_quant_benchmark.data import MultimodalSample
from experiments.vlm_quant_benchmark.dataset_protocol import (
    canonical_answers,
    docvqa_anls_score,
    normalize_vqa_answer,
    textvqa_consensus_score,
)
from experiments.vlm_quant_benchmark.evaluation import (
    processor_inputs,
    target_token_span,
)

from .progress import StageProgress, atomic_json, safe_id


_TEXTVQA_ANSWERS: dict[str, str] | None = None


def _textvqa_answers() -> dict[str, str]:
    global _TEXTVQA_ANSWERS
    if _TEXTVQA_ANSWERS is None:
        path = (
            Path(__file__).resolve().parents[2]
            / "data/vlm_quant_benchmark/LMUData/TextVQA_VAL_local.tsv"
        )
        csv.field_size_limit(sys.maxsize)
        with path.open(newline="") as handle:
            _TEXTVQA_ANSWERS = {
                str(row["index"]): row["answer"]
                for row in csv.DictReader(handle, delimiter="\t")
            }
    return _TEXTVQA_ANSWERS


def _prediction_score(sample: MultimodalSample, prediction: str) -> float:
    if sample.dataset == "DocVQA_VAL":
        return docvqa_anls_score(sample.target, prediction)
    if sample.dataset == "TextVQA_VAL":
        answers = _textvqa_answers().get(sample.official_index)
        if answers is not None:
            return textvqa_consensus_score(answers, prediction)
        return float(normalize_vqa_answer(sample.target) == normalize_vqa_answer(prediction))
    answers = {normalize_vqa_answer(item) for item in canonical_answers(sample.target)}
    normalized = normalize_vqa_answer(prediction)
    if normalized in answers:
        return 1.0
    expected = {answer.upper() for answer in answers}
    inferred = re.findall(r"\b([A-E])\b", prediction.upper())
    choice = inferred[-1] if inferred else normalized[:1].upper()
    return float(choice in expected)


def _teacher_inputs(processor, sample: MultimodalSample, min_pixels: int, max_pixels: int):
    prompt = processor_inputs(
        processor, sample, False, min_pixels=min_pixels, max_pixels=max_pixels,
    )
    teacher = processor_inputs(
        processor, sample, True, min_pixels=min_pixels, max_pixels=max_pixels,
    )
    tokenizer = processor.tokenizer
    terminator = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if terminator is None or terminator < 0:
        raise ValueError("processor tokenizer is missing <|im_end|>")
    labels, positions = target_token_span(
        prompt["input_ids"], teacher["input_ids"], im_end_token_id=terminator,
        excluded_trailing_token_ids=set(tokenizer.encode("\n", add_special_tokens=False)),
    )
    return prompt, teacher, labels, positions


def _target_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    shifted_logits = logits[..., :-1, :].float()
    shifted_labels = labels[..., 1:].to(logits.device)
    mask = shifted_labels.ne(-100)
    if not mask.any():
        raise ValueError("teacher-forced sample has no target tokens")
    losses = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1), ignore_index=-100, reduction="none",
    ).reshape_as(shifted_labels)[mask]
    return {
        "tokens": int(mask.sum()), "total_nll": float(losses.sum()),
        "nll": float(losses.mean()), "ppl": math.exp(min(float(losses.mean()), 80.0)),
    }


def _reference_path(reference_dir: Path, sample_id: str) -> Path:
    return reference_dir / f"{safe_id(sample_id)}.pt"


def _compare_logits(candidate: torch.Tensor, reference: torch.Tensor) -> dict:
    candidate = candidate.float()
    reference = reference.float()
    reference_log = F.log_softmax(reference, dim=-1)
    candidate_log = F.log_softmax(candidate, dim=-1)
    kl = F.kl_div(candidate_log, reference_log.exp(), reduction="batchmean")
    agree = candidate.argmax(-1).eq(reference.argmax(-1)).float().mean()
    return {"logits_kl": float(kl), "target_argmax_consistency": float(agree)}


def _hidden_nmse(candidate: tuple[torch.Tensor, ...], reference: list[torch.Tensor]) -> dict:
    if len(candidate) - 1 != len(reference):
        raise RuntimeError("hidden-state layer count changed from baseline")
    result = {}
    for index, (left, right) in enumerate(zip(candidate[1:], reference)):
        left, right = left.detach().float().cpu(), right.float()
        error = (left - right).square().mean()
        power = right.square().mean().clamp_min(torch.finfo(torch.float32).tiny)
        result[str(index)] = float(error / power)
    return result


def evaluate_variant(
    model,
    processor,
    samples: list[MultimodalSample],
    *,
    progress: StageProgress,
    variant: str,
    min_pixels: int,
    max_pixels: int,
    max_new_tokens: int,
    generate: bool,
    reference_dir: Path | None = None,
    write_reference: bool = False,
    capture_hidden: bool = False,
    teacher_timeout_seconds: float | None = None,
    generation_timeout_seconds: float | None = None,
) -> dict:
    if write_reference and reference_dir is None:
        raise ValueError("write_reference requires reference_dir")
    if reference_dir is not None:
        reference_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    for sample in samples:
        if progress.completed_sample(variant, sample.id) is not None:
            continue
        prompt, teacher, labels, positions = _teacher_inputs(
            processor, sample, min_pixels, max_pixels,
        )
        started = time.perf_counter()
        with torch.inference_mode():
            output = model(
                **teacher, use_cache=False, return_dict=True,
                output_hidden_states=capture_hidden,
            )
        teacher_seconds = time.perf_counter() - started
        if teacher_timeout_seconds is not None and teacher_seconds > teacher_timeout_seconds:
            raise TimeoutError(
                f"teacher-forced sample {sample.id} took {teacher_seconds:.1f}s "
                f"> {teacher_timeout_seconds:.1f}s"
            )
        metrics = _target_metrics(output.logits, labels)
        prediction_rows = torch.tensor([position - 1 for position in positions], dtype=torch.long)
        target_logits = output.logits[0, prediction_rows].detach().cpu()
        comparison, block_nmse = {}, {}
        reference = None
        ref_path = None if reference_dir is None else _reference_path(reference_dir, sample.id)
        if write_reference:
            reference = {
                "format_version": 1, "id": sample.id,
                "target_logits": target_logits.to(torch.float16),
                "hidden_states": (
                    [value.detach().cpu().to(torch.float16) for value in output.hidden_states[1:]]
                    if capture_hidden else []
                ),
            }
        elif ref_path is not None:
            if not ref_path.is_file():
                raise FileNotFoundError(f"missing BF16 reference artifact: {ref_path}")
            reference = torch.load(ref_path, map_location="cpu", weights_only=True)
            comparison = _compare_logits(target_logits, reference["target_logits"])
            if reference.get("hidden_states"):
                if not capture_hidden:
                    raise ValueError("hidden comparison requires capture_hidden=True")
                block_nmse = _hidden_nmse(output.hidden_states, reference["hidden_states"])

        prediction, generation_seconds = "", 0.0
        if generate:
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **prompt, do_sample=False, max_new_tokens=max_new_tokens, use_cache=True,
                )
            generation_seconds = time.perf_counter() - started
            if (
                generation_timeout_seconds is not None
                and generation_seconds > generation_timeout_seconds
            ):
                raise TimeoutError(
                    f"generation sample {sample.id} took {generation_seconds:.1f}s "
                    f"> {generation_timeout_seconds:.1f}s"
                )
            prompt_length = prompt["input_ids"].shape[-1]
            prediction = processor.batch_decode(
                generated[:, prompt_length:], skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
            if reference is not None and not write_reference and "prediction" in reference:
                comparison["output_consistency"] = float(prediction == reference["prediction"])
        score = _prediction_score(sample, prediction) if generate else math.nan
        row = {
            "id": sample.id, "dataset": sample.dataset, "target": sample.target,
            "prediction": prediction, "score": score, "target_text": metrics,
            "comparison_to_bf16": comparison, "block_nmse": block_nmse,
            "teacher_seconds": teacher_seconds, "generation_seconds": generation_seconds,
            "sequence_tokens": int(teacher["input_ids"].shape[-1]),
        }
        if write_reference:
            reference["prediction"] = prediction
            temporary = ref_path.with_suffix(".tmp")
            torch.save(reference, temporary)
            temporary.replace(ref_path)
        progress.complete_sample(variant, sample.id, row)
        print(
            f"[{progress.root.name}/{variant}] {sample.id} "
            f"teacher={teacher_seconds:.1f}s generation={generation_seconds:.1f}s",
            flush=True,
        )
        del output, target_logits, prompt, teacher, labels
    rows = progress.rows(variant)
    summary = aggregate_rows(rows)
    atomic_json(progress.root / variant / "metrics.json", summary)
    return summary


def aggregate_rows(rows: list[dict]) -> dict:
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
    datasets = {}
    for dataset, values in sorted(by_dataset.items()):
        scores = [float(row["score"]) for row in values if math.isfinite(float(row["score"]))]
        datasets[dataset] = {
            "samples": len(values),
            "score": 100.0 * sum(scores) / len(scores) if scores else math.nan,
            "nll": sum(row["target_text"]["total_nll"] for row in values)
            / sum(row["target_text"]["tokens"] for row in values),
        }
    tokens = sum(row["target_text"]["tokens"] for row in rows)
    total_nll = sum(row["target_text"]["total_nll"] for row in rows)
    nll = total_nll / tokens if tokens else math.nan
    comparisons = [row["comparison_to_bf16"] for row in rows]
    finite_scores = [value["score"] for value in datasets.values() if math.isfinite(value["score"])]
    return {
        "samples": len(rows), "datasets": datasets,
        "mean_score": sum(finite_scores) / len(finite_scores) if finite_scores else math.nan,
        "target_nll": nll, "target_ppl": math.exp(min(nll, 80.0)) if tokens else math.nan,
        "target_tokens": tokens,
        "mean_logits_kl": _mean_present(comparisons, "logits_kl"),
        "target_argmax_consistency": _mean_present(comparisons, "target_argmax_consistency"),
        "output_consistency": _mean_present(comparisons, "output_consistency"),
        "teacher_seconds": sum(row["teacher_seconds"] for row in rows),
        "generation_seconds": sum(row["generation_seconds"] for row in rows),
    }


def _mean_present(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row]
    return sum(values) / len(values) if values else math.nan


def screen_gate(candidate: dict, baseline: dict) -> dict:
    retentions = {}
    for dataset, base in baseline["datasets"].items():
        score = candidate["datasets"][dataset]["score"]
        if base["score"] == 0:
            # A ratio is undefined at a zero baseline.  Treat non-regression as
            # full retention so the gate remains finite and reproducible.
            retention = 1.0 if score >= 0 else 0.0
        else:
            retention = score / base["score"]
        retentions[dataset] = retention
    mean_retention = sum(retentions.values()) / len(retentions)
    nll_increase = candidate["target_nll"] / baseline["target_nll"] - 1.0
    finite = all(math.isfinite(value) for value in (
        candidate["target_nll"], candidate["target_ppl"], candidate["mean_score"],
    ))
    passed = (
        finite and mean_retention >= 0.90 and min(retentions.values()) >= 0.80
        and nll_increase <= 0.15
    )
    return {
        "passed": passed, "finite": finite, "retention": retentions,
        "mean_retention": mean_retention, "worst_retention": min(retentions.values()),
        "target_nll_relative_increase": nll_increase,
    }


def teacher_nll_only(model, processor, samples, *, min_pixels: int, max_pixels: int) -> dict:
    rows = []
    with torch.inference_mode():
        for sample in samples:
            _prompt, teacher, labels, _positions = _teacher_inputs(
                processor, sample, min_pixels, max_pixels,
            )
            output = model(**teacher, use_cache=False, return_dict=True)
            metric = _target_metrics(output.logits, labels)
            rows.append({"id": sample.id, "dataset": sample.dataset, **metric})
    total_tokens = sum(row["tokens"] for row in rows)
    total_nll = sum(row["total_nll"] for row in rows)
    by_source = {}
    for source in sorted({row["dataset"] for row in rows}):
        subset = [row for row in rows if row["dataset"] == source]
        by_source[source] = sum(row["total_nll"] for row in subset) / sum(
            row["tokens"] for row in subset
        )
    return {"nll": total_nll / total_tokens, "by_source": by_source, "rows": rows}
