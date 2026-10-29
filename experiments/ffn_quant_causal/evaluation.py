from __future__ import annotations

import math
import time
from collections import defaultdict
from pathlib import Path

import torch

from experiments.infinitevl_quant.evaluation import (
    _compare_logits,
    _target_metrics,
    _teacher_inputs,
)
from experiments.infinitevl_quant.progress import StageProgress, atomic_json, safe_id
from experiments.infinitevl_quant_ffn_bf16_4x64.scoring import (
    SCORER_ID,
    score_prediction,
)
from experiments.vlm_quant_benchmark.data import MultimodalSample

from .budget import WallclockBudget


def _reference_path(root: Path, sample_id: str) -> Path:
    return root / f"{safe_id(sample_id)}.pt"


def evaluate_variant(
    model,
    processor,
    samples: list[MultimodalSample],
    *,
    progress: StageProgress,
    variant: str,
    reference_dir: Path,
    write_reference: bool,
    min_pixels: int,
    max_pixels: int,
    max_new_tokens: int,
    generate: bool,
    wallclock_budget: WallclockBudget | None = None,
    teacher_timeout_seconds: float = 600.0,
    generation_timeout_seconds: float = 1800.0,
) -> dict:
    """Evaluate either supported model with the corrected deterministic scorer.

    Reference target logits are stored as FP32.  This deliberately does not
    reuse the older FP16 reference caches, because causal KL comparisons must
    not include cache-rounding error.
    """
    expected_ids = [sample.id for sample in samples]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError(f"{variant} input samples contain duplicate IDs")
    reference_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    if wallclock_budget is not None:
        wallclock_budget.check(
            stage=progress.root.name,
            boundary="before_variant",
            variant=variant,
        )
    for sample in samples:
        reference_path = _reference_path(reference_dir, sample.id)
        completed = progress.completed_sample(variant, sample.id)
        # A crash can occur after the JSON row is committed but before a
        # reference tensor (or vice versa).  A baseline row is resumable only
        # when both artifacts exist; candidate rows need no reference write.
        if completed is not None and (not write_reference or reference_path.is_file()):
            continue
        if wallclock_budget is not None:
            wallclock_budget.check(
                stage=progress.root.name,
                boundary="before_sample",
                variant=variant,
                sample_id=sample.id,
            )
        prompt, teacher, labels, positions = _teacher_inputs(
            processor, sample, min_pixels, max_pixels,
        )
        started = time.perf_counter()
        with torch.inference_mode():
            output = model(**teacher, use_cache=False, return_dict=True)
        teacher_seconds = time.perf_counter() - started
        if teacher_seconds > teacher_timeout_seconds:
            raise TimeoutError(
                f"teacher-forced sample {sample.id} completed in {teacher_seconds:.1f}s, "
                f"exceeding the post-call threshold {teacher_timeout_seconds:.1f}s; "
                "the in-flight Python call was not hard-interrupted"
            )
        target = _target_metrics(output.logits, labels)
        prediction_rows = torch.tensor(
            [position - 1 for position in positions], dtype=torch.long,
        )
        target_logits = output.logits[0, prediction_rows].detach().float().cpu()
        comparison: dict[str, float] = {}
        if write_reference:
            temporary = reference_path.with_suffix(reference_path.suffix + ".tmp")
            torch.save({
                "format_version": 2,
                "id": sample.id,
                "target_logits": target_logits,
            }, temporary)
            temporary.replace(reference_path)
        else:
            if not reference_path.is_file():
                raise FileNotFoundError(f"missing FP32 reference artifact: {reference_path}")
            reference = torch.load(reference_path, map_location="cpu", weights_only=True)
            if reference.get("format_version") != 2 or reference.get("id") != sample.id:
                raise RuntimeError(f"invalid reference artifact: {reference_path}")
            comparison = _compare_logits(target_logits, reference["target_logits"])

        prediction = ""
        generation_seconds = 0.0
        generation_tokens = 0
        truncated = False
        if generate:
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **prompt,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                )
            generation_seconds = time.perf_counter() - started
            if generation_seconds > generation_timeout_seconds:
                raise TimeoutError(
                    f"generation sample {sample.id} completed in {generation_seconds:.1f}s, "
                    f"exceeding the post-call threshold {generation_timeout_seconds:.1f}s; "
                    "the in-flight Python call was not hard-interrupted"
                )
            prompt_length = int(prompt["input_ids"].shape[-1])
            new_ids = generated[:, prompt_length:]
            generation_tokens = int(new_ids.shape[-1])
            truncated = generation_tokens >= max_new_tokens
            prediction = processor.batch_decode(
                new_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()
        scored = score_prediction(sample, prediction) if generate else None
        row = {
            "id": sample.id,
            "dataset": sample.dataset,
            "target": sample.target,
            "prediction": prediction,
            "score": math.nan if scored is None else float(scored.score),
            "scoring": None if scored is None else {
                "scorer": SCORER_ID,
                "rule": scored.rule,
                "extracted": scored.extracted,
            },
            "target_text": target,
            "comparison_to_bf16": comparison,
            "teacher_seconds": teacher_seconds,
            "generation_seconds": generation_seconds,
            "generation_tokens": generation_tokens,
            "generation_truncated": truncated,
            "sequence_tokens": int(teacher["input_ids"].shape[-1]),
            "wallclock_enforcement": (
                "cooperative_before_and_after_sample_no_hard_interrupt"
            ),
        }
        progress.complete_sample(variant, sample.id, row)
        print(
            f"[{progress.root.name}/{variant}] {sample.id} "
            f"teacher={teacher_seconds:.1f}s generation={generation_seconds:.1f}s",
            flush=True,
        )
        del output, target_logits, prompt, teacher, labels
        if wallclock_budget is not None:
            # The row above is already atomically committed.  If the deadline
            # passed during teacher forcing or generation, stopping here keeps
            # that completed sample and starts no further model call.
            wallclock_budget.check(
                stage=progress.root.name,
                boundary="after_sample_committed",
                variant=variant,
                sample_id=sample.id,
            )
    rows = validate_variant_rows(progress.rows(variant), samples, variant=variant)
    summary = aggregate_rows(rows)
    atomic_json(progress.root / variant / "metrics.json", summary)
    return summary


def validate_variant_rows(
    rows: list[dict],
    samples: list[MultimodalSample],
    *,
    variant: str,
) -> list[dict]:
    """Reject missing, duplicate, or foreign rows before any aggregation."""

    expected_ids = [sample.id for sample in samples]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError(f"{variant} expected sample IDs are not unique")
    expected = {sample.id: sample.dataset for sample in samples}
    actual_ids = [row.get("id") for row in rows]
    if any(not isinstance(identifier, str) or not identifier for identifier in actual_ids):
        raise RuntimeError(f"{variant} contains a row without a valid sample ID")
    if len(actual_ids) != len(set(actual_ids)):
        raise RuntimeError(f"{variant} contains duplicate completed sample rows")
    actual = set(actual_ids)
    missing, foreign = set(expected) - actual, actual - set(expected)
    if missing or foreign or len(rows) != len(samples):
        raise RuntimeError(
            f"{variant} completed sample set mismatch: "
            f"missing={sorted(missing)[:8]}, foreign={sorted(foreign)[:8]}, "
            f"rows={len(rows)}, expected={len(samples)}"
        )
    mismatched = [
        str(row["id"])
        for row in rows
        if row.get("dataset") != expected[str(row["id"])]
    ]
    if mismatched:
        raise RuntimeError(
            f"{variant} contains rows with mismatched datasets: {mismatched[:8]}"
        )
    return rows


def aggregate_rows(rows: list[dict]) -> dict:
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
    datasets = {}
    for dataset, values in sorted(by_dataset.items()):
        scores = [float(row["score"]) for row in values if math.isfinite(float(row["score"]))]
        tokens = sum(int(row["target_text"]["tokens"]) for row in values)
        total_nll = sum(float(row["target_text"]["total_nll"]) for row in values)
        datasets[dataset] = {
            "samples": len(values),
            "score": 100.0 * sum(scores) / len(scores) if scores else math.nan,
            "target_nll_token_weighted": total_nll / tokens if tokens else math.nan,
            "target_nll_sample_mean": sum(
                float(row["target_text"]["nll"]) for row in values
            ) / len(values),
        }
    tokens = sum(int(row["target_text"]["tokens"]) for row in rows)
    total_nll = sum(float(row["target_text"]["total_nll"]) for row in rows)
    sample_nll = [float(row["target_text"]["nll"]) for row in rows]
    comparisons = [row.get("comparison_to_bf16", {}) for row in rows]
    finite_scores = [value["score"] for value in datasets.values() if math.isfinite(value["score"])]

    def mean_present(key: str) -> float:
        values = [float(row[key]) for row in comparisons if key in row]
        return sum(values) / len(values) if values else math.nan

    token_nll = total_nll / tokens if tokens else math.nan
    return {
        "samples": len(rows),
        "datasets": datasets,
        "mean_score": sum(finite_scores) / len(finite_scores) if finite_scores else math.nan,
        "target_nll": token_nll,
        "target_nll_sample_mean": sum(sample_nll) / len(sample_nll) if sample_nll else math.nan,
        "target_ppl": math.exp(min(token_nll, 80.0)) if tokens else math.nan,
        "target_tokens": tokens,
        "mean_logits_kl": mean_present("logits_kl"),
        "target_argmax_consistency": mean_present("target_argmax_consistency"),
        "truncated_samples": sum(bool(row.get("generation_truncated")) for row in rows),
        "teacher_seconds": sum(float(row["teacher_seconds"]) for row in rows),
        "generation_seconds": sum(float(row["generation_seconds"]) for row in rows),
        "scorer": SCORER_ID,
    }
