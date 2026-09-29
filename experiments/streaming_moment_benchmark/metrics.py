"""Evidence, quality, degradation, and resource metrics."""
from __future__ import annotations

import math
import re
import string
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np

from .frames import FrameRef
from .protocol import EvidenceSpan, Question, StreamTrial
from .selection import SelectionResult


def _in_span(timestamp_sec: float, span: EvidenceSpan) -> bool:
    return span.start_sec - 1e-6 <= timestamp_sec <= span.end_sec + 1e-6


def refs_in_evidence(refs: Sequence[FrameRef], spans: Sequence[EvidenceSpan]) -> list[FrameRef]:
    return [ref for ref in refs if any(_in_span(ref.timestamp_sec, span) for span in spans)]


def span_recall(refs: Sequence[FrameRef], spans: Sequence[EvidenceSpan]) -> float | None:
    if not spans:
        return None
    return sum(any(_in_span(ref.timestamp_sec, span) for ref in refs) for span in spans) / len(spans)


def selection_metrics(
    trial: StreamTrial,
    result: SelectionResult,
    observed_refs: Sequence[FrameRef],
) -> dict[str, float | int | bool | None]:
    observed_evidence = refs_in_evidence(observed_refs, trial.evidence_spans)
    captured_evidence = refs_in_evidence(result.captured, trial.evidence_spans)
    final_evidence = refs_in_evidence(result.final_refs, trial.evidence_spans)
    denominator = len(observed_evidence)
    minimum = trial.metadata.get("min_required_frames")
    minimum = int(minimum) if minimum is not None else None
    required_coverage = (
        min(1.0, len(final_evidence) / minimum)
        if minimum is not None and minimum > 0
        else (float(bool(final_evidence)) if trial.evidence_spans else None)
    )
    capture_span = span_recall(result.captured, trial.evidence_spans)
    final_span = span_recall(result.final_refs, trial.evidence_spans)
    evidence_available = None
    if trial.evidence_spans:
        evidence_available = bool(final_span == 1.0 and (minimum is None or len(final_evidence) >= minimum))
    return {
        "frames_seen": result.frames_seen,
        "frames_admitted": result.frames_admitted,
        "frames_retained": len(result.final_refs),
        "retained_rgb_bytes": sum(item.image.nbytes for item in result.final_observations),
        "budget_respected": len(result.final_refs) <= result.memory_frames,
        "observed_evidence_frames": denominator,
        "captured_evidence_frames": len(captured_evidence),
        "retained_evidence_frames": len(final_evidence),
        "capture_frame_recall": (
            len({item.frame_id for item in captured_evidence}) / denominator if denominator else None
        ),
        "retention_frame_recall": len({item.frame_id for item in final_evidence}) / denominator if denominator else None,
        "capture_span_recall": capture_span,
        "retention_span_recall": final_span,
        "retention_loss": (
            max(0.0, float(capture_span) - float(final_span))
            if capture_span is not None and final_span is not None
            else None
        ),
        "required_frame_coverage": required_coverage,
        "evidence_available": evidence_available,
    }


def normalize_answer(value: str) -> str:
    text = value.lower().strip().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def score_answer(question: Question, prediction: str) -> dict[str, Any]:
    if question.answer_type == "multiple_choice":
        match = re.search(r"(?:^|\b)([A-E])(?:\b|$)", prediction.upper())
        choice = match.group(1) if match else ""
        if not choice:
            normalized = normalize_answer(prediction)
            for key, text in question.choices.items():
                if normalize_answer(text) == normalized:
                    choice = key
                    break
        return {"parsed_answer": choice, "score": float(choice == question.answer), "metric": "mcq_accuracy"}
    references = [question.answer]
    if question.answer_text and question.answer_text not in references:
        references.append(question.answer_text)
    score = max(float(normalize_answer(prediction) == normalize_answer(reference)) for reference in references)
    return {"parsed_answer": prediction.strip(), "score": score, "metric": "normalized_exact_match"}


def _mean(values: list[float | int | bool | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else None


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], int(row["memory_frames"]), float(row["encode_fps"]))].append(row)
    configurations = []
    for (method, memory_frames, encode_fps), items in sorted(groups.items()):
        delays = []
        scores = []
        condition_scores: dict[str, list[float]] = defaultdict(list)
        for item in items:
            delay = item.get("query_delay_sec")
            score = item.get("prediction", {}).get("score")
            if delay is not None and score is not None:
                delays.append(float(delay))
                scores.append(float(score))
                condition = item.get("condition", {})
                key = f"delay={condition.get('delay_sec')}|profile={condition.get('distractor_profile')}"
                condition_scores[key].append(float(score))
        slope = None
        if len(delays) >= 2 and len(set(delays)) >= 2:
            slope = float(np.polyfit(np.asarray(delays), np.asarray(scores), 1)[0] * 60.0)
            if abs(slope) < 1e-12:
                slope = 0.0
        worst_group = None
        if condition_scores:
            key, values = min(condition_scores.items(), key=lambda pair: sum(pair[1]) / len(pair[1]))
            worst_group = {"condition": key, "accuracy": sum(values) / len(values), "samples": len(values)}
        configurations.append(
            {
                "method": method,
                "memory_frames": memory_frames,
                "encode_fps": encode_fps,
                "samples": len(items),
                "accuracy": _mean([item.get("prediction", {}).get("score") for item in items]),
                "capture_span_recall": _mean([item["metrics"].get("capture_span_recall") for item in items]),
                "retention_span_recall": _mean([item["metrics"].get("retention_span_recall") for item in items]),
                "required_frame_coverage": _mean([item["metrics"].get("required_frame_coverage") for item in items]),
                "retention_loss": _mean([item["metrics"].get("retention_loss") for item in items]),
                "retained_frames": _mean([item["metrics"].get("frames_retained") for item in items]),
                "degradation_per_delay_minute": slope,
                "worst_group": worst_group,
                "budget_violations": sum(not item["metrics"]["budget_respected"] for item in items),
            }
        )
    by_method: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for item in configurations:
        if item["accuracy"] is not None:
            by_method[(item["method"], item["encode_fps"])].append(item)
    budget_auc = []
    for (method, encode_fps), items in sorted(by_method.items()):
        points = sorted((item["memory_frames"], item["accuracy"]) for item in items)
        auc = None
        if len(points) >= 2 and points[-1][0] > points[0][0]:
            x = np.asarray([item[0] for item in points], dtype=float)
            y = np.asarray([item[1] for item in points], dtype=float)
            auc = float(np.trapz(y, x) / (x[-1] - x[0]))
        budget_auc.append({"method": method, "encode_fps": encode_fps, "accuracy_budget_auc": auc, "points": points})
    unique_trials: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        unique_trials.setdefault(str(row.get("trial_id", f"row-{index}")), row)
    oracle_values = []
    for row in unique_trials.values():
        stage = row.get("stage_predictions", {}).get("critical_only", {})
        value = stage.get("score") if stage else row.get("oracle_ladder", {}).get("critical_only")
        if value is not None:
            oracle_values.append(float(value))
    oracle_accuracy = _mean(oracle_values)
    baseline_configs = [
        item for item in configurations if item["method"] != "future_utility" and item["accuracy"] is not None
    ]
    best_baseline = max((item["accuracy"] for item in baseline_configs), default=None)
    gap_pp = (
        100.0 * (float(oracle_accuracy) - float(best_baseline))
        if oracle_accuracy is not None and best_baseline is not None
        else None
    )
    matched_gains = []
    for candidate in configurations:
        if candidate["method"] != "future_utility" or candidate["accuracy"] is None:
            continue
        matches = [
            item
            for item in baseline_configs
            if item["memory_frames"] == candidate["memory_frames"] and item["encode_fps"] == candidate["encode_fps"]
        ]
        if matches:
            matched_gains.append(100.0 * (candidate["accuracy"] - max(item["accuracy"] for item in matches)))
    utility_gain_pp = max(matched_gains, default=None)
    gates = {
        "critical_oracle_accuracy": oracle_accuracy,
        "best_fixed_budget_baseline_accuracy": best_baseline,
        "oracle_gap_pp": gap_pp,
        "failure_exists_gate_10pp": None if gap_pp is None else gap_pp >= 10.0,
        "best_matched_future_utility_gain_pp": utility_gain_pp,
        "method_gain_gate_5pp": None if utility_gain_pp is None else utility_gain_pp >= 5.0,
        "all_budgets_respected": all(item["budget_violations"] == 0 for item in configurations),
        "standard_benchmark_regression_gate": "not_evaluated",
    }
    return {
        "format_version": 1,
        "configurations": configurations,
        "accuracy_budget_curves": budget_auc,
        "research_gates": gates,
    }
