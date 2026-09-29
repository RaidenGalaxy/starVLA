from __future__ import annotations

import math
import random
from collections import defaultdict


def _interval(values: list[float], samples: int) -> list[float]:
    values.sort()
    return [
        values[int(0.025 * samples)],
        values[min(samples - 1, int(0.975 * samples))],
    ]


def stratified_paired_bootstrap(
    baseline_rows: list[dict],
    candidate_rows: list[dict],
    *,
    samples: int = 2000,
    seed: int = 42,
) -> dict:
    """Bootstrap macro QA delta and token-pooled NLL changes by dataset."""
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    baseline = {row["id"]: row for row in baseline_rows}
    candidate = {row["id"]: row for row in candidate_rows}
    if set(baseline) != set(candidate):
        raise ValueError("paired bootstrap requires identical sample IDs")
    by_dataset: dict[str, list[str]] = defaultdict(list)
    for identifier, row in baseline.items():
        by_dataset[row["dataset"]].append(identifier)
    if not by_dataset:
        raise ValueError("paired bootstrap requires at least one sample")

    def summarize(selected: list[str]) -> tuple[float, float, float]:
        selected_by_dataset: dict[str, list[str]] = defaultdict(list)
        for identifier in selected:
            selected_by_dataset[baseline[identifier]["dataset"]].append(identifier)
        dataset_deltas = []
        base_total = cand_total = 0.0
        base_tokens = cand_tokens = 0
        for identifiers in selected_by_dataset.values():
            dataset_deltas.append(100.0 * sum(
                float(candidate[item]["score"]) - float(baseline[item]["score"])
                for item in identifiers
            ) / len(identifiers))
            for item in identifiers:
                left, right = baseline[item]["target_text"], candidate[item]["target_text"]
                base_total += float(left["total_nll"])
                cand_total += float(right["total_nll"])
                base_tokens += int(left["tokens"])
                cand_tokens += int(right["tokens"])
        base_nll = base_total / base_tokens
        cand_nll = cand_total / cand_tokens
        return sum(dataset_deltas) / len(dataset_deltas), cand_nll - base_nll, cand_nll / base_nll - 1.0

    ordered = [item for dataset in sorted(by_dataset) for item in by_dataset[dataset]]
    point = summarize(ordered)
    rng = random.Random(seed)
    qa_draws, nll_draws, relative_draws = [], [], []
    for _ in range(samples):
        selected = []
        for dataset in sorted(by_dataset):
            identifiers = by_dataset[dataset]
            selected.extend(rng.choice(identifiers) for _ in identifiers)
        qa, nll, relative = summarize(selected)
        qa_draws.append(qa)
        nll_draws.append(nll)
        relative_draws.append(relative)
    if not all(math.isfinite(value) for values in (qa_draws, nll_draws, relative_draws) for value in values):
        raise FloatingPointError("bootstrap produced a non-finite statistic")
    return {
        "bootstrap_samples": samples,
        "macro_score_delta_points": {"point": point[0], "ci95": _interval(qa_draws, samples)},
        "target_nll_delta": {"point": point[1], "ci95": _interval(nll_draws, samples)},
        "target_nll_relative_increase": {
            "point": point[2], "ci95": _interval(relative_draws, samples),
        },
    }

