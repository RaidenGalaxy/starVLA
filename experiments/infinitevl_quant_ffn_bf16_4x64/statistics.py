from __future__ import annotations

import math
import random
from collections import defaultdict


def _percentile(values: list[float], probability: float) -> float:
    values.sort()
    index = min(len(values) - 1, max(0, int(probability * len(values))))
    return values[index]


def _summarize(
    selected: list[str], baseline: dict[str, dict], candidate: dict[str, dict],
) -> dict:
    by_dataset: dict[str, list[str]] = defaultdict(list)
    for identifier in selected:
        by_dataset[baseline[identifier]["dataset"]].append(identifier)
    base_scores, candidate_scores, retentions = {}, {}, {}
    base_total = candidate_total = 0.0
    base_tokens = candidate_tokens = 0
    for dataset, identifiers in by_dataset.items():
        base_score = 100.0 * sum(float(baseline[item]["score"]) for item in identifiers) / len(identifiers)
        candidate_score = 100.0 * sum(float(candidate[item]["score"]) for item in identifiers) / len(identifiers)
        base_scores[dataset], candidate_scores[dataset] = base_score, candidate_score
        retentions[dataset] = (
            1.0 if base_score == 0 and candidate_score >= 0
            else 0.0 if base_score == 0
            else candidate_score / base_score
        )
        for item in identifiers:
            left, right = baseline[item]["target_text"], candidate[item]["target_text"]
            base_total += float(left["total_nll"])
            candidate_total += float(right["total_nll"])
            base_tokens += int(left["tokens"])
            candidate_tokens += int(right["tokens"])
    base_nll = base_total / base_tokens
    candidate_nll = candidate_total / candidate_tokens
    macro_delta = sum(
        candidate_scores[key] - base_scores[key] for key in by_dataset
    ) / len(by_dataset)
    nll_relative = candidate_nll / base_nll - 1.0
    passed = (
        sum(retentions.values()) / len(retentions) >= 0.90
        and min(retentions.values()) >= 0.80
        and nll_relative <= 0.15
    )
    return {
        "macro_delta": macro_delta,
        "nll_delta": candidate_nll - base_nll,
        "nll_relative": nll_relative,
        "mean_retention": sum(retentions.values()) / len(retentions),
        "worst_retention": min(retentions.values()),
        "passed": passed,
    }


def stratified_gate_bootstrap(
    baseline_rows: list[dict],
    candidate_rows: list[dict],
    *,
    samples: int,
    seed: int,
) -> dict:
    baseline = {row["id"]: row for row in baseline_rows}
    candidate = {row["id"]: row for row in candidate_rows}
    if set(baseline) != set(candidate):
        raise ValueError("paired bootstrap requires identical sample IDs")
    by_dataset: dict[str, list[str]] = defaultdict(list)
    for identifier, row in baseline.items():
        by_dataset[row["dataset"]].append(identifier)
    ordered = [item for dataset in sorted(by_dataset) for item in by_dataset[dataset]]
    point = _summarize(ordered, baseline, candidate)
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        selected = []
        for dataset in sorted(by_dataset):
            identifiers = by_dataset[dataset]
            selected.extend(rng.choice(identifiers) for _ in identifiers)
        draws.append(_summarize(selected, baseline, candidate))
    if not all(
        math.isfinite(float(row[key]))
        for row in draws
        for key in ("macro_delta", "nll_delta", "nll_relative")
    ):
        raise FloatingPointError("bootstrap produced non-finite statistics")

    def interval(key: str) -> list[float]:
        values = [float(row[key]) for row in draws]
        return [_percentile(values, 0.025), _percentile(values, 0.975)]

    return {
        "bootstrap_samples": samples,
        "point": point,
        "macro_score_delta_points_ci95": interval("macro_delta"),
        "target_nll_delta_ci95": interval("nll_delta"),
        "target_nll_relative_increase_ci95": interval("nll_relative"),
        "gate_pass_probability": sum(bool(row["passed"]) for row in draws) / samples,
    }


def exact_mcnemar(baseline_rows: list[dict], candidate_rows: list[dict]) -> dict[str, dict]:
    baseline = {row["id"]: row for row in baseline_rows}
    result = {}
    for dataset in ("MMMU_DEV_VAL", "ScienceQA_TEST"):
        rows = [row for row in candidate_rows if row["dataset"] == dataset]
        improved = sum(baseline[row["id"]]["score"] == 0 and row["score"] == 1 for row in rows)
        regressed = sum(baseline[row["id"]]["score"] == 1 and row["score"] == 0 for row in rows)
        discordant = improved + regressed
        if discordant:
            tail = sum(math.comb(discordant, k) for k in range(min(improved, regressed) + 1)) / (2 ** discordant)
            p_value = min(1.0, 2.0 * tail)
        else:
            p_value = 1.0
        result[dataset] = {
            "improved": improved,
            "regressed": regressed,
            "discordant": discordant,
            "two_sided_p": p_value,
        }
    return result
