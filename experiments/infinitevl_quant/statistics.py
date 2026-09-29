from __future__ import annotations

import random
from collections import defaultdict


def paired_bootstrap(
    baseline_rows: list[dict],
    candidate_rows: list[dict],
    *,
    samples: int = 2000,
    seed: int = 42,
) -> dict:
    baseline = {row["id"]: row for row in baseline_rows}
    candidate = {row["id"]: row for row in candidate_rows}
    if set(baseline) != set(candidate):
        raise ValueError("paired bootstrap requires identical sample IDs")
    by_dataset = defaultdict(list)
    for identifier, row in baseline.items():
        by_dataset[row["dataset"]].append(identifier)
    rng = random.Random(seed)
    result = {}
    for dataset, identifiers in sorted(by_dataset.items()):
        point = 100.0 * sum(
            candidate[item]["score"] - baseline[item]["score"] for item in identifiers
        ) / len(identifiers)
        draws = []
        for _ in range(samples):
            selected = [rng.choice(identifiers) for _ in identifiers]
            draws.append(100.0 * sum(
                candidate[item]["score"] - baseline[item]["score"] for item in selected
            ) / len(selected))
        draws.sort()
        result[dataset] = {
            "delta_points": point,
            "ci95": [draws[int(0.025 * samples)], draws[min(samples - 1, int(0.975 * samples))]],
            "bootstrap_samples": samples,
        }
    return result


def prediction_transitions(baseline_rows: list[dict], candidate_rows: list[dict]) -> dict:
    baseline = {row["id"]: row for row in baseline_rows}
    counts = {"improved": 0, "regressed": 0, "unchanged": 0, "prediction_changed": 0}
    rows = []
    for candidate in candidate_rows:
        reference = baseline[candidate["id"]]
        delta = candidate["score"] - reference["score"]
        status = "improved" if delta > 0 else "regressed" if delta < 0 else "unchanged"
        counts[status] += 1
        counts["prediction_changed"] += int(candidate["prediction"] != reference["prediction"])
        rows.append({
            "id": candidate["id"], "dataset": candidate["dataset"], "status": status,
            "score_delta": delta, "baseline": reference["prediction"],
            "candidate": candidate["prediction"],
        })
    return {"counts": counts, "rows": rows}


def retention(candidate: dict, baseline: dict) -> dict:
    values = {}
    for dataset, base in baseline["datasets"].items():
        score = candidate["datasets"][dataset]["score"]
        values[dataset] = (1.0 if score >= 0 else 0.0) if base["score"] == 0 else score / base["score"]
    return {
        "by_dataset": values,
        "mean": sum(values.values()) / len(values),
        "worst": min(values.values()),
    }
