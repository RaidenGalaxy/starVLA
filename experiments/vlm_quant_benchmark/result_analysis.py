from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .metrics import bootstrap_ci
from .reporting import atomic_json


def _sample_map(rows: list[dict], score_field: str) -> dict[str, tuple[str, float]]:
    mapped = {}
    for row in rows:
        identifier = str(row["id"])
        if identifier in mapped:
            raise ValueError(f"duplicate sample id: {identifier}")
        score = row.get(score_field)
        if score is None or not math.isfinite(float(score)):
            raise ValueError(f"missing/non-finite {score_field} for {identifier}")
        mapped[identifier] = (str(row["dataset"]), float(score))
    return mapped


def paired_score_difference(
    candidate_rows: list[dict],
    reference_rows: list[dict],
    *,
    score_field: str = "official_score",
    bootstrap_samples: int = 10_000,
    seed: int = 42,
) -> dict:
    """Paired higher-is-better score delta with a deterministic percentile CI."""
    candidate = _sample_map(candidate_rows, score_field)
    reference = _sample_map(reference_rows, score_field)
    if candidate.keys() != reference.keys():
        missing = sorted(reference.keys() - candidate.keys())
        extra = sorted(candidate.keys() - reference.keys())
        raise ValueError(f"paired sample mismatch; missing={missing[:8]}, extra={extra[:8]}")
    deltas = []
    for identifier in sorted(reference):
        candidate_dataset, candidate_score = candidate[identifier]
        reference_dataset, reference_score = reference[identifier]
        if candidate_dataset != reference_dataset:
            raise ValueError(f"dataset mismatch for {identifier}")
        deltas.append(candidate_score - reference_score)
    lower, upper = bootstrap_ci(deltas, samples=bootstrap_samples, seed=seed)
    mean = sum(deltas) / len(deltas)
    conclusion = (
        "improved"
        if lower > 0
        else "regressed"
        if upper < 0
        else "indistinguishable_or_uncertain"
    )
    return {
        "paired_samples": len(deltas),
        "mean_score_delta": mean,
        "bootstrap_95_ci": [lower, upper],
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "conclusion": conclusion,
    }


def summarize_candidate(
    candidate_rows: list[dict],
    reference_rows: list[dict],
    *,
    score_field: str = "official_score",
    seed: int = 42,
) -> dict:
    datasets = sorted({str(row["dataset"]) for row in reference_rows})
    summaries = {}
    retentions = []
    for dataset in datasets:
        candidate = [row for row in candidate_rows if str(row["dataset"]) == dataset]
        reference = [row for row in reference_rows if str(row["dataset"]) == dataset]
        paired = paired_score_difference(
            candidate, reference, score_field=score_field, seed=seed,
        )
        candidate_mean = sum(float(row[score_field]) for row in candidate) / len(candidate)
        reference_mean = sum(float(row[score_field]) for row in reference) / len(reference)
        retention = candidate_mean / reference_mean if reference_mean != 0 else math.nan
        if math.isfinite(retention):
            retentions.append(retention)
        summaries[dataset] = paired | {
            "candidate_score": candidate_mean,
            "reference_score": reference_mean,
            "retention": retention,
        }
    return {
        "datasets": summaries,
        "worst_retention": min(retentions) if retentions else math.nan,
        "macro_retention": sum(retentions) / len(retentions) if retentions else math.nan,
    }


def select_best_local_w4(
    candidates: list[dict],
    reference_rows: list[dict],
    *,
    score_field: str = "official_score",
    seed: int = 42,
) -> dict:
    """Apply the fixed worst-retention, macro, BPW, decoder-scope tie break."""
    ranked = []
    for candidate in candidates:
        summary = summarize_candidate(
            candidate["samples"], reference_rows, score_field=score_field, seed=seed,
        )
        row = {
            "run_id": candidate["run_id"],
            "method": candidate["method"],
            "scheme_id": candidate["scheme_id"],
            "scope": candidate["scope"],
            "weight_bits": candidate.get("weight_bits", 4),
            "activation_bits": candidate.get("activation_bits"),
            "implementation_tier": candidate["implementation_tier"],
            "logical_bits_per_weight": float(candidate["logical_bits_per_weight"]),
            **summary,
        }
        ranked.append(row)
    ranked.sort(key=lambda row: (
        -row["worst_retention"],
        -row["macro_retention"],
        row["logical_bits_per_weight"],
        0 if row["scope"] == "decoder_canonical" else 1,
        row["run_id"],
    ))
    if not ranked:
        raise ValueError("no local W4 candidates")
    best = ranked[0]
    return {
        "method": best["method"],
        "scheme_id": best["scheme_id"],
        "scope": best["scope"],
        "weight_bits": best["weight_bits"],
        "activation_bits": best["activation_bits"],
        "implementation_tier": best["implementation_tier"],
        "logical_bits_per_weight": best["logical_bits_per_weight"],
        "selection_rule": [
            "maximum worst per-dataset retention",
            "maximum macro retention",
            "minimum logical bits per weight",
            "decoder_canonical scope",
        ],
        "best_run_id": ranked[0]["run_id"],
        "ranking": ranked,
    }


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate paired VLM screen results")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    reference = _read_json(args.reference)
    candidates = [_read_json(path) for path in args.candidate]
    payload = select_best_local_w4(
        candidates,
        reference["samples"],
        seed=args.seed,
    )
    atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
