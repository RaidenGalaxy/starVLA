"""Compare two frozen-inference stages with source-group paired bootstrap CIs."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .protocol import atomic_json, file_sha256


def _load(paths: list[Path]) -> list[dict[str, Any]]:
    by_sample: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status", "ok") != "ok":
                    continue
                previous = by_sample.get(row["sample_id"])
                if previous is not None and (
                    previous["prediction"] != row["prediction"]
                    or previous["question"]["answer"] != row["question"]["answer"]
                ):
                    raise ValueError(f"conflicting duplicate sample_id {row['sample_id']}")
                by_sample[row["sample_id"]] = row
    if not by_sample:
        raise ValueError(f"no successful rows in {paths}")
    return list(by_sample.values())


def _bootstrap(values: np.ndarray, samples: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    estimates = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "p_le_zero": float(np.mean(estimates <= 0.0)),
    }


def _load_metadata(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    metadata_by_sample: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = row["sample_id"]
            metadata = row.get("metadata", {})
            previous = metadata_by_sample.get(sample_id)
            if previous is not None and previous != metadata:
                raise ValueError(f"conflicting metadata for sample_id {sample_id}")
            metadata_by_sample[sample_id] = metadata
    return metadata_by_sample


def _group_summary(paired: list[dict[str, Any]], key: str) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        value = row.get(key)
        if value is not None:
            groups[str(value)].append(row)
    return {
        name: {
            "samples": len(rows),
            "reference_accuracy": float(np.mean([row["reference_score"] for row in rows])),
            "candidate_accuracy": float(np.mean([row["candidate_score"] for row in rows])),
            "reference_minus_candidate": float(np.mean([row["delta"] for row in rows])),
        }
        for name, rows in sorted(groups.items())
    }


def _distribution(values: list[Any]) -> dict[str, int]:
    return dict(sorted(Counter("<none>" if value is None else str(value) for value in values).items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional JSONL manifest supplying category/subclass/task_type metadata.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gate-pp", type=float, default=10.0)
    args = parser.parse_args()

    reference = _load(args.reference)
    candidate = _load(args.candidate)
    metadata_by_sample = _load_metadata(args.manifest)
    reference_by_sample = {row["sample_id"]: row for row in reference}
    candidate_by_sample = {row["sample_id"]: row for row in candidate}
    sample_ids = sorted(set(reference_by_sample) & set(candidate_by_sample))
    if not sample_ids:
        raise ValueError("reference and candidate contain no paired sample_id values")

    paired = []
    grouped_deltas: dict[str, list[float]] = defaultdict(list)
    answers = []
    for sample_id in sample_ids:
        ref = reference_by_sample[sample_id]
        cand = candidate_by_sample[sample_id]
        if ref["source_group"] != cand["source_group"]:
            raise ValueError(f"source_group mismatch for {sample_id}")
        if ref["question"]["answer"] != cand["question"]["answer"]:
            raise ValueError(f"answer mismatch for {sample_id}")
        ref_score = float(ref["prediction"]["score"])
        cand_score = float(cand["prediction"]["score"])
        delta = ref_score - cand_score
        grouped_deltas[ref["source_group"]].append(delta)
        answers.append(ref["question"]["answer"])
        metadata = metadata_by_sample.get(sample_id, {})
        paired.append(
            {
                "sample_id": sample_id,
                "source_group": ref["source_group"],
                "reference_score": ref_score,
                "candidate_score": cand_score,
                "delta": delta,
                "reference_answer": ref["prediction"].get("parsed_answer"),
                "candidate_answer": cand["prediction"].get("parsed_answer"),
                "gold_answer": ref["question"]["answer"],
                "category": metadata.get("category"),
                "subclass": metadata.get("subclass"),
                "task_type": metadata.get("task_type"),
            }
        )

    source_deltas = np.asarray(
        [np.mean(values) for _, values in sorted(grouped_deltas.items())], dtype=float
    )
    bootstrap = _bootstrap(source_deltas, args.bootstrap_samples, args.seed)
    reference_correct = [row for row in paired if row["reference_score"] == 1.0]
    reference_incorrect = [row for row in paired if row["reference_score"] == 0.0]
    forgetting_values = np.asarray(
        [float(row["candidate_score"] == 0.0) for row in reference_correct], dtype=float
    )
    forgetting = (
        _bootstrap(forgetting_values, args.bootstrap_samples, args.seed + 1)
        if len(forgetting_values)
        else None
    )
    reference_accuracy = float(np.mean([item["reference_score"] for item in paired]))
    candidate_accuracy = float(np.mean([item["candidate_score"] for item in paired]))
    majority_count = max(Counter(answers).values())
    gate = bootstrap["mean"] >= args.gate_pp / 100.0
    report = {
        "format_version": 3,
        "inputs": {
            "reference": [
                {"path": str(path.resolve()), "sha256": file_sha256(path)} for path in args.reference
            ],
            "candidate": [
                {"path": str(path.resolve()), "sha256": file_sha256(path)} for path in args.candidate
            ],
            "manifest": (
                {"path": str(args.manifest.resolve()), "sha256": file_sha256(args.manifest)}
                if args.manifest is not None
                else None
            ),
        },
        "paired_samples": len(paired),
        "source_groups": len(source_deltas),
        "reference_accuracy": reference_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "majority_answer_accuracy": majority_count / len(answers),
        "reference_minus_candidate": bootstrap,
        "degradation_gate_pp": args.gate_pp,
        "degradation_gate_passed": gate,
        "answer_stability": {
            "same_answer": sum(
                row["reference_answer"] == row["candidate_answer"] for row in paired
            ),
            "changed_answer": sum(
                row["reference_answer"] != row["candidate_answer"] for row in paired
            ),
            "agreement": float(
                np.mean(
                    [row["reference_answer"] == row["candidate_answer"] for row in paired]
                )
            ),
        },
        "exploratory_conditional_robustness": {
            "reference_correct_samples": len(reference_correct),
            "retained_correct": sum(row["candidate_score"] == 1.0 for row in reference_correct),
            "forgotten": sum(row["candidate_score"] == 0.0 for row in reference_correct),
            "forgetting_rate": forgetting,
            "reference_incorrect_samples": len(reference_incorrect),
            "candidate_only_gains": sum(
                row["candidate_score"] == 1.0 for row in reference_incorrect
            ),
            "candidate_only_gain_rate": (
                float(np.mean([row["candidate_score"] == 1.0 for row in reference_incorrect]))
                if reference_incorrect
                else None
            ),
        },
        "paired_outcomes": {
            "both_correct": sum(
                row["reference_score"] == 1.0 and row["candidate_score"] == 1.0 for row in paired
            ),
            "reference_only_correct": sum(
                row["reference_score"] == 1.0 and row["candidate_score"] == 0.0 for row in paired
            ),
            "candidate_only_correct": sum(
                row["reference_score"] == 0.0 and row["candidate_score"] == 1.0 for row in paired
            ),
            "both_incorrect": sum(
                row["reference_score"] == 0.0 and row["candidate_score"] == 0.0 for row in paired
            ),
        },
        "prediction_distribution": {
            "reference": _distribution([row["reference_answer"] for row in paired]),
            "candidate": _distribution([row["candidate_answer"] for row in paired]),
            "gold": _distribution([row["gold_answer"] for row in paired]),
        },
        "by_category": _group_summary(paired, "category"),
        "by_task_type": _group_summary(paired, "task_type"),
        "paired_rows": paired,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "analysis.json", report)
    lines = [
        "# Paired streaming stage-gap analysis",
        "",
        f"Paired samples/source groups: {len(paired)}/{len(source_deltas)}.",
        f"Reference accuracy: {100 * reference_accuracy:.2f}%.",
        f"Candidate accuracy: {100 * candidate_accuracy:.2f}%.",
        f"Majority-answer baseline: {100 * majority_count / len(answers):.2f}%.",
        (
            "Reference - candidate: "
            f"{100 * bootstrap['mean']:.2f} pp "
            f"(95% paired bootstrap CI {100 * bootstrap['ci95_low']:.2f}, "
            f"{100 * bootstrap['ci95_high']:.2f}); "
            f"{args.gate_pp:g} pp gate: {gate}."
        ),
        (
            "Paired outcomes (both correct / reference only / candidate only / both wrong): "
            f"{report['paired_outcomes']['both_correct']} / "
            f"{report['paired_outcomes']['reference_only_correct']} / "
            f"{report['paired_outcomes']['candidate_only_correct']} / "
            f"{report['paired_outcomes']['both_incorrect']}."
        ),
        (
            "Answer agreement: "
            f"{report['answer_stability']['same_answer']}/{len(paired)} "
            f"({100 * report['answer_stability']['agreement']:.2f}%)."
        ),
        "",
    ]
    if forgetting is not None:
        lines.extend(
            [
                "## Exploratory conditional robustness",
                "",
                (
                    f"Among {len(reference_correct)} reference-correct samples, "
                    f"{report['exploratory_conditional_robustness']['forgotten']} became incorrect: "
                    f"forgetting rate {100 * forgetting['mean']:.2f}% "
                    f"(95% bootstrap CI {100 * forgetting['ci95_low']:.2f}, "
                    f"{100 * forgetting['ci95_high']:.2f})."
                ),
                "This subset metric is exploratory and is not the pre-specified net-degradation gate.",
                "",
            ]
        )
    if report["by_category"]:
        lines.extend(
            [
                "## Category breakdown",
                "",
                "| Category | n | Reference | Candidate | Drop |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name, summary in report["by_category"].items():
            lines.append(
                f"| {name} | {summary['samples']} | "
                f"{100 * summary['reference_accuracy']:.2f}% | "
                f"{100 * summary['candidate_accuracy']:.2f}% | "
                f"{100 * summary['reference_minus_candidate']:.2f} pp |"
            )
        lines.append("")
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote paired analysis for {len(paired)} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
