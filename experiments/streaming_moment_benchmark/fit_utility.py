"""Fit the query-agnostic utility scorer from train-only frame labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .protocol import file_sha256
from .utility import fit_ridge_utility, save_utility_model


def grouped_cv_alpha(
    features: np.ndarray,
    targets: np.ndarray,
    groups: list[str],
    feature_names: list[str],
    *,
    alphas: list[float],
    folds: int = 4,
    seed: int = 42,
) -> tuple[float, list[dict[str, float | int]]]:
    unique = sorted(set(groups), key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest())
    folds = min(folds, len(unique))
    if folds < 2:
        return alphas[0], [{"alpha": alpha, "folds": 0, "rmse": float("nan")} for alpha in alphas]
    assignment = {group: index % folds for index, group in enumerate(unique)}
    group_array = np.asarray(groups)
    report: list[dict[str, float | int]] = []
    for alpha in alphas:
        squared_error = 0.0
        examples = 0
        completed_folds = 0
        for fold in range(folds):
            validation = np.asarray([assignment[group] == fold for group in group_array], dtype=bool)
            training = ~validation
            if validation.sum() == 0 or training.sum() < 2:
                continue
            model = fit_ridge_utility(
                features[training],
                targets[training],
                feature_names,
                alpha=alpha,
                metadata={"cv_fold": fold},
            )
            residual = model.predict(features[validation]) - targets[validation]
            squared_error += float(np.square(residual).sum())
            examples += int(validation.sum())
            completed_folds += 1
        rmse = float(np.sqrt(squared_error / examples)) if examples else float("inf")
        report.append({"alpha": alpha, "folds": completed_folds, "examples": examples, "rmse": rmse})
    best = min(report, key=lambda row: (float(row["rmse"]), float(row["alpha"])))
    return float(best["alpha"]), report


def _counterfactual_labels(path: Path) -> dict[tuple[str, str], float]:
    labels: dict[tuple[str, str], float] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["trial_id"]), str(row["frame_id"]))
            if "utility" in row:
                utility = float(row["utility"])
            else:
                utility = max(0.0, float(row["full_correct_logprob"]) - float(row["dropped_correct_logprob"]))
            if key in labels:
                raise ValueError(f"duplicate counterfactual label at line {line_number}: {key}")
            labels[key] = utility
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label-source", choices=["annotated", "counterfactual"], default="counterfactual")
    parser.add_argument("--counterfactual-labels", type=Path)
    parser.add_argument("--train-splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    parser.add_argument("--cv-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.label_source == "counterfactual" and args.counterfactual_labels is None:
        parser.error("--counterfactual-labels is required for counterfactual labels")
    counterfactual = _counterfactual_labels(args.counterfactual_labels) if args.counterfactual_labels else {}
    features: list[list[float]] = []
    targets: list[float] = []
    feature_names: list[str] | None = None
    source_groups: set[str] = set()
    row_groups: list[str] = []
    used_splits: set[str] = set()
    skipped_missing_labels = 0
    for feature_path in args.features:
        with feature_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row: dict[str, Any] = json.loads(line)
                if row["split"] not in args.train_splits:
                    continue
                names = [str(item) for item in row["feature_names"]]
                if feature_names is None:
                    feature_names = names
                elif names != feature_names:
                    raise ValueError("feature schema changes within the input")
                if args.label_source == "annotated":
                    target = float(bool(row["evidence_overlap"]))
                else:
                    key = (str(row["trial_id"]), str(row["frame_id"]))
                    if key not in counterfactual:
                        skipped_missing_labels += 1
                        continue
                    target = counterfactual[key]
                group = str(row["source_group"])
                features.append([float(value) for value in row["features"]])
                targets.append(target)
                row_groups.append(group)
                source_groups.add(group)
                used_splits.add(str(row["split"]))
    if feature_names is None or len(features) < 2:
        raise ValueError("no train/validation feature rows with usable labels")
    feature_array = np.asarray(features)
    target_array = np.asarray(targets)
    if args.alpha is None:
        if any(alpha < 0 for alpha in args.alphas) or args.cv_folds < 2:
            parser.error("CV alphas must be non-negative and cv-folds must be at least 2")
        alpha, cv_report = grouped_cv_alpha(
            feature_array,
            target_array,
            row_groups,
            feature_names,
            alphas=args.alphas,
            folds=args.cv_folds,
            seed=args.seed,
        )
    else:
        alpha = args.alpha
        cv_report = []
    model = fit_ridge_utility(
        feature_array,
        target_array,
        feature_names,
        alpha=alpha,
        metadata={
            "label_source": args.label_source,
            "train_splits": sorted(used_splits),
            "source_groups": len(source_groups),
            "feature_inputs": [
                {"path": str(path.resolve()), "sha256": file_sha256(path)} for path in args.features
            ],
            "counterfactual_labels_sha256": (
                file_sha256(args.counterfactual_labels) if args.counterfactual_labels else None
            ),
            "skipped_missing_labels": skipped_missing_labels,
            "grouped_cv": cv_report,
            "selected_alpha": alpha,
            "cv_folds_requested": args.cv_folds,
            "cv_seed": args.seed,
            "runtime_query_visible": False,
        },
    )
    save_utility_model(args.output, model)
    print(f"wrote utility model fitted on {len(features)} rows from {len(source_groups)} source groups to {args.output}")


if __name__ == "__main__":
    main()
