"""Preregistered selection logic for formal W4 promotion."""
from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict

import numpy as np


PROMOTION_CANDIDATES = {
    "mbq_no_reweight",
    "mbq_sample_average",
    "mbq_modality_normalized",
    "mbq_no_median_floor",
    "squeezellm_no_sparse",
}


def rank_promotion_variants(
    rows: list[dict], *, bootstrap_samples: int = 2000, seed: int = 42,
) -> list[dict]:
    """Rank by dataset-balanced paired NLL improvement over MBQ current."""
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    baseline = {
        (str(row["dataset"]), str(row["id"])): float(row["target_nll"])
        for row in rows
        if row.get("variant") == "mbq_current"
        and row.get("evaluation_split") == "ablation"
    }
    grouped: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    for row in rows:
        variant = str(row.get("variant", ""))
        key = (str(row.get("dataset")), str(row.get("id")))
        if (
            variant in PROMOTION_CANDIDATES
            and row.get("evaluation_split") == "ablation"
            and key in baseline
        ):
            grouped[variant].append((key[0], key[1], baseline[key] - float(row["target_nll"])))
    ranked = []
    for variant, values in sorted(grouped.items()):
        datasets = sorted({value[0] for value in values})
        if not datasets:
            continue
        per_dataset = {
            dataset: [value[2] for value in values if value[0] == dataset]
            for dataset in datasets
        }
        balanced = statistics.fmean(
            statistics.fmean(per_dataset[dataset]) for dataset in datasets
        )
        rng = random.Random(f"promotion:{seed}:{variant}")
        boot = []
        for _ in range(bootstrap_samples):
            boot.append(statistics.fmean(
                statistics.fmean(rng.choices(bucket, k=len(bucket)))
                for bucket in per_dataset.values()
            ))
        ranked.append({
            "variant": variant,
            "paired_samples": len(values),
            "datasets": len(datasets),
            "dataset_balanced_nll_improvement": balanced,
            "ci_low": float(np.quantile(boot, 0.025)),
            "ci_high": float(np.quantile(boot, 0.975)),
            "exploratory_promotion": float(np.quantile(boot, 0.025)) <= 0,
            "per_dataset_improvement": {
                dataset: statistics.fmean(bucket)
                for dataset, bucket in per_dataset.items()
            },
        })
    ranked.sort(
        key=lambda row: (
            -row["dataset_balanced_nll_improvement"], row["variant"],
        )
    )
    for index, row in enumerate(ranked, 1):
        row["rank"] = index
        row["selected"] = index <= 2
    return ranked
