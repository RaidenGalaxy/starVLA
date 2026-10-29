from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from experiments.vlm_quant_benchmark.data import (
    MultimodalSample,
    file_digest,
    load_manifest,
    normalized_question_digest,
)


DATASETS = ("DocVQA_VAL", "MMMU_DEV_VAL", "ScienceQA_TEST", "TextVQA_VAL")


def _fingerprints(samples: list[MultimodalSample]) -> dict[str, set[str]]:
    return {
        "id": {sample.id for sample in samples},
        "image": {file_digest(image) for sample in samples for image in sample.images},
        "question": {
            sample.question_sha256
            or normalized_question_digest(sample.dataset, sample.prompt)
            for sample in samples
        },
    }


def _overlap(left: list[MultimodalSample], right: list[MultimodalSample]) -> dict[str, int]:
    left_hashes, right_hashes = _fingerprints(left), _fingerprints(right)
    return {
        key: len(left_hashes[key] & right_hashes[key])
        for key in ("id", "image", "question")
    }


def build_full_splits(path: Path) -> tuple[dict[str, list[MultimodalSample]], dict]:
    buckets: dict[str, list[MultimodalSample]] = defaultdict(list)
    for sample in load_manifest(path):
        buckets[sample.dataset].append(sample)
    if tuple(sorted(buckets)) != tuple(sorted(DATASETS)):
        raise ValueError(f"expected datasets {DATASETS}, got {tuple(sorted(buckets))}")
    if any(len(rows) != 64 for rows in buckets.values()):
        raise ValueError(
            f"full evaluation requires exactly 64 rows per dataset: "
            f"{ {key: len(value) for key, value in buckets.items()} }"
        )

    def positions(start: int, stop: int) -> list[MultimodalSample]:
        return [sample for dataset in sorted(buckets) for sample in buckets[dataset][start:stop]]

    splits = {
        "full256": positions(0, 64),
        "legacy128": positions(0, 32),
        "unseen_holdout128": positions(32, 64),
        "original_pilot64": positions(0, 16),
        "diagnosis_search32": positions(16, 24),
        "diagnosis_confirm32": positions(24, 32),
        "smoke4": [buckets[dataset][32] for dataset in sorted(buckets)],
    }
    checks = {
        "legacy_vs_unseen": _overlap(splits["legacy128"], splits["unseen_holdout128"]),
        "search_vs_unseen": _overlap(splits["diagnosis_search32"], splits["unseen_holdout128"]),
        "confirm_vs_unseen": _overlap(splits["diagnosis_confirm32"], splits["unseen_holdout128"]),
    }
    failures = {name: value for name, value in checks.items() if any(value.values())}
    if failures:
        raise ValueError(f"full-evaluation split leakage: {failures}")
    audit = {
        "selection": {
            "full256": "manifest positions 0:64 per dataset",
            "legacy128": "manifest positions 0:32 per dataset",
            "unseen_holdout128": "manifest positions 32:64 per dataset",
            "original_pilot64": "manifest positions 0:16 per dataset",
            "diagnosis_search32": "manifest positions 16:24 per dataset",
            "diagnosis_confirm32": "manifest positions 24:32 per dataset",
            "smoke4": "manifest position 32 per dataset",
        },
        "counts": {name: len(rows) for name, rows in splits.items()},
        "per_dataset": {
            name: {
                dataset: sum(sample.dataset == dataset for sample in rows)
                for dataset in DATASETS
            }
            for name, rows in splits.items()
        },
        "zero_overlap": checks,
    }
    return splits, audit

