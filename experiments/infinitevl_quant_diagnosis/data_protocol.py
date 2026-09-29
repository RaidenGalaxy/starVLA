from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

from experiments.infinitevl_quant.data_protocol import build_splits as build_original_splits
from experiments.vlm_quant_benchmark.data import (
    MultimodalSample,
    assert_disjoint,
    file_digest,
    load_manifest,
    normalized_question_digest,
)


def _stable_key(sample: MultimodalSample) -> str:
    return hashlib.sha256(f"infinitevl-cpu-v1:{sample.id}".encode()).hexdigest()


def _buckets(samples: list[MultimodalSample]) -> dict[str, list[MultimodalSample]]:
    result: dict[str, list[MultimodalSample]] = defaultdict(list)
    for sample in samples:
        result[sample.dataset].append(sample)
    return dict(sorted(result.items()))


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


def build_diagnosis_splits(
    calibration_path: Path, evaluation_path: Path,
) -> tuple[dict[str, list[MultimodalSample]], dict]:
    calibration = _buckets(load_manifest(calibration_path))
    evaluation = _buckets(load_manifest(evaluation_path))
    if len(calibration) != 4 or len(evaluation) != 4:
        raise ValueError("diagnosis requires exactly four calibration/evaluation sources")

    diagnosis_calibration, diagnosis_acceptance = [], []
    for source, rows in calibration.items():
        ordered = sorted(rows, key=_stable_key)
        if len(ordered) < 12:
            raise ValueError(f"calibration source {source} has fewer than 12 samples")
        diagnosis_calibration.extend(ordered[6:10])
        diagnosis_acceptance.extend(ordered[10:12])

    diagnosis_search, diagnosis_confirm = [], []
    for dataset, rows in evaluation.items():
        if len(rows) < 32:
            raise ValueError(f"evaluation dataset {dataset} has fewer than 32 samples")
        diagnosis_search.extend(rows[16:24])
        diagnosis_confirm.extend(rows[24:32])

    splits = {
        "diagnosis_calibration": diagnosis_calibration,
        "diagnosis_acceptance": diagnosis_acceptance,
        "diagnosis_search": diagnosis_search,
        "diagnosis_scan": [
            sample for dataset in sorted(evaluation)
            for sample in evaluation[dataset][16:18]
        ],
        "diagnosis_confirm": diagnosis_confirm,
    }
    assert_disjoint(diagnosis_calibration, diagnosis_acceptance)
    assert_disjoint(diagnosis_calibration, diagnosis_search)
    assert_disjoint(diagnosis_calibration, diagnosis_confirm)
    assert_disjoint(diagnosis_acceptance, diagnosis_search)
    assert_disjoint(diagnosis_acceptance, diagnosis_confirm)
    assert_disjoint(diagnosis_search, diagnosis_confirm)

    original = build_original_splits(calibration_path, evaluation_path)
    old_calibration = list({sample.id: sample for name in ("mbq", "vvs", "qig", "acceptance")
                            for sample in original[name]}.values())
    old_evaluation = list({sample.id: sample for name in ("smoke", "screen", "confirm")
                           for sample in original[name]}.values())
    checks = {
        "new_calibration_vs_old_calibration": _overlap(
            diagnosis_calibration + diagnosis_acceptance, old_calibration,
        ),
        "new_evaluation_vs_old_evaluation": _overlap(
            diagnosis_search + diagnosis_confirm, old_evaluation,
        ),
        "search_vs_confirm": _overlap(diagnosis_search, diagnosis_confirm),
        "calibration_vs_acceptance": _overlap(
            diagnosis_calibration, diagnosis_acceptance,
        ),
    }
    failures = {name: value for name, value in checks.items() if any(value.values())}
    if failures:
        raise ValueError(f"diagnosis split leakage: {failures}")
    audit = {
        "selection": {
            "diagnosis_calibration": "stable-hash positions 6:10 per source",
            "diagnosis_acceptance": "stable-hash positions 10:12 per source",
            "diagnosis_search": "manifest positions 16:24 per dataset",
            "diagnosis_scan": "manifest positions 16:18 per dataset",
            "diagnosis_confirm": "manifest positions 24:32 per dataset",
        },
        "counts": {name: len(rows) for name, rows in splits.items()},
        "zero_overlap": checks,
    }
    return splits, audit

