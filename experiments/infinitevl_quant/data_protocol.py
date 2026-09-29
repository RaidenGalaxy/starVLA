from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

from experiments.vlm_quant_benchmark.data import (
    MultimodalSample,
    assert_disjoint,
    file_digest,
    load_manifest,
    normalized_question_digest,
    sample_manifest_digest,
    write_manifest,
)


def _stable_key(sample: MultimodalSample, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{sample.id}".encode()).hexdigest()


def _by_dataset(samples: list[MultimodalSample]) -> dict[str, list[MultimodalSample]]:
    result: dict[str, list[MultimodalSample]] = defaultdict(list)
    for sample in samples:
        result[sample.dataset].append(sample)
    return dict(sorted(result.items()))


def build_splits(calibration_path: Path, evaluation_path: Path) -> dict[str, list[MultimodalSample]]:
    calibration = load_manifest(calibration_path)
    evaluation = load_manifest(evaluation_path)
    calibration_buckets = _by_dataset(calibration)
    evaluation_buckets = _by_dataset(evaluation)
    if len(calibration_buckets) != 4 or len(evaluation_buckets) != 4:
        raise ValueError("the InfiniteVL pilot requires exactly four sources/datasets")

    mbq, vvs, qig, acceptance = [], [], [], []
    for source, rows in calibration_buckets.items():
        ordered = sorted(rows, key=lambda item: _stable_key(item, "infinitevl-cpu-v1"))
        if len(ordered) < 6:
            raise ValueError(f"calibration source {source} has fewer than six samples")
        mbq.extend(ordered[:4])
        vvs.extend(ordered[:1])
        qig.extend(ordered[:2])
        acceptance.extend(ordered[4:6])

    smoke, screen, confirm = [], [], []
    for dataset, rows in evaluation_buckets.items():
        if len(rows) < 16:
            raise ValueError(f"evaluation dataset {dataset} has fewer than 16 samples")
        smoke.extend(rows[:1])
        screen.extend(rows[:8])
        confirm.extend(rows[8:16])

    assert_disjoint(mbq, acceptance)
    assert_disjoint(mbq, screen)
    assert_disjoint(mbq, confirm)
    assert_disjoint(acceptance, screen)
    assert_disjoint(acceptance, confirm)
    assert_disjoint(screen, confirm)
    return {
        "mbq": mbq, "vvs": vvs, "qig": qig, "acceptance": acceptance,
        "smoke": smoke, "screen": screen, "confirm": confirm,
    }


def write_split_manifests(root: Path, splits: dict[str, list[MultimodalSample]]) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, samples in splits.items():
        path = root / f"{name}.jsonl"
        write_manifest(path, samples)
        result[name] = {
            "path": str(path), "samples": len(samples),
            "digest": sample_manifest_digest(samples),
            "ids": [sample.id for sample in samples],
        }
    return result


def sample_hashes(sample: MultimodalSample) -> tuple[str, tuple[str, ...], str]:
    return (
        sample.id,
        tuple(file_digest(image) for image in sample.images),
        sample.question_sha256 or normalized_question_digest(sample.dataset, sample.prompt),
    )
