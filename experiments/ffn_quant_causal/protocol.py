from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

from experiments.vlm_quant_benchmark.data import (
    MultimodalSample,
    file_digest,
    load_manifest,
    normalized_question_digest,
    sample_manifest_digest,
    write_manifest,
)


CALIBRATION_SOURCES = 4
CALIBRATION_ROWS_PER_SOURCE = 32
EVALUATION_DATASETS = (
    "DocVQA_VAL",
    "MMMU_DEV_VAL",
    "ScienceQA_TEST",
    "TextVQA_VAL",
)
EVALUATION_ROWS_PER_DATASET = 64


def _buckets(samples: list[MultimodalSample]) -> dict[str, list[MultimodalSample]]:
    result: dict[str, list[MultimodalSample]] = defaultdict(list)
    for sample in samples:
        result[sample.dataset].append(sample)
    return dict(sorted(result.items()))


def _stable_key(sample: MultimodalSample) -> str:
    return hashlib.sha256(f"ffn-quant-causal-v1:{sample.id}".encode()).hexdigest()


def _fingerprints(samples: list[MultimodalSample]) -> dict[str, set[str]]:
    return {
        "id": {sample.id for sample in samples},
        "image": {
            file_digest(image) for sample in samples for image in sample.images
        },
        "question": {
            sample.question_sha256
            or normalized_question_digest(sample.dataset, sample.prompt)
            for sample in samples
        },
    }


def _overlap(
    left: list[MultimodalSample], right: list[MultimodalSample],
) -> dict[str, int]:
    left_fingerprints = _fingerprints(left)
    right_fingerprints = _fingerprints(right)
    return {
        key: len(left_fingerprints[key] & right_fingerprints[key])
        for key in ("id", "image", "question")
    }


def audit_split_disjointness(
    splits: dict[str, list[MultimodalSample]],
) -> dict:
    """Audit within-protocol disjointness; historical reuse is reported separately."""

    disjoint_pairs = (
        ("mbq_calibration", "qig_acceptance"),
        ("mbq_calibration", "screen"),
        ("mbq_calibration", "confirm"),
        ("qig_acceptance", "screen"),
        ("qig_acceptance", "confirm"),
        ("screen", "confirm"),
    )
    checks = {
        f"{left}_vs_{right}": _overlap(splits[left], splits[right])
        for left, right in disjoint_pairs
    }
    failures = {name: row for name, row in checks.items() if any(row.values())}
    if failures:
        raise ValueError(f"causal-protocol split leakage: {failures}")

    mbq_ids = {sample.id for sample in splits["mbq_calibration"]}
    qig_ids = {sample.id for sample in splits["qig"]}
    vvs_ids = {sample.id for sample in splits["vvs"]}
    if not vvs_ids <= qig_ids <= mbq_ids:
        raise ValueError("VVS and QIG must be fixed nested calibration subsets")

    evaluation_union = {
        sample.id
        for name in ("qig_acceptance", "screen", "confirm")
        for sample in splits[name]
    }
    if evaluation_union != {sample.id for sample in splits["full256"]}:
        raise ValueError("acceptance/screen/confirm must partition full256 by ID")
    smoke_ids = {sample.id for sample in splits["smoke4"]}
    diagnostic_ids = {sample.id for sample in splits["diagnostic8"]}
    screen_ids = {sample.id for sample in splits["screen"]}
    if not smoke_ids <= screen_ids or not diagnostic_ids <= screen_ids:
        raise ValueError("all exploratory diagnostics must be subsets of screen")

    return {
        "zero_overlap": checks,
        "nested_subsets": {
            "vvs_within_qig": True,
            "qig_within_mbq_calibration": True,
            "note": "nested calibration subsets are intentional and are not holdouts",
        },
        "evaluation_partition": {
            "covers_full256_by_id": True,
            "qig_acceptance_reserved_for_qig_layer_acceptance": True,
            "qig_acceptance_not_used_by_rtn_or_within_protocol_holdout": True,
            "full256_is_descriptive_only": True,
            "smoke4_within_screen": True,
            "diagnostic8_within_screen": True,
            "confirm_compatibility_key": "confirm",
            "confirm_report_label": "within_protocol_holdout192",
            "evaluation_history": "historically_reused_evaluation",
            "historically_unseen_holdout": False,
        },
    }


def build_protocol_splits(
    calibration_path: Path, evaluation_path: Path,
) -> tuple[dict[str, list[MultimodalSample]], dict]:
    calibration = _buckets(load_manifest(calibration_path))
    evaluation = _buckets(load_manifest(evaluation_path))
    if len(calibration) != CALIBRATION_SOURCES:
        raise ValueError(
            f"expected {CALIBRATION_SOURCES} calibration sources, got {tuple(calibration)}"
        )
    calibration_counts = {key: len(value) for key, value in calibration.items()}
    if set(calibration_counts.values()) != {CALIBRATION_ROWS_PER_SOURCE}:
        raise ValueError(
            "expected exactly 32 calibration rows per source: "
            f"{calibration_counts}"
        )
    if tuple(evaluation) != tuple(sorted(EVALUATION_DATASETS)):
        raise ValueError(
            f"expected evaluation datasets {EVALUATION_DATASETS}, got {tuple(evaluation)}"
        )
    evaluation_counts = {key: len(value) for key, value in evaluation.items()}
    if set(evaluation_counts.values()) != {EVALUATION_ROWS_PER_DATASET}:
        raise ValueError(
            "expected exactly 64 evaluation rows per dataset: "
            f"{evaluation_counts}"
        )

    mbq_calibration = [
        sample for source in sorted(calibration) for sample in calibration[source]
    ]
    qig, vvs = [], []
    for source in sorted(calibration):
        ordered = sorted(calibration[source], key=_stable_key)
        qig.extend(ordered[:8])
        vvs.extend(ordered[:4])

    full256 = [
        sample for dataset in sorted(evaluation) for sample in evaluation[dataset]
    ]
    qig_acceptance = [
        sample for dataset in sorted(evaluation) for sample in evaluation[dataset][0:8]
    ]
    screen = [
        sample for dataset in sorted(evaluation) for sample in evaluation[dataset][8:16]
    ]
    confirm = [
        sample for dataset in sorted(evaluation) for sample in evaluation[dataset][16:64]
    ]
    splits = {
        "mbq_calibration": mbq_calibration,
        "vvs": vvs,
        "qig": qig,
        "qig_acceptance": qig_acceptance,
        "screen": screen,
        "diagnostic8": [
            sample
            for dataset in sorted(evaluation)
            for sample in evaluation[dataset][8:10]
        ],
        "confirm": confirm,
        "full256": full256,
        # Smoke/trace diagnostics reuse one already-screened item per dataset.
        # ``confirm`` is retained as a compatibility key only: its report label
        # is within_protocol_holdout192 because this manifest was used by prior
        # experiments and therefore is not a historically unseen holdout.
        "smoke4": [evaluation[dataset][8] for dataset in sorted(evaluation)],
    }
    disjointness = audit_split_disjointness(splits)
    audit = {
        "selection": {
            "mbq_calibration": "all 32 rows from each of four calibration sources",
            "vvs": "first 4/source after stable SHA256 ordering; subset of qig",
            "qig": "first 8/source after stable SHA256 ordering; subset of calibration",
            "qig_acceptance": "evaluation manifest positions 0:8 per dataset",
            "screen": "evaluation manifest positions 8:16 per dataset",
            "diagnostic8": "evaluation manifest positions 8:10 per dataset; screen subset",
            "confirm": (
                "evaluation manifest positions 16:64 per dataset; compatibility key for "
                "within_protocol_holdout192; historically reused evaluation"
            ),
            "full256": "evaluation manifest positions 0:64; descriptive only",
            "smoke4": "evaluation manifest position 8 per dataset; screen subset",
        },
        "counts": {name: len(rows) for name, rows in splits.items()},
        "per_source_or_dataset": {
            name: {
                dataset: sum(sample.dataset == dataset for sample in rows)
                for dataset in sorted({sample.dataset for sample in rows})
            }
            for name, rows in splits.items()
        },
        "digests": {
            name: sample_manifest_digest(rows) for name, rows in splits.items()
        },
        "holdout_semantics": {
            "compatibility_split_key": "confirm",
            "report_label": "within_protocol_holdout192",
            "evaluation_history": "historically_reused_evaluation",
            "historically_unseen": False,
            "interpretation": (
                "disjoint from calibration, acceptance, and screen inside this protocol; "
                "historically reused and not newly unseen across prior experiments"
            ),
        },
        **disjointness,
    }
    return splits, audit


def write_split_manifests(
    root: Path, splits: dict[str, list[MultimodalSample]],
) -> dict[str, dict]:
    root.mkdir(parents=True, exist_ok=True)
    result = {}
    for name, samples in splits.items():
        path = root / f"{name}.jsonl"
        write_manifest(path, samples)
        result[name] = {
            "path": str(path),
            "samples": len(samples),
            "digest": sample_manifest_digest(samples),
        }
    return result
