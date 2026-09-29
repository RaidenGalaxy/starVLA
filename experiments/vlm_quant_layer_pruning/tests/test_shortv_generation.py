from __future__ import annotations

import json

import pytest

from experiments.vlm_quant_layer_pruning.analyze_shortv_generation import (
    analyze_comparison,
)
from experiments.vlm_quant_layer_pruning.prepare_shortv_generation import (
    DATASETS,
    assert_content_disjoint,
    select_first_per_dataset,
)


def _write_scores(root, manifest, score):
    for dataset in DATASETS:
        path = root / dataset / "official_samples.jsonl"
        path.parent.mkdir(parents=True)
        with path.open("w") as handle:
            for row in manifest:
                if row["dataset"] == dataset:
                    handle.write(json.dumps({
                        "id": row["id"], "official_score": score(dataset),
                    }) + "\n")


def test_generation_subset_is_ordered_and_content_disjoint():
    rows = [
        {
            "id": f"{dataset}:{index}",
            "dataset": dataset,
            "question_sha256": f"q-{dataset}-{index}",
            "image_sha256": f"i-{dataset}-{index}",
        }
        for dataset in DATASETS for index in range(4)
    ]
    selected = select_first_per_dataset(rows, 2)
    assert [row["id"] for row in selected[:2]] == ["DocVQA_VAL:0", "DocVQA_VAL:1"]
    assert len(selected) == 8
    assert assert_content_disjoint(selected, [{
        "id": "other", "question_sha256": "q-other", "image_sha256": "i-other",
    }]) == {"id": 0, "question_sha256": 0, "image_sha256": 0}


def test_generation_gate_uses_paired_stratified_bootstrap(tmp_path):
    manifest = [
        {"id": f"{dataset}:{index}", "dataset": dataset}
        for dataset in DATASETS for index in range(4)
    ]
    baseline = tmp_path / "baseline"
    frozen = tmp_path / "frozen"
    _write_scores(baseline, manifest, lambda _dataset: 1.0)
    _write_scores(frozen, manifest, lambda _dataset: 0.995)
    result = analyze_comparison(
        manifest, baseline, frozen, bootstrap_samples=100, seed=7,
    )
    assert result["safe"] is True
    assert result["macro_score_drop"] == pytest.approx(0.005)

    failing = tmp_path / "failing"
    _write_scores(
        failing, manifest,
        lambda dataset: 0.9 if dataset == "DocVQA_VAL" else 1.0,
    )
    result = analyze_comparison(
        manifest, baseline, failing, bootstrap_samples=100, seed=7,
    )
    assert result["safe"] is False
    assert result["max_dataset_score_drop"] > 0.05
