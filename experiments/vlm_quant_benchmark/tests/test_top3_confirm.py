import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from experiments.vlm_quant_benchmark.dataset_protocol import ACTIVE_DATASET_NAMES
from experiments.vlm_quant_benchmark.prepare_vlmeval_screen import extend_positions
from experiments.vlm_quant_benchmark.run_top3_confirm import confirm_configs
from experiments.vlm_quant_benchmark.run_vlmeval_official import (
    merge_prediction_donors,
)
from experiments.vlm_quant_benchmark import top3_confirm_report as report


def test_top3_matrix_is_exactly_bf16_plus_three_paired_w4_w3_runs():
    configs = confirm_configs()
    assert len(configs) == 7
    assert [row["bits"] for row in configs] == [None, 4, 4, 4, 3, 3, 3]
    assert "rtn" not in {row["method"] for row in configs}
    assert {row["method"] for row in configs[1:]} == {
        "mbq", "omniquant", "squeezellm",
    }


def test_extend_positions_preserves_prefix_and_adds_disjoint_rows():
    data = pd.DataFrame({
        "index": [str(index) for index in range(12)],
        "answer_type": ["open", "mcq"] * 6,
        "category": ["a", "b", "c"] * 4,
    })
    positions, additional = extend_positions(
        data, count=8, seed=42, preserved_indices=["7", "2"],
    )
    assert positions[:2] == [7, 2]
    assert len(positions) == len(set(positions)) == 8
    assert len(additional) == 6
    assert not {7, 2} & set(additional)


def test_extend_positions_rejects_partial_circular_group():
    data = pd.DataFrame({
        "index": ["0", "1", "2", "3"],
        "g_index": ["g0", "g0", "g1", "g1"],
    })
    with pytest.raises(ValueError, match="splits"):
        extend_positions(data, count=3, seed=42, preserved_indices=["0"])


def _config(method="mbq"):
    return {
        "method": method,
        "scope": "decoder_canonical",
        "weight_bits": 4,
        "activation_bits": None,
        "group_size": 128,
        "zero_point": True,
        "weight_granularity": "groupwise",
    }


def _donor(root: Path, config: dict, rows: list[dict]):
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps(config))
    dataset = root / "DocVQA_VAL"
    dataset.mkdir()
    (dataset / "generations.partial.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )


def test_multiple_prediction_donors_merge_and_record_sources(tmp_path):
    expected = {"1": "hash-1", "2": "hash-2", "3": "hash-3"}
    first, second = tmp_path / "first", tmp_path / "second"
    _donor(first, _config(), [{
        "index": "1", "dataset_row_sha256": "hash-1", "prediction": "a",
    }])
    _donor(second, _config(), [
        {"index": "1", "dataset_row_sha256": "hash-1", "prediction": "a"},
        {"index": "2", "dataset_row_sha256": "hash-2", "prediction": "b"},
        {"index": "9", "dataset_row_sha256": "other", "prediction": "ignored"},
    ])
    rows, details = merge_prediction_donors(
        [], [first, second], config=_config(),
        dataset_name="DocVQA_VAL", expected_rows=expected,
    )
    assert {row["index"] for row in rows} == {"1", "2"}
    assert all(row["prediction_source"].startswith("donor:") for row in rows)
    assert [item["matched"] for item in details] == [1, 1]


def test_prediction_donor_conflict_mismatch_and_duplicates_fail(tmp_path):
    expected = {"1": "hash-1"}
    first, conflict = tmp_path / "first", tmp_path / "conflict"
    row = {"index": "1", "dataset_row_sha256": "hash-1", "prediction": "a"}
    _donor(first, _config(), [row])
    _donor(conflict, _config(), [row | {"prediction": "different"}])
    with pytest.raises(ValueError, match="conflict"):
        merge_prediction_donors(
            [], [first, conflict], config=_config(),
            dataset_name="DocVQA_VAL", expected_rows=expected,
        )

    mismatch = tmp_path / "mismatch"
    _donor(mismatch, _config("omniquant"), [row])
    with pytest.raises(ValueError, match="config mismatch"):
        merge_prediction_donors(
            [], [mismatch], config=_config(),
            dataset_name="DocVQA_VAL", expected_rows=expected,
        )

    duplicate = tmp_path / "duplicate"
    _donor(duplicate, _config(), [row, row])
    with pytest.raises(ValueError, match="duplicate donor"):
        merge_prediction_donors(
            [], [duplicate], config=_config(),
            dataset_name="DocVQA_VAL", expected_rows=expected,
        )


def test_quality_classification_thresholds():
    reference = {name: 100.0 for name in ACTIVE_DATASET_NAMES}
    assert report.classify_quality(
        {name: 98.0 for name in ACTIVE_DATASET_NAMES}, reference,
    )["tier"] == "high_precision"
    assert report.classify_quality(
        {name: 96.0 for name in ACTIVE_DATASET_NAMES}, reference,
    )["tier"] == "usable"
    assert report.classify_quality(
        {name: 80.0 for name in ACTIVE_DATASET_NAMES}, reference,
    )["tier"] == "degraded"


def _result(root: Path, method: str, bits: int | None, count: int):
    scheme = (
        "bf16_value_fp32_compute"
        if bits is None else f"{method}_w{bits}a16_g128_asym"
    )
    directory = root / scheme
    directory.mkdir(parents=True)
    samples = [
        {
            "dataset": dataset,
            "index": str(index),
            "official_score": 1.0 if method != "squeezellm" or index % 20 else 0.0,
        }
        for dataset in ACTIVE_DATASET_NAMES
        for index in range(count)
    ]
    payload = {
        "status": "complete",
        "method": method,
        "scheme_id": scheme,
        "scope": "decoder_canonical",
        "weight_bits": bits,
        "activation_bits": None,
        "logical_bits_per_weight": 16.0 if bits is None else float(bits),
        "datasets": {name: {"samples": count} for name in ACTIVE_DATASET_NAMES},
        "samples": samples,
    }
    (directory / "official_result.json").write_text(json.dumps(payload))


def test_report_accepts_256_top3_and_keeps_rtn_historical(
    tmp_path, monkeypatch,
):
    confirm, historical = tmp_path / "confirm", tmp_path / "historical"
    _result(confirm, "bf16_value_fp32_compute", None, 256)
    for bits in (4, 3):
        for method in ("mbq", "omniquant", "squeezellm"):
            _result(confirm, method, bits, 256)
    _result(historical, "bf16_value_fp32_compute", None, 64)
    for bits in (4, 3):
        _result(historical, "rtn", bits, 64)

    monkeypatch.setattr(
        report, "score_ci",
        lambda _dataset, _values: {"type": "test", "interval": [0.0, 100.0]},
    )
    monkeypatch.setattr(
        report, "paired_bootstrap",
        lambda candidate, reference, **_kwargs: {
            "delta_pp": 100 * (sum(candidate) - sum(reference)) / len(candidate),
            "bootstrap_95_ci": [-1.0, 1.0],
            "conclusion": "indistinguishable",
        },
    )
    output = tmp_path / "summary"
    monkeypatch.setattr(sys, "argv", [
        "top3_confirm_report",
        "--confirm-root", str(confirm),
        "--screen64-root", str(historical),
        "--corpus-root", str(tmp_path / "corpus"),
        "--output-dir", str(output),
    ])
    report.main()
    summary = json.loads((output / "summary.json").read_text())
    selection = json.loads((output / "selection.json").read_text())
    assert len(summary["runs"]) == 7
    assert len(summary["historical_rtn"]) == 2
    assert all(
        winner["method"] != "rtn"
        for winner in selection["winners"].values()
    )
