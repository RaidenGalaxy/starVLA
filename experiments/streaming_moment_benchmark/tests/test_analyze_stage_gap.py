from __future__ import annotations

import json

import pytest

from experiments.streaming_moment_benchmark.analyze_stage_gap import (
    _distribution,
    _group_summary,
    _load,
    _load_metadata,
)


def _result(sample_id: str, answer: str = "A", prediction: str = "A") -> dict:
    return {
        "sample_id": sample_id,
        "status": "ok",
        "question": {"answer": answer},
        "prediction": {"parsed_answer": prediction, "score": float(answer == prediction)},
    }


def test_load_deduplicates_consistent_rows_and_rejects_conflicts(tmp_path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(json.dumps(_result("sample-1")) + "\n", encoding="utf-8")
    second.write_text(json.dumps(_result("sample-1")) + "\n", encoding="utf-8")
    assert len(_load([first, second])) == 1

    second.write_text(json.dumps(_result("sample-1", prediction="B")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="conflicting duplicate"):
        _load([first, second])


def test_metadata_grouping_and_none_distribution(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"sample_id": "sample-1", "metadata": {"category": "animal"}},
        {"sample_id": "sample-1", "metadata": {"category": "animal"}},
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert _load_metadata(manifest)["sample-1"]["category"] == "animal"
    assert _distribution(["A", None, "A"]) == {"<none>": 1, "A": 2}

    summary = _group_summary(
        [
            {
                "category": "animal",
                "reference_score": 1.0,
                "candidate_score": 0.0,
                "delta": 1.0,
            }
        ],
        "category",
    )
    assert summary["animal"]["reference_minus_candidate"] == 1.0
