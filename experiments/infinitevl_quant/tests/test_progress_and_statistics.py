from __future__ import annotations

import json

import pytest

from experiments.infinitevl_quant.progress import StageProgress
from experiments.infinitevl_quant.statistics import paired_bootstrap, prediction_transitions


def test_progress_resumes_only_missing_samples(tmp_path):
    progress = StageProgress(tmp_path, "screen", config_digest="abc", inputs={"model": "x"})
    progress.complete_sample("w4", "sample:1", {"id": "sample:1", "value": 3})
    assert progress.completed_sample("w4", "sample:1")["value"] == 3
    assert progress.completed_sample("w4", "sample:2") is None
    assert len(progress.rows("w4")) == 1
    (progress.root / "metrics.json").write_text("{}")
    progress.mark_complete(("metrics.json",))
    assert progress.is_complete(("metrics.json",))
    assert not progress.is_complete(("other.json",))
    with pytest.raises(RuntimeError, match="different inputs"):
        StageProgress(tmp_path, "screen", config_digest="changed", inputs={"model": "x"})


def test_paired_bootstrap_and_transitions_are_paired():
    baseline = [
        {"id": "a", "dataset": "D", "score": 0.0, "prediction": "x"},
        {"id": "b", "dataset": "D", "score": 1.0, "prediction": "y"},
    ]
    candidate = [
        {"id": "a", "dataset": "D", "score": 1.0, "prediction": "z"},
        {"id": "b", "dataset": "D", "score": 0.0, "prediction": "q"},
    ]
    result = paired_bootstrap(baseline, candidate, samples=100, seed=1)
    assert result["D"]["delta_points"] == 0.0
    transitions = prediction_transitions(baseline, candidate)
    assert transitions["counts"]["improved"] == 1
    assert transitions["counts"]["regressed"] == 1
