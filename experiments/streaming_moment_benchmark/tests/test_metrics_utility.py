from __future__ import annotations

import numpy as np

from experiments.streaming_moment_benchmark.fit_utility import grouped_cv_alpha
from experiments.streaming_moment_benchmark.analyze_cpu_pilot import paired_group_bootstrap
from experiments.streaming_moment_benchmark.metrics import aggregate_results
from experiments.streaming_moment_benchmark.utility import UtilityModel, fit_ridge_utility


def test_ridge_utility_roundtrip_and_signal() -> None:
    features = np.asarray([[0, 1], [0, 2], [1, 1], [1, 2]], dtype=float)
    targets = np.asarray([0, 0, 1, 1], dtype=float)
    model = fit_ridge_utility(features, targets, ["event", "noise"], alpha=0.01)
    restored = UtilityModel.from_dict(model.to_dict())
    predictions = restored.predict(np.asarray([[0, 1], [1, 1]], dtype=float))
    assert predictions[1] > predictions[0]


def test_grouped_cross_validation_selects_from_declared_grid() -> None:
    features = np.asarray([[0.0], [0.2], [1.0], [1.2], [2.0], [2.2], [3.0], [3.2]])
    targets = features[:, 0] * 0.5
    alpha, report = grouped_cv_alpha(
        features,
        targets,
        ["a", "a", "b", "b", "c", "c", "d", "d"],
        ["x"],
        alphas=[0.1, 1.0, 10.0],
        folds=4,
    )
    assert alpha in {0.1, 1.0, 10.0}
    assert all(item["folds"] == 4 for item in report)


def test_paired_bootstrap_resamples_source_groups() -> None:
    result = paired_group_bootstrap({"a": [1.0, 1.0], "b": [0.5]}, samples=1000, seed=3)
    assert result["groups"] == 2
    assert result["mean"] == 0.75
    assert result["ci95_low"] >= 0.5


def test_aggregate_reports_delay_slope_and_auc() -> None:
    rows = []
    for memory, delay, score in [(4, 0, 1), (4, 60, 0), (8, 0, 1), (8, 60, 1)]:
        rows.append(
            {
                "method": "x",
                "memory_frames": memory,
                "encode_fps": 1.0,
                "query_delay_sec": delay,
                "condition": {"delay_sec": delay, "distractor_profile": "clean"},
                "prediction": {"score": score},
                "metrics": {
                    "capture_span_recall": 1.0,
                    "retention_span_recall": score,
                    "required_frame_coverage": score,
                    "retention_loss": 1 - score,
                    "frames_retained": memory,
                    "budget_respected": True,
                },
            }
        )
    summary = aggregate_results(rows)
    configs = {item["memory_frames"]: item for item in summary["configurations"]}
    assert configs[4]["degradation_per_delay_minute"] < 0
    assert configs[8]["degradation_per_delay_minute"] == 0
    assert summary["accuracy_budget_curves"][0]["accuracy_budget_auc"] == 0.75
