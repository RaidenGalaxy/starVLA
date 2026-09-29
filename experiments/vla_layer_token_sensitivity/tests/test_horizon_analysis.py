import numpy as np

from experiments.vla_layer_token_sensitivity.analyze_qwen_oft_horizon import analyze_horizon


def _metric(step_values: list[float], action_mse: float | None = None) -> dict:
    values = np.asarray(step_values, dtype=np.float64)
    return {
        "action_step_mse": values.tolist(),
        "action_mse": float(values.mean() if action_mse is None else action_mse),
    }


def test_horizon_analyzer_detects_late_signal_hidden_by_overall_average() -> None:
    rows = []
    quant = np.full(16, 0.1)
    skip = np.full(16, 0.1)
    interaction = np.asarray([-0.15] * 10 + [0.30] * 5 + [0.0])
    assert abs(float(interaction.mean())) < 1e-12
    joint = quant + skip + interaction

    for episode in range(8):
        sample = f"heldout-{episode}"
        common = {
            "schema_version": 2,
            "sample_id": sample,
            "episode_index": episode,
            "input_condition": "state",
        }
        rows.extend(
            [
                {
                    **common,
                    "base_state": "fp",
                    "layer": -1,
                    "operation": "reference",
                    "metrics": {"vs_ground_truth": _metric([0.1] * 16, action_mse=0.1)},
                    "baseline_vs_ground_truth": _metric([1.0] * 16, action_mse=1.0),
                },
                {
                    **common,
                    "base_state": "w4",
                    "layer": -1,
                    "operation": "reference",
                    "metrics": {"vs_fp": _metric(quant.tolist())},
                },
                {
                    **common,
                    "base_state": "fp",
                    "layer": 34,
                    "operation": "skip",
                    "metrics": {"vs_fp": _metric(skip.tolist())},
                },
                {
                    **common,
                    "base_state": "w4",
                    "layer": 34,
                    "operation": "skip",
                    "metrics": {
                        "vs_fp": _metric(joint.tolist()),
                        "vs_base": _metric(skip.tolist()),
                    },
                },
            ]
        )

    result = analyze_horizon(rows, primary_layer=34, bootstrap_samples=200, seed=7)

    assert result["primary"]["decision"] == "CONFIRM_GO"
    assert result["primary"]["late_interaction"]["ci95"][0] > 0.0
    assert result["primary"]["late_minus_early"]["ci95"][0] > 0.0
    assert result["full_36_layer_gate"]["decision"] == "PENDING"
