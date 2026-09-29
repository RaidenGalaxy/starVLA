from experiments.vla_layer_token_sensitivity.analyze_qwen_oft_scan import analyze


def _summary(action_mse: float, action_drift: float, vision_drift: float) -> dict:
    return {
        "action_mse": action_mse,
        "final_hidden_cosine_drift": {
            "vision": vision_drift,
            "language": vision_drift,
            "action_query": action_drift,
        },
    }


def test_preregistered_gate_accepts_rank_inversion_and_positive_interaction() -> None:
    layers = [0, 1, 2, 3]
    rows = []
    for episode in range(4):
        sample_id = f"sample-{episode}"
        rows.extend(
            [
                {
                    "sample_id": sample_id,
                    "episode_index": episode,
                    "base_state": "fp",
                    "layer": -1,
                    "operation": "reference",
                    "token_counts": {"vision": 100, "language": 10, "action_query": 1},
                },
                {
                    "sample_id": sample_id,
                    "episode_index": episode,
                    "base_state": "w4",
                    "layer": -1,
                    "operation": "reference",
                    "metrics": {"vs_fp": _summary(0.1, 0.0, 0.0)},
                },
            ]
        )
        for position, layer in enumerate(layers):
            fp_error = 0.01 * (position + 1)
            rows.append(
                {
                    "sample_id": sample_id,
                    "episode_index": episode,
                    "base_state": "fp",
                    "layer": layer,
                    "operation": "skip",
                    "metrics": {
                        "vs_fp": _summary(
                            fp_error,
                            action_drift=float(position + 1),
                            vision_drift=float(len(layers) - position),
                        )
                    },
                }
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "episode_index": episode,
                    "base_state": "w4",
                    "layer": layer,
                    "operation": "skip",
                    "metrics": {
                        "vs_base": _summary(float(len(layers) - position), 0.0, 0.0),
                        "vs_fp": _summary(0.1 + fp_error + 0.5, 0.0, 0.0),
                    },
                }
            )

    result = analyze(rows, bootstrap_samples=100, seed=7)

    assert result["decision"] == "GO"
    assert result["gates"] == {
        "A_token_conditioned_ranking": True,
        "B_fp_w4_ranking_inversion": True,
        "C_positive_interaction_fraction": True,
    }
    assert result["C"]["positive_layer_fraction"] == 1.0
