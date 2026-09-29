from __future__ import annotations

from experiments.vlm_quant_layer_pruning.analyze import analyze, bh_adjust


def _synthetic_rows():
    rows = []
    layers = list(range(10))
    for sample_index in range(24):
        sample = f"s{sample_index:02d}"
        dataset = "a" if sample_index < 12 else "b"
        offset = (sample_index % 3) * 1e-7
        for layer in layers:
            for state in ("fp", "mbq_w4"):
                if state == "fp":
                    delta = (layer + 1) * 1e-3 + offset
                    vision = (layer + 1) * 1e-4
                    prompt = (layer + 1) * 1.1e-4
                    answer = (layer + 1) * 0.9e-4
                else:
                    delta = (10 - layer) * 1e-3 + offset
                    vision = (10 - layer) * 1e-4
                    prompt = (10 - layer) * 1.1e-4
                    answer = (layer + 1) * 0.9e-4
                rows.append({
                    "sample_id": sample,
                    "dataset": dataset,
                    "quant_scheme": state,
                    "operation": "skip",
                    "layer_indices": [layer],
                    "metrics": {
                        "delta_target_nll": delta,
                        "final_hidden_cosine_drift": {
                            "vision": vision,
                            "prompt_language": prompt,
                            "answer_predictor": answer,
                        },
                    },
                })
    return rows


def test_synthetic_rank_inversion_and_token_divergence_pass_gate():
    result = analyze(_synthetic_rows(), bootstrap_samples=200, seed=7)
    assert result["Q_rank_change"]["passes"] is True
    assert result["T_token_divergence"]["passes"] is True
    assert len(result["I_positive_interaction"]["passing_layers"]) >= 2
    assert result["go"] is True


def test_bh_adjust_is_monotonic_in_rank_order():
    adjusted = bh_adjust([0.01, 0.04, 0.03, 0.20])
    ordered = sorted(zip([0.01, 0.04, 0.03, 0.20], adjusted))
    assert [value for _, value in ordered] == sorted(value for _, value in ordered)

