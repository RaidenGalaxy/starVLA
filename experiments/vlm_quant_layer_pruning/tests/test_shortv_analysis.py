from __future__ import annotations

from experiments.vlm_quant_layer_pruning.analyze_shortv import analyze
from experiments.vlm_quant_layer_pruning.shortv_probe import SCHEMA_NAME


def _rows(
    *, invert_w4: bool, positive_interaction: bool,
    candidate_scheme: str = "mbq_w4",
) -> list[dict]:
    rows = []
    layers = list(range(10))
    for sample_index in range(24):
        sample = f"s{sample_index:02d}"
        dataset = "a" if sample_index < 12 else "b"
        jitter = (sample_index % 3) * 1e-8
        for state, target_nll in (("fp", 1.0), (candidate_scheme, 1.1)):
            rows.append({
                "schema_name": SCHEMA_NAME,
                "sample_id": sample,
                "dataset": dataset,
                "quant_scheme": state,
                "operation": "baseline",
                "layer_indices": [],
                "metrics": {"target_nll": target_nll + jitter},
            })
        for layer in layers:
            fp_lc = (layer + 1) * 1e-4 + jitter
            w4_lc = ((10 - layer) if invert_w4 else (layer + 1)) * 1e-4 + jitter
            fp_delta = (layer + 1) * 1e-3 + jitter
            if positive_interaction:
                w4_delta = fp_delta + (10 - layer) * 1e-3
            else:
                w4_delta = fp_delta
            for state, lc, delta in (
                ("fp", fp_lc, fp_delta),
                (candidate_scheme, w4_lc, w4_delta),
            ):
                rows.append({
                    "schema_name": SCHEMA_NAME,
                    "sample_id": sample,
                    "dataset": dataset,
                    "quant_scheme": state,
                    "operation": "freeze_visual",
                    "layer_indices": [layer],
                    "metrics": {
                        "answer_logit_kl": lc,
                        "delta_target_nll": delta,
                    },
                })
    return rows


def test_qshortv_rank_inversion_and_interaction_pass_gate():
    result = analyze(
        _rows(invert_w4=True, positive_interaction=True),
        bootstrap_samples=200,
        seed=7,
    )
    assert result["Q_visual_rank_change"]["passes"] is True
    assert result["I_visual_positive_interaction"]["passes"] is True
    assert result["go"] is True


def test_qshortv_stable_fp_w4_profiles_stop():
    result = analyze(
        _rows(invert_w4=False, positive_interaction=False),
        bootstrap_samples=200,
        seed=7,
    )
    assert result["Q_visual_rank_change"]["passes"] is False
    assert result["I_visual_positive_interaction"]["passes"] is False
    assert result["go"] is False


def test_qshortv_supports_w3_candidate_scheme():
    result = analyze(
        _rows(
            invert_w4=True,
            positive_interaction=True,
            candidate_scheme="mbq_w3",
        ),
        candidate_scheme="mbq_w3",
        bootstrap_samples=200,
        seed=7,
    )
    assert result["candidate_scheme"] == "mbq_w3"
    assert result["Q_visual_rank_change"]["passes"] is True
    assert result["quantization_baseline"]["delta_target_nll_median"] > 0
