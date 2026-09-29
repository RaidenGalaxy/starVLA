from __future__ import annotations

import pytest

from experiments.vlm_quant_layer_pruning.analyze_shortv_budget import analyze
from experiments.vlm_quant_layer_pruning.rank_shortv_layers import build_ranking
from experiments.vlm_quant_layer_pruning.shortv_probe import SCHEMA_NAME


def _single_layer_rows() -> list[dict]:
    rows = []
    for sample_index in range(8):
        sample = f"s{sample_index}"
        dataset = "a" if sample_index < 4 else "b"
        for layer in range(36):
            rows.append({
                "schema_name": SCHEMA_NAME,
                "sample_id": sample,
                "dataset": dataset,
                "quant_scheme": "fp",
                "operation": "freeze_visual",
                "layer_indices": [layer],
                "metrics": {"answer_logit_kl": (layer + 1) * 1e-4},
            })
    return rows


def _budget_rows() -> list[dict]:
    rows = []
    values = {
        "fp": {4: (0.01, 0.005), 8: (0.02, 0.01)},
        "mbq_w4": {4: (0.02, 0.01), 8: (0.12, 0.06)},
        "mbq_w3": {4: (0.12, 0.06), 8: (0.18, 0.10)},
    }
    for sample_index in range(8):
        sample = f"s{sample_index}"
        dataset = "a" if sample_index < 4 else "b"
        for state in values:
            rows.append({
                "schema_name": SCHEMA_NAME,
                "sample_id": sample,
                "dataset": dataset,
                "quant_scheme": state,
                "operation": "baseline",
                "layer_indices": [],
                "metrics": {"target_nll": 1.0},
            })
            for budget, (delta, kl) in values[state].items():
                rows.append({
                    "schema_name": SCHEMA_NAME,
                    "sample_id": sample,
                    "dataset": dataset,
                    "quant_scheme": state,
                    "operation": "freeze_visual",
                    "layer_indices": list(range(budget)),
                    "metrics": {
                        "delta_target_nll": delta,
                        "answer_logit_kl": kl,
                    },
                })
    return rows


def test_fp_ranking_uses_all_layers_and_orders_by_upper_risk():
    result = build_ranking(_single_layer_rows(), bootstrap_samples=50, seed=7)
    assert result["samples"] == 8
    assert result["ranking"] == list(range(36))
    assert len(result["per_layer"]) == 36


def test_budget_frontiers_and_refinement_are_deterministic():
    ranking = {"ranking": list(range(36))}
    result = analyze(
        _budget_rows(), ranking, bootstrap_samples=50, seed=7,
    )
    assert result["proxy_safe_frontier"] == {"fp": 8, "mbq_w4": 4, "mbq_w3": 0}
    assert result["frontier_spread_layers"] == 8
    assert result["refinement_budgets"] == [1, 2, 3, 5, 6, 7]


def test_budget_rejects_non_prefix_layer_set():
    rows = _budget_rows()
    row = next(item for item in rows if item["operation"] == "freeze_visual")
    row["layer_indices"] = [1, 2, 3, 4]
    with pytest.raises(ValueError, match="not the FP-ranking prefix"):
        analyze(rows, {"ranking": list(range(36))}, bootstrap_samples=10, seed=7)


def test_budget_accepts_state_specific_refinement_grids():
    rows = _budget_rows()
    keep = {
        "fp": {4, 8},
        "mbq_w4": {4},
        "mbq_w3": {4, 8},
    }
    rows = [
        row for row in rows
        if row["operation"] == "baseline"
        or len(row["layer_indices"]) in keep[row["quant_scheme"]]
    ]
    result = analyze(
        rows, {"ranking": list(range(36))}, bootstrap_samples=50, seed=7,
    )
    assert result["budgets_by_state"] == {
        "fp": [0, 4, 8],
        "mbq_w4": [0, 4],
        "mbq_w3": [0, 4, 8],
    }
    assert result["proxy_safe_frontier"] == {"fp": 8, "mbq_w4": 4, "mbq_w3": 0}


def test_budget_allows_low_level_cpu_baseline_jitter():
    rows = _budget_rows()
    duplicate = dict(rows[0])
    duplicate["metrics"] = {"target_nll": 1.0 + 2e-6}
    result = analyze(
        [*rows, duplicate], {"ranking": list(range(36))},
        bootstrap_samples=10, seed=7,
    )
    assert result["samples"] == 8
