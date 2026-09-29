from __future__ import annotations

from experiments.infinitevl_quant_ffn_bf16.statistics import stratified_paired_bootstrap


def _row(identifier, dataset, score, nll):
    return {
        "id": identifier, "dataset": dataset, "score": score,
        "target_text": {"total_nll": nll, "tokens": 1},
    }


def test_stratified_bootstrap_is_paired_and_deterministic():
    baseline = [_row("a", "A", 0.0, 1.0), _row("b", "A", 1.0, 1.0),
                _row("c", "B", 1.0, 2.0), _row("d", "B", 0.0, 2.0)]
    candidate = [_row("a", "A", 1.0, 2.0), _row("b", "A", 0.0, 2.0),
                 _row("c", "B", 1.0, 1.0), _row("d", "B", 0.0, 1.0)]
    first = stratified_paired_bootstrap(baseline, candidate, samples=100, seed=7)
    second = stratified_paired_bootstrap(baseline, candidate, samples=100, seed=7)
    assert first == second
    assert first["macro_score_delta_points"]["point"] == 0.0
    assert first["target_nll_relative_increase"]["point"] == 0.0

