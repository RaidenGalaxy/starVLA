from __future__ import annotations

import math

from ..config import MEDIATOR_GROUPS
from ..statistics import (
    gain_ratio_bootstrap,
    length_stratified_gap_reduction_bootstrap,
    powerset,
    select_mediators,
    shapley_values,
)


def _row(identifier: str, dataset: str, nll: float, length: int = 300) -> dict:
    return {
        "id": identifier,
        "dataset": dataset,
        "sequence_tokens": length,
        "target_text": {"tokens": 2, "total_nll": 2 * nll},
    }


def test_all_16_mediator_cells_and_shapley_efficiency() -> None:
    contributions = {"q": 1.0, "write": 2.0, "decay": 3.0, "output_gate": 4.0}
    cells = {subset: 20.0 - sum(contributions[group] for group in subset) for subset in powerset()}
    result = shapley_values(cells)
    assert len(cells) == 16
    assert result["contributions"] == contributions
    assert math.isclose(
        sum(result["contributions"].values()),
        result["total_repair_effect"],
    )
    datasets = {f"d{index}": result for index in range(4)}
    selection = select_mediators(datasets, minimum_share=0.15, maximum=2)
    assert selection["selected"] == ["output_gate", "decay"]


def test_gain_bootstrap_uses_samples_not_layer_pairs_as_units() -> None:
    rows = []
    for sample, dataset, ratios in (
        ("a", "d1", (2.0, 8.0)),
        ("b", "d2", (0.5, 2.0)),
    ):
        for pair, ratio in zip(("3-4", "15-16"), ratios, strict=True):
            rows.extend(
                [
                    {
                        "id": f"{sample}-{pair}-gdn",
                        "sample_id": sample,
                        "dataset": dataset,
                        "pair": pair,
                        "structure": "gdn",
                        "block_gain": ratio,
                    },
                    {
                        "id": f"{sample}-{pair}-swa",
                        "sample_id": sample,
                        "dataset": dataset,
                        "pair": pair,
                        "structure": "swa",
                        "block_gain": 1.0,
                    },
                ]
            )
    result = gain_ratio_bootstrap(rows, samples=100, seed=3)
    assert result["paired_samples"] == 2
    assert result["layer_pairs_per_sample"] == 2
    assert result["sample_pair_cells"] == 4
    assert math.isclose(result["point"], 2.0)


def test_length_strata_and_trend_bootstrap_are_sample_paired() -> None:
    specifications = (
        ("a", "d1", 128, 0.02),
        ("b", "d2", 192, 0.03),
        ("c", "d1", 300, 0.06),
        ("d", "d2", 480, 0.08),
        ("e", "d1", 700, 0.15),
        ("f", "d2", 900, 0.18),
    )
    baseline = [_row(item, dataset, 1.0, length) for item, dataset, length, _ in specifications]
    raw = [_row(item, dataset, 1.2, length) for item, dataset, length, _ in specifications]
    repaired = [_row(item, dataset, 1.2 - benefit, length) for item, dataset, length, benefit in specifications]
    result = length_stratified_gap_reduction_bootstrap(
        baseline,
        raw,
        repaired,
        samples=200,
        seed=9,
    )
    assert set(result["bins"]) == {"le_256", "257_512", "gt_512"}
    assert all(value["paired_samples"] == 2 for value in result["bins"].values())
    assert all(value["status"] == "estimated" for value in result["bins"].values())
    assert result["trend"]["slope_per_log2_token"] > 0
    assert result["trend"]["increases_with_length"]
    assert set(MEDIATOR_GROUPS) == {"q", "write", "decay", "output_gate"}


def test_empty_preregistered_length_bin_is_reported_not_redefined() -> None:
    baseline = [_row("a", "d1", 1.0, 300), _row("b", "d2", 1.0, 700)]
    raw = [_row("a", "d1", 1.2, 300), _row("b", "d2", 1.2, 700)]
    repaired = [_row("a", "d1", 1.1, 300), _row("b", "d2", 1.0, 700)]
    result = length_stratified_gap_reduction_bootstrap(
        baseline,
        raw,
        repaired,
        samples=20,
        seed=13,
    )
    assert result["bins"]["le_256"] == {
        "status": "not_estimable",
        "paired_samples": 0,
        "reason": "no confirm192 samples in the preregistered length bin",
    }
    assert result["bins"]["257_512"]["status"] == "estimated"
    assert result["bins"]["gt_512"]["status"] == "estimated"
