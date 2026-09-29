from __future__ import annotations

import pytest
import torch

from experiments.infinitevl_quant.state_probe import PREFIXES, compare_states


def _states():
    return {
        f"language_model.layers.{index}.self_attn": {
            prefix: torch.ones(1, 1, 2, 3) * (index + prefix)
            for prefix in PREFIXES
        }
        for index in range(27)
    }


def test_state_comparison_requires_and_reports_exact_27_by_4_matrix():
    reference = _states()
    candidate = {
        layer: {prefix: value * 1.01 for prefix, value in prefixes.items()}
        for layer, prefixes in reference.items()
    }
    result = compare_states(reference, candidate)
    assert len(result["rows"]) == 27 * 4
    assert set(result["by_prefix"]) == {str(value) for value in PREFIXES}
    assert result["state_drift_auc"] > 0


def test_state_comparison_rejects_missing_prefix():
    reference, candidate = _states(), _states()
    candidate[next(iter(candidate))].pop(512)
    with pytest.raises(ValueError, match="all required prefixes"):
        compare_states(reference, candidate)

