import math

import pytest

from experiments.infinitevl_quant_diagnosis.attribution import (
    MODULES,
    all_scope_masks,
    mask_key,
    pair_interactions,
    shapley_three,
)


def test_shapley_recovers_additive_effects():
    effects = dict(zip(MODULES, (3.0, 2.0, 1.0), strict=True))
    values = {
        mask_key(mask): 10.0 + sum(effects[module] for module in mask)
        for mask in all_scope_masks()
    }
    result = shapley_three(values)
    assert result["contribution"] == pytest.approx(effects)
    assert math.isclose(sum(result["fraction"].values()), 1.0)
    assert all(value == 0.0 for value in pair_interactions(values).values())


def test_pair_interaction_detects_ffn_swa_synergy():
    values = {}
    for mask in all_scope_masks():
        value = float(len(mask))
        if {"decoder_ffn", "swa_attention"}.issubset(mask):
            value += 4.0
        values[mask_key(mask)] = value
    interactions = pair_interactions(values)
    assert interactions["decoder_ffn*swa_attention"] == 4.0
