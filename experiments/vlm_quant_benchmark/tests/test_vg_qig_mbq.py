from __future__ import annotations

import hashlib
import json
import dataclasses

import torch
from torch import nn

from experiments.vlm_quant_benchmark.config import (
    ExperimentConfig, ImplementationTier, MethodName,
)
from experiments.vlm_quant_benchmark.selection import SelectedLinear
from experiments.vlm_quant_benchmark.methods import W8A8Linear
from experiments.vlm_quant_benchmark.vg_qig_mbq import (
    BlockTrace, auto_gate_layers, block_objective, choose_qig_steps,
    make_quantized_block, qig_token_importance, search_vulnerable_block,
    vg_cache_key, vg_vulnerability_cache_key, vulnerability_score,
)


class ToyAttention(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)

    def forward(self, x):
        return self.o_proj((self.q_proj(x) + self.k_proj(x) + self.v_proj(x)) / 3)


class ToyMLP(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.gate_proj = nn.Linear(width, width, bias=False)
        self.up_proj = nn.Linear(width, width, bias=False)
        self.down_proj = nn.Linear(width, width, bias=False)

    def forward(self, x):
        return self.down_proj(torch.sigmoid(self.gate_proj(x)) * self.up_proj(x))


class ToyBlock(nn.Module):
    def __init__(self, width: int = 8):
        super().__init__()
        self.self_attn = ToyAttention(width)
        self.mlp = ToyMLP(width)

    def forward(self, hidden_states, **_kwargs):
        return (hidden_states + self.self_attn(hidden_states) + self.mlp(hidden_states),)


def vg_config(method=MethodName.VG_QIG_MBQ):
    return ExperimentConfig(
        method, weight_bits=3, group_size=4,
        mbq_scale_grid=(0.0, 0.5), vg_coordinate_sweeps=1,
        implementation_tier=ImplementationTier.CPU_RESEARCH,
    )


def selected(block):
    result = []
    for name, module in block.named_modules():
        if isinstance(module, nn.Linear):
            full = f"model.language_model.layers.0.{name}"
            result.append(SelectedLinear(full, module, "decoder", module.weight.numel()))
    return result


def trace(hidden=None):
    hidden = torch.randn(1, 6, 8) if hidden is None else hidden
    return BlockTrace(
        "one", "source", hidden, {},
        torch.tensor([[True, True, False, False, False, False]]),
        torch.tensor([[False, False, False, True, True, True]]),
    )


def test_legacy_mbq_serialization_omits_all_vg_fields():
    config = ExperimentConfig(MethodName.MBQ, weight_bits=3)
    payload = config.to_dict()
    assert not any(key.startswith("vg_") for key in payload)
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert config.digest == expected


def test_vulnerability_score_boundaries():
    assert vulnerability_score(2.0, 1.0) == 0.5
    assert vulnerability_score(2.0, 3.0) == 0.0
    assert vulnerability_score(2.0, 0.0) == 1.0
    assert vulnerability_score(0.0, 0.0) == 0.0


def test_cache_keys_isolate_methods_but_share_vulnerability_stage():
    qig = vg_config(MethodName.VG_QIG_MBQ)
    uniform = dataclasses.replace(qig, method=MethodName.VG_UNIFORM_MBQ)
    kwargs = {"model_revision": "m", "calibration_digest": "c", "selected_names": ["x"]}
    assert vg_cache_key(config=qig, **kwargs) != vg_cache_key(config=uniform, **kwargs)
    assert vg_vulnerability_cache_key(
        config=qig, **kwargs,
    ) == vg_vulnerability_cache_key(config=uniform, **kwargs)


def test_auto_gate_is_source_balanced_and_has_empty_fallback():
    scores = {
        0: [("a", "x", 0.8), ("b", "y", 0.8)],
        1: [("a", "x", 0.1), ("b", "y", 0.1)],
    }
    chosen, audit = auto_gate_layers(
        scores, relative_threshold=0.5, max_layers=1,
        bootstrap_samples=50, inclusion_probability=0.5, seed=7,
    )
    assert chosen == [0]
    assert audit["inclusion_probability"][0] == 1.0
    empty, empty_audit = auto_gate_layers(
        {0: [("a", "x", 0.0)]}, relative_threshold=0.05, max_layers=4,
        bootstrap_samples=10, inclusion_probability=0.5, seed=1,
    )
    assert empty == [] and empty_audit["empty_gate_fallback"]


def test_zero_discrepancy_qig_is_finite_and_modality_normalized():
    block = ToyBlock().eval()
    item = trace()
    weights = qig_token_importance(block, block, item, 2, iqr_multiplier=1.5)
    assert torch.isfinite(weights["vision"]).all()
    assert torch.isclose(weights["vision"].sum(), torch.tensor(1.0))
    assert torch.isclose(weights["answer_language"].sum(), torch.tensor(1.0))
    assert not weights["vision"][~item.vision_mask].any()


def test_adaptive_step_selection_and_fallback():
    vector = torch.arange(10).float()
    chosen, audit = choose_qig_steps(
        {2: [vector], 4: [vector.clone()], 8: [vector.flip(0)]},
        rank_threshold=0.98, jaccard_threshold=0.9,
    )
    assert chosen == 4 and audit["converged"]
    chosen, audit = choose_qig_steps(
        {2: [vector], 4: [vector.flip(0)]},
        rank_threshold=0.98, jaccard_threshold=0.9,
    )
    assert chosen == 4 and not audit["converged"]


def test_block_search_base_candidate_guarantees_non_regression():
    torch.manual_seed(3)
    block = ToyBlock().eval()
    modules = selected(block)
    scales = {item.name: torch.ones(item.module.in_features) for item in modules}
    result, audit = search_vulnerable_block(
        block, modules, scales, [trace()], None, vg_config(),
    )
    assert set(result) == set(scales)
    assert audit["non_regression"]
    assert audit["final_loss"] <= audit["initial_loss"] + 1e-12


def test_sample_normalized_objective_is_invariant_to_token_duplication():
    torch.manual_seed(4)
    block = ToyBlock().eval()
    modules = selected(block)
    scales = {item.name: torch.ones(item.module.in_features) for item in modules}
    quant = make_quantized_block(block, modules, scales, vg_config())
    original = trace()
    duplicated = BlockTrace(
        original.sample_id, original.source,
        original.hidden.repeat_interleave(2, dim=1), {},
        original.vision_mask.repeat_interleave(2, dim=1),
        original.answer_mask.repeat_interleave(2, dim=1),
    )
    first, _ = block_objective(block, quant, [original], None)
    second, _ = block_objective(block, quant, [duplicated], None)
    assert abs(first - second) < 1e-7


def test_w4a8_vg_config_requires_symmetric_per_output_channel_weights():
    config = ExperimentConfig(
        MethodName.VG_QIG_MBQ,
        weight_bits=4,
        activation_bits=8,
        weight_granularity="per_output_channel",
        zero_point=False,
        implementation_tier=ImplementationTier.CPU_RESEARCH,
    )
    assert config.scheme_id == "w4a8_g128_sym"
    assert config.weight_granularity == "per_output_channel"
    assert config.zero_point is False


def test_w4a8_quantized_block_applies_dynamic_activation_qdq():
    torch.manual_seed(5)
    block = ToyBlock().eval()
    modules = selected(block)
    scales = {item.name: torch.ones(item.module.in_features) for item in modules}
    config = ExperimentConfig(
        MethodName.VG_QIG_MBQ,
        weight_bits=4,
        activation_bits=8,
        weight_granularity="per_output_channel",
        zero_point=False,
        group_size=4,
        implementation_tier=ImplementationTier.CPU_RESEARCH,
    )
    quantized = make_quantized_block(block, modules, scales, config)
    assert all(
        isinstance(module, W8A8Linear)
        for module in quantized.modules()
        if isinstance(module, W8A8Linear)
    )
    assert sum(isinstance(module, W8A8Linear) for module in quantized.modules()) == 7


def test_w2a16_vg_config_uses_asymmetric_groupwise_weights():
    config = ExperimentConfig(
        MethodName.VG_QIG_MBQ,
        weight_bits=2,
        implementation_tier=ImplementationTier.CPU_RESEARCH,
    )
    assert config.scheme_id == "w2a16_g128_asym"
    assert config.activation_bits is None
    assert config.weight_granularity == "groupwise"
    assert config.zero_point is True
