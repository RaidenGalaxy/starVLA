from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from experiments.vlm_quant_benchmark.config import ExperimentConfig, MethodName, QuantScope
from experiments.vlm_quant_benchmark.mbq import (
    MBQSearchResult,
    ModalityActivationStats,
    apply_mbq,
    collect_mbq_statistics,
    load_mbq_search,
    mbq_cache_key,
    save_mbq_search,
    search_mbq_scales,
)
from experiments.vlm_quant_benchmark.selection import select_linears
from experiments.vlm_quant_benchmark.run_mbq_comparison import comparison_configs
from experiments.vlm_quant_benchmark.run_staged import LOCAL_METHODS, SCHEMES


class ToyAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=True)
        self.k_proj = nn.Linear(width, width, bias=True)
        self.v_proj = nn.Linear(width, width, bias=True)
        self.o_proj = nn.Linear(width, width, bias=False)

    def forward(self, value):
        mixed = self.q_proj(value) + self.k_proj(value) + self.v_proj(value)
        mixed = mixed + mixed.mean(dim=1, keepdim=True)
        return self.o_proj(torch.tanh(mixed))


class ToyMLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gate_proj = nn.Linear(width, width * 2, bias=False)
        self.up_proj = nn.Linear(width, width * 2, bias=False)
        self.down_proj = nn.Linear(width * 2, width, bias=False)

    def forward(self, value):
        return self.down_proj(F.silu(self.gate_proj(value)) * self.up_proj(value))


class ToyBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.self_attn = ToyAttention(width)
        self.mlp = ToyMLP(width)

    def forward(self, value):
        value = value + self.self_attn(value)
        return value + self.mlp(value)


class ToyMBQVLM(nn.Module):
    def __init__(self, width=8, vocab=19):
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([ToyBlock(width)])
        self.lm_head = nn.Linear(width, vocab, bias=False)

    def forward(self, input_ids, labels=None, **_kwargs):
        hidden = self.embed(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(loss=loss, logits=logits)


def mbq_config(*, activation_bits=None, weight_bits=4):
    return ExperimentConfig(
        MethodName.MBQ,
        scope=QuantScope.DECODER_CANONICAL,
        weight_bits=weight_bits,
        activation_bits=activation_bits,
        group_size=4,
        zero_point=activation_bits is None,
        weight_granularity=(
            "per_output_channel" if activation_bits == 8 else "groupwise"
        ),
        mbq_scale_grid=(0.0, 0.5),
        scheme_id=f"toy_w{weight_bits}a{activation_bits or 16}",
    )


def toy_batch():
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    labels = torch.full_like(input_ids, -100)
    labels[:, 4:] = input_ids[:, 4:]
    return {
        "id": "toy",
        "model_inputs": {"input_ids": input_ids, "labels": labels},
        "vision_mask": torch.tensor([[False, True, True, False, False, False]]),
        "language_mask": labels.ne(-100),
    }


def test_mbq_configuration_gates():
    mbq_config()
    mbq_config(weight_bits=3)
    mbq_config(activation_bits=8)
    with pytest.raises(ValueError, match="A16 or A8"):
        ExperimentConfig(MethodName.RTN, activation_bits=4)
    with pytest.raises(ValueError, match="W3A16 only"):
        mbq_config(weight_bits=3, activation_bits=8)
    with pytest.raises(ValueError, match="decoder_canonical"):
        ExperimentConfig(MethodName.MBQ, scope=QuantScope.FULL_VLM_EXTENDED)
    with pytest.raises(ValueError, match="per-output-channel"):
        ExperimentConfig(
            MethodName.MBQ, activation_bits=8, zero_point=False,
            weight_granularity="groupwise",
        )


def test_mbq_collects_modal_activations_and_sft_gradient_ratios():
    torch.manual_seed(7)
    model = ToyMBQVLM()
    selected = select_linears(model, QuantScope.DECODER_CANONICAL, strict_qwen_3b=False)
    stats, ratios, counts = collect_mbq_statistics(
        model, selected, [toy_batch()], reservoir_rows=4, seed=42,
        attention_implementation="eager",
    )
    assert len(selected) == 7
    assert set(stats) == {item.name for item in selected}
    assert len(ratios) == 2
    assert counts == {
        "vision": 2,
        "language": 2,
        "samples": 1,
        "gradient_aggregation": "pooled_elements",
    }
    assert all(value >= 0 and torch.isfinite(torch.tensor(value)) for value in ratios.values())
    assert all(value.vision.shape[0] == 2 for value in stats.values())


@pytest.mark.parametrize("weight_bits,activation_bits", [(3, None), (4, None), (4, 8)])
def test_mbq_search_apply_and_restore_are_finite(weight_bits, activation_bits):
    torch.manual_seed(11)
    model = ToyMBQVLM()
    selected = select_linears(model, QuantScope.DECODER_CANONICAL, strict_qwen_3b=False)
    stats = {}
    for item in selected:
        stats[item.name] = ModalityActivationStats(
            vision=torch.randn(4, item.module.in_features),
            language=torch.randn(4, item.module.in_features),
            vision_rows=4,
            language_rows=4,
        )
    ratios = {
        "model.layers.0.self_attn.o_proj": 0.1,
        "model.layers.0.mlp.down_proj": 0.2,
    }
    config = mbq_config(activation_bits=activation_bits, weight_bits=weight_bits)
    result = search_mbq_scales(
        selected, stats, ratios, config,
        {"vision": 4, "language": 4, "samples": 1},
    )
    assert len(result.scales) == 7
    original = {item.name: item.module for item in selected}
    applied = apply_mbq(model, selected, config, result)
    output = model(input_ids=torch.tensor([[1, 2, 3]])).logits
    assert torch.isfinite(output).all()
    assert applied.audit["module_count"] == 7
    assert not applied.audit["deployment_kernel_claim"]
    applied.restore()
    assert all(model.get_submodule(name) is module for name, module in original.items())


def test_mbq_cache_key_and_round_trip(tmp_path):
    model = ToyMBQVLM()
    selected = select_linears(model, QuantScope.DECODER_CANONICAL, strict_qwen_3b=False)
    config = mbq_config()
    key = mbq_cache_key(
        model_revision="model-a",
        calibration_digest="calibration-a",
        selected_names=[item.name for item in selected],
        config=config,
        reservoir_rows=4,
        attention_implementation="sdpa",
    )
    different = mbq_cache_key(
        model_revision="model-a",
        calibration_digest="calibration-b",
        selected_names=[item.name for item in selected],
        config=config,
        reservoir_rows=4,
        attention_implementation="sdpa",
    )
    w3 = mbq_cache_key(
        model_revision="model-a",
        calibration_digest="calibration-a",
        selected_names=[item.name for item in selected],
        config=mbq_config(weight_bits=3),
        reservoir_rows=4,
        attention_implementation="sdpa",
    )
    assert key != different
    assert key != w3
    result = MBQSearchResult(
        scales={item.name: torch.ones(item.module.in_features) for item in selected},
        audit={"algorithm": "test"},
    )
    path = tmp_path / "mbq.pt"
    save_mbq_search(path, key=key, result=result)
    loaded = load_mbq_search(path, key=key, selected=selected)
    assert loaded is not None and loaded.audit == result.audit
    assert load_mbq_search(path, key=different, selected=selected) is None


def test_mbq_comparison_is_six_matched_opt_in_configs(tmp_path):
    calibration = tmp_path / "calibration.jsonl"
    evaluation = tmp_path / "evaluation.jsonl"
    calibration.write_text("{}\n")
    evaluation.write_text("{}\n")
    args = SimpleNamespace(
        preset="smoke",
        calibration_manifest=calibration,
        evaluation_manifest=evaluation,
        seed=42,
    )
    configs = comparison_configs(args)
    assert len(configs) == 5
    assert [config.method for config in configs] == [
        MethodName.RTN,
        MethodName.MBQ,
        MethodName.RTN,
        MethodName.SMOOTHQUANT,
        MethodName.MBQ,
    ]
    assert [config.activation_bits for config in configs] == [None, None, 8, 8, 8]
    assert not any(config.method == MethodName.AWQ for config in configs)
    assert "mbq" not in LOCAL_METHODS.split(",")
    assert not any(method == "mbq" for method, *_rest in SCHEMES)
