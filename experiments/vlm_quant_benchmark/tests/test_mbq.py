import dataclasses
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from experiments.vlm_quant_benchmark.config import ExperimentConfig, MethodName, QuantScope
from experiments.vlm_quant_benchmark.mbq import (
    MBQSearchResult,
    ModalityActivationStats,
    aggregate_gradient_weights,
    apply_mbq,
    collect_mbq_raw_statistics,
    collect_mbq_statistics,
    load_mbq_raw_statistics,
    load_mbq_search,
    mbq_cache_key,
    mbq_statistics_cache_key,
    save_mbq_raw_statistics,
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



def trimodal_config(*, median_floor=False):
    return dataclasses.replace(
        mbq_config(),
        mbq_modality_mode="vision_prompt_answer",
        mbq_algorithm_version=5,
        mbq_median_floor=median_floor,
        scheme_id="toy_trimodal_w4a16",
    )


def trimodal_batch():
    return toy_batch() | {
        "source": "toy",
        "answer_language_mask": toy_batch()["language_mask"],
        "prompt_language_mask": torch.tensor(
            [[True, False, False, True, False, False]],
        ),
    }


def test_trimodal_configuration_is_explicit_and_versioned():
    trimodal_config()
    with pytest.raises(ValueError, match="version 5"):
        dataclasses.replace(
            mbq_config(), mbq_modality_mode="vision_prompt_answer",
        )
    with pytest.raises(ValueError, match="modality mode"):
        dataclasses.replace(mbq_config(), mbq_modality_mode="unknown")


def test_trimodal_collection_and_gradient_weights():
    torch.manual_seed(13)
    model = ToyMBQVLM()
    selected = select_linears(
        model, QuantScope.DECODER_CANONICAL, strict_qwen_3b=False,
    )
    raw = collect_mbq_raw_statistics(
        model,
        selected,
        [trimodal_batch()],
        reservoir_rows=4,
        seed=42,
        attention_implementation="eager",
        modality_mode="vision_prompt_answer",
        algorithm_version=5,
    )
    assert raw.modality_mode == "vision_prompt_answer"
    assert raw.token_counts == {
        "vision": 2,
        "prompt_language": 2,
        "answer_language": 2,
        "samples": 1,
    }
    assert all(value.prompt_language is not None for value in raw.stats.values())
    weights = aggregate_gradient_weights(
        raw.gradient_records, "pooled_elements",
    )
    assert len(weights) == 2
    assert all(value["answer_language"] == 1.0 for value in weights.values())
    assert all(
        set(value) == {"vision", "prompt_language", "answer_language"}
        and all(torch.isfinite(torch.tensor(item)) for item in value.values())
        for value in weights.values()
    )


def test_trimodal_search_objective_and_audit_are_modality_explicit():
    torch.manual_seed(17)
    model = ToyMBQVLM()
    selected = select_linears(
        model, QuantScope.DECODER_CANONICAL, strict_qwen_3b=False,
    )
    stats = {
        item.name: ModalityActivationStats(
            vision=torch.randn(4, item.module.in_features),
            language=torch.randn(3, item.module.in_features),
            vision_rows=40,
            language_rows=3,
            prompt_language=torch.randn(2, item.module.in_features),
            prompt_language_rows=12,
        )
        for item in selected
    }
    raw_weights = {
        "model.layers.0.self_attn.o_proj": {
            "vision": 0.1, "prompt_language": 0.7, "answer_language": 1.0,
        },
        "model.layers.0.mlp.down_proj": {
            "vision": 0.2, "prompt_language": 1.3, "answer_language": 1.0,
        },
    }
    result = search_mbq_scales(
        selected,
        stats,
        raw_weights,
        trimodal_config(),
        {"vision": 40, "prompt_language": 12, "answer_language": 3, "samples": 1},
    )
    assert result.audit["modality_mode"] == "vision_prompt_answer"
    assert result.audit["reference_modality"] == "answer_language"
    for group in result.audit["groups"].values():
        assert set(group["effective_modality_weights"]) == {
            "vision", "prompt_language", "answer_language",
        }
        assert sum(group["modality_mass_share"].values()) == pytest.approx(1.0)
        for history in group["history"]:
            for errors in history["module_mae"].values():
                assert set(errors) == {
                    "vision", "prompt_language", "answer_language",
                }


def test_trimodal_raw_cache_isolated_and_round_trips(tmp_path):
    torch.manual_seed(19)
    model = ToyMBQVLM()
    selected = select_linears(
        model, QuantScope.DECODER_CANONICAL, strict_qwen_3b=False,
    )
    common = {
        "model_revision": "model",
        "calibration_digest": "calibration",
        "selected_names": [item.name for item in selected],
        "reservoir_rows": 4,
        "seed": 42,
        "min_pixels": 1,
        "max_pixels": 2,
        "attention_implementation": "eager",
    }
    legacy_key = mbq_statistics_cache_key(**common)
    trimodal_key = mbq_statistics_cache_key(
        **common,
        modality_mode="vision_prompt_answer",
        algorithm_version=5,
    )
    assert legacy_key != trimodal_key
    raw = collect_mbq_raw_statistics(
        model,
        selected,
        [trimodal_batch()],
        reservoir_rows=4,
        seed=42,
        attention_implementation="eager",
        modality_mode="vision_prompt_answer",
        algorithm_version=5,
    )
    cache = tmp_path / "trimodal.pt"
    save_mbq_raw_statistics(cache, key=trimodal_key, raw=raw)
    assert load_mbq_raw_statistics(
        cache,
        key=trimodal_key,
        selected=selected,
        modality_mode="vision_prompt_answer",
        algorithm_version=5,
    ) is not None
    assert load_mbq_raw_statistics(
        cache, key=trimodal_key, selected=selected,
    ) is None



def factorial_config(activation_mode: str, objective_mode: str):
    return dataclasses.replace(
        trimodal_config(),
        mbq_activation_mode=activation_mode,
        mbq_objective_mode=objective_mode,
        scheme_id=f"toy_factorial_{activation_mode}_{objective_mode}",
    )


def test_factorial_config_defaults_preserve_legacy_serialization_and_validate():
    legacy = mbq_config()
    assert "mbq_activation_mode" not in legacy.to_dict()
    assert "mbq_objective_mode" not in legacy.to_dict()
    assert legacy.effective_mbq_activation_mode == "vision_answer"
    assert legacy.effective_mbq_objective_mode == "vision_answer"
    with pytest.raises(ValueError, match="raw statistics"):
        dataclasses.replace(
            legacy,
            mbq_activation_mode="vision_prompt_answer",
        )


def test_factorial_search_splits_activation_and_objective_axes():
    torch.manual_seed(29)
    model = ToyMBQVLM()
    selected = select_linears(
        model, QuantScope.DECODER_CANONICAL, strict_qwen_3b=False,
    )
    stats = {
        item.name: ModalityActivationStats(
            vision=torch.full((4, item.module.in_features), 0.5),
            language=torch.full((3, item.module.in_features), 1.0),
            vision_rows=40,
            language_rows=3,
            prompt_language=torch.arange(
                2 * item.module.in_features, dtype=torch.float32,
            ).reshape(2, item.module.in_features) + 10,
            prompt_language_rows=12,
        )
        for item in selected
    }
    weights = {
        "model.layers.0.self_attn.o_proj": {
            "vision": 0.1, "prompt_language": 0.7, "answer_language": 1.0,
        },
        "model.layers.0.mlp.down_proj": {
            "vision": 0.2, "prompt_language": 1.3, "answer_language": 1.0,
        },
    }
    modes = ("vision_answer", "vision_prompt_answer")
    results = {
        (activation, objective): search_mbq_scales(
            selected,
            stats,
            weights,
            factorial_config(activation, objective),
            {
                "vision": 40,
                "prompt_language": 12,
                "answer_language": 3,
                "samples": 1,
            },
        )
        for activation in modes
        for objective in modes
    }
    for activation in modes:
        left = results[(activation, "vision_answer")].audit["groups"]
        right = results[(activation, "vision_prompt_answer")].audit["groups"]
        assert {
            name: group["activation_digest"] for name, group in left.items()
        } == {
            name: group["activation_digest"] for name, group in right.items()
        }
    for objective in modes:
        left = results[("vision_answer", objective)].audit["groups"]
        right = results[("vision_prompt_answer", objective)].audit["groups"]
        assert all(
            left[name]["activation_digest"] != right[name]["activation_digest"]
            for name in left
        )
        assert all(
            left[name]["history"][0]["weighted_mae"]
            == pytest.approx(right[name]["history"][0]["weighted_mae"])
            for name in left
        )
    two_objective = results[("vision_prompt_answer", "vision_answer")]
    tri_objective = results[("vision_prompt_answer", "vision_prompt_answer")]
    for group in two_objective.audit["groups"].values():
        assert set(group["effective_modality_weights"]) == {
            "vision", "answer_language",
        }
        assert all(
            set(errors) == {"vision", "language"}
            for point in group["history"]
            for errors in point["module_mae"].values()
        )
    for group in tri_objective.audit["groups"].values():
        assert set(group["effective_modality_weights"]) == {
            "vision", "prompt_language", "answer_language",
        }


def test_factorial_search_cache_keys_are_isolated():
    configs = [
        factorial_config(activation, objective)
        for activation in ("vision_answer", "vision_prompt_answer")
        for objective in ("vision_answer", "vision_prompt_answer")
    ]
    keys = {
        mbq_cache_key(
            model_revision="model",
            calibration_digest="calibration",
            selected_names=["model.layers.0.self_attn.q_proj"],
            config=config,
            reservoir_rows=4,
            attention_implementation="eager",
        )
        for config in configs
    }
    assert len(keys) == 4
from experiments.vlm_quant_benchmark.run_mbq_ava_objective_formal import (
    CONTROL,
    P100,
    VARIANT_SPECS,
    objective_configs,
)
from experiments.vlm_quant_benchmark.run_mbq_factorial_formal import factorial_configs


def test_prompt_objective_scale_validates_and_preserves_default_serialization():
    base = factorial_config("vision_answer", "vision_prompt_answer")
    assert "mbq_prompt_objective_scale" not in base.to_dict()
    scaled = dataclasses.replace(base, mbq_prompt_objective_scale=0.25)
    assert scaled.to_dict()["mbq_prompt_objective_scale"] == pytest.approx(0.25)
    with pytest.raises(ValueError, match="finite and non-negative"):
        dataclasses.replace(base, mbq_prompt_objective_scale=-0.1)
    with pytest.raises(ValueError, match="finite and non-negative"):
        dataclasses.replace(base, mbq_prompt_objective_scale=float("inf"))
    with pytest.raises(ValueError, match="prompt-aware objective"):
        dataclasses.replace(
            factorial_config("vision_answer", "vision_answer"),
            mbq_prompt_objective_scale=0.25,
        )
    common = {
        "model_revision": "model",
        "calibration_digest": "calibration",
        "selected_names": ["model.layers.0.self_attn.q_proj"],
        "reservoir_rows": 4,
        "attention_implementation": "eager",
    }
    assert mbq_cache_key(config=base, **common) != mbq_cache_key(
        config=scaled, **common,
    )


def test_prompt_objective_scale_changes_only_prompt_weight_and_keeps_ava():
    torch.manual_seed(41)
    model = ToyMBQVLM()
    selected = select_linears(
        model, QuantScope.DECODER_CANONICAL, strict_qwen_3b=False,
    )
    stats = {
        item.name: ModalityActivationStats(
            vision=torch.randn(4, item.module.in_features),
            language=torch.randn(3, item.module.in_features),
            vision_rows=40,
            language_rows=3,
            prompt_language=torch.randn(2, item.module.in_features),
            prompt_language_rows=12,
        )
        for item in selected
    }
    weights = {
        "model.layers.0.self_attn.o_proj": {
            "vision": 0.1, "prompt_language": 0.8, "answer_language": 1.0,
        },
        "model.layers.0.mlp.down_proj": {
            "vision": 0.2, "prompt_language": 1.2, "answer_language": 1.0,
        },
    }
    base = dataclasses.replace(
        factorial_config("vision_answer", "vision_prompt_answer"),
        mbq_median_floor=False,
    )
    scaled = dataclasses.replace(base, mbq_prompt_objective_scale=0.25)
    token_counts = {
        "vision": 40, "prompt_language": 12,
        "answer_language": 3, "samples": 1,
    }
    full = search_mbq_scales(selected, stats, weights, base, token_counts)
    quarter = search_mbq_scales(selected, stats, weights, scaled, token_counts)
    for name, group in quarter.audit["groups"].items():
        reference = full.audit["groups"][name]
        assert group["activation_modalities"] == ["vision", "answer_language"]
        assert group["activation_digest"] == reference["activation_digest"]
        assert group["effective_modality_weights"]["vision"] == pytest.approx(
            reference["effective_modality_weights"]["vision"]
        )
        assert group["effective_modality_weights"]["answer_language"] == pytest.approx(
            reference["effective_modality_weights"]["answer_language"]
        )
        assert group["effective_modality_weights"]["prompt_language"] == pytest.approx(
            reference["effective_modality_weights"]["prompt_language"] * 0.25
        )
        assert group["pre_prompt_scale_modality_weights"] == (
            reference["pre_prompt_scale_modality_weights"]
        )
        assert sum(group["modality_mass_share"].values()) == pytest.approx(1.0)


def test_fixed_ava_objective_matrix_is_locked_and_endpoints_are_compatible():
    configs = objective_configs(calibration_samples=128, seed=42)
    assert list(configs) == list(VARIANT_SPECS)
    assert set(configs) == {
        CONTROL,
        "ava_ovpa_tw_p0125",
        "ava_ovpa_tw_p025",
        "ava_ovpa_tw_p050",
        P100,
        "ava_ovpa_mm_p025",
        "ava_ovpa_mm_p050",
        "ava_ovpa_mm_p100",
    }
    assert all(
        config.effective_mbq_activation_mode == "vision_answer"
        and config.mbq_modality_mode == "vision_prompt_answer"
        and not config.mbq_median_floor
        and config.mbq_gradient_aggregation == "pooled_elements"
        for config in configs.values()
    )
    factorial = factorial_configs(calibration_samples=128, seed=42)
    assert configs[CONTROL].digest == factorial["A_VA__O_VA"].digest
    assert configs[P100].digest == factorial["A_VA__O_VPA"].digest
