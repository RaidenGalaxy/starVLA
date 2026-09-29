import dataclasses
import math
from pathlib import Path

import pytest
import torch

from experiments.vlm_quant_benchmark.analyze_sensitivity_correlation import (
    _bh_fdr,
    attribution_decision,
    build_quant_degradation,
    sample_sensitivity_features,
    summarize_sensitivity,
)
from experiments.vlm_quant_benchmark.config import ExperimentConfig, MethodName
from experiments.vlm_quant_benchmark.data import MultimodalSample
from experiments.vlm_quant_benchmark.mbq import (
    ModalityActivationStats,
    _candidate_error,
    aggregate_gradient_ratios,
    mbq_cache_key,
    mbq_statistics_cache_key,
)
from experiments.vlm_quant_benchmark.promotion import rank_promotion_variants
from experiments.vlm_quant_benchmark.residuals import (
    aggregate_mbq_block_rms,
    capture_reconstruction_outputs,
    prediction_change_summary,
    reconstruction_rows,
    residual_summary,
)
from experiments.vlm_quant_benchmark.run_sensitivity_ablation import ablation_configs
from experiments.vlm_quant_benchmark.run_mbq_trimodal_formal import (
    _paired_effect,
    trimodal_configs,
)
from experiments.vlm_quant_benchmark.run_mbq_factorial_formal import (
    factorial_configs,
    factorial_contrast_values,
)
from experiments.vlm_quant_benchmark.second_wave_methods import squeezellm_linear
from experiments.vlm_quant_benchmark.calibration import ActivationStats
from experiments.vlm_quant_benchmark.selection import select_linears
from experiments.vlm_quant_benchmark.selection import SelectedLinear
from experiments.vlm_quant_benchmark.config import QuantScope
from experiments.vlm_quant_benchmark.sensitivity import (
    causal_predictor_mask,
    collect_modality_perturbations,
    collect_sample_sensitivity,
    stable_hash_subset,
    vision_mass_share,
)
from experiments.vlm_quant_benchmark.tests.test_mbq import ToyMBQVLM, toy_batch


def _sample(dataset: str, identifier: int) -> MultimodalSample:
    return MultimodalSample(
        id=f"{dataset}:{identifier}",
        dataset=dataset,
        split="test",
        image=Path(__file__),
        prompt="question",
        target="answer",
    )


def test_stable_hash_subsets_are_reproducible_and_disjoint():
    samples = [
        _sample(dataset, index)
        for dataset in ("a", "b")
        for index in range(12)
    ]
    first = stable_hash_subset(samples, 4, salt="one")
    assert [item.id for item in first] == [
        item.id for item in stable_hash_subset(samples, 4, salt="one")
    ]
    second = stable_hash_subset(
        samples, 4, salt="two",
        exclude_ids={item.id for item in first},
    )
    assert not ({item.id for item in first} & {item.id for item in second})


def test_gradient_aggregation_and_effective_mass_are_explicit():
    records = {
        "layer": [
            {
                "source": "a",
                "vision_sum": 100.0,
                "vision_count": 100,
                "vision_mean": 1.0,
                "language_sum": 10.0,
                "language_count": 1,
                "language_mean": 10.0,
            },
            {
                "source": "b",
                "vision_sum": 4.0,
                "vision_count": 1,
                "vision_mean": 4.0,
                "language_sum": 1.0,
                "language_count": 1,
                "language_mean": 1.0,
            },
        ],
    }
    pooled = aggregate_gradient_ratios(records, "pooled_elements")["layer"]
    sample = aggregate_gradient_ratios(records, "sample_mean")["layer"]
    balanced = aggregate_gradient_ratios(records, "source_balanced")["layer"]
    assert pooled != sample
    assert sample == balanced
    assert vision_mass_share(0.067, 102726, 551) == pytest.approx(0.9259, abs=1e-3)


def test_causal_predictor_mask_shifts_labels_left():
    labels = torch.tensor([[False, False, False, True, True, False]])
    shifted = causal_predictor_mask(labels)
    assert shifted.tolist() == [[False, False, True, True, False, False]]
    assert int(shifted.sum()) == int(labels.sum())


def test_mbq_objective_uses_unweighted_normalizer():
    torch.manual_seed(8)
    linear = torch.nn.Linear(4, 3, bias=False)
    selected = SelectedLinear(
        "model.layers.0.self_attn.o_proj",
        linear,
        "decoder_attention",
        linear.weight.numel(),
    )
    stats = ModalityActivationStats(
        vision=torch.randn(2, 4),
        language=torch.randn(1, 4),
        vision_rows=20,
        language_rows=3,
    )
    base = ExperimentConfig(
        MethodName.MBQ,
        group_size=4,
        mbq_objective_normalization="token_weighted",
    )
    loss, errors = _candidate_error(
        [selected], stats, torch.ones(4), 0.2, base,
    )
    expected = (
        0.2 * errors[selected.name]["vision"] * 20 * 3
        + errors[selected.name]["language"] * 3 * 3
    ) / (20 * 3 + 3 * 3)
    assert loss == pytest.approx(expected)

    normalized, errors = _candidate_error(
        [selected],
        stats,
        torch.ones(4),
        0.2,
        dataclasses.replace(base, mbq_objective_normalization="modality_mean"),
    )
    expected = (
        0.2 * errors[selected.name]["vision"]
        + errors[selected.name]["language"]
    ) / 2
    assert normalized == pytest.approx(expected)


def test_sensitivity_collection_and_perturbation_on_toy_model():
    torch.manual_seed(4)
    model = ToyMBQVLM()
    selected = select_linears(
        model, QuantScope.DECODER_CANONICAL, strict_qwen_3b=False,
    )
    batch = toy_batch() | {
        "source": "toy",
        "prompt_language_mask": torch.tensor(
            [[True, False, False, True, False, False]],
        ),
    }
    rows = collect_sample_sensitivity(
        model, selected, [batch], attention_implementation="eager",
    )
    assert len(rows) == 2
    assert {row["kind"] for row in rows} == {"attention", "mlp"}
    assert all(row["token_counts"]["prompt_language"] == 2 for row in rows)
    assert all("vision_to_answer_predictor_ratio" in row for row in rows)
    perturbations = collect_modality_perturbations(
        model, batch, layers_to_test=(0,), noise_rms=0.01, seed=42,
        attention_implementation="eager",
    )
    assert {row["modality"] for row in perturbations} == {
        "vision", "answer_language", "prompt_language",
    }
    assert all(math.isfinite(row["delta_nll"]) for row in perturbations)
    assert all(
        row["perturbation_location"] == "decoder_block_output"
        for row in perturbations
    )


def test_residual_capture_and_pooled_layer_rms_on_toy_model():
    torch.manual_seed(3)
    reference = ToyMBQVLM()
    candidate = ToyMBQVLM()
    candidate.load_state_dict(reference.state_dict())
    candidate.model.layers[0].mlp.down_proj.weight.data.add_(0.01)
    batch = toy_batch() | {
        "source": "toy",
        "prompt_language_mask": torch.tensor(
            [[True, False, False, True, False, False]],
        ),
    }
    selected = select_linears(
        reference, QuantScope.DECODER_CANONICAL, strict_qwen_3b=False,
    )
    anchors = [
        item.name for item in selected
        if item.name.endswith((".self_attn.o_proj", ".mlp.down_proj"))
    ]
    ref = capture_reconstruction_outputs(
        reference, batch, layers_to_test=(0,), anchor_names=anchors,
        attention_implementation="eager",
    )
    quant = capture_reconstruction_outputs(
        candidate, batch, layers_to_test=(0,), anchor_names=anchors,
        attention_implementation="eager",
    )
    rows = reconstruction_rows(batch, ref, quant, method="mbq_current")
    assert len(rows) == 12
    layer_rms = aggregate_mbq_block_rms(rows)
    assert set(layer_rms) == {0}
    assert layer_rms[0] > 0
    zeros = residual_summary(ref["block:0"], ref["block:0"], batch["vision_mask"])
    assert zeros["rmse"] == 0


def test_mbq_raw_statistics_key_is_shared_across_search_variants():
    base = ExperimentConfig(MethodName.MBQ, scheme_id="base")
    changed = dataclasses.replace(
        base, mbq_gradient_aggregation="sample_mean", scheme_id="sample",
    )
    common = {
        "model_revision": "model",
        "calibration_digest": "calibration",
        "selected_names": ["model.layers.0.self_attn.q_proj"],
        "reservoir_rows": 32,
        "attention_implementation": "sdpa",
    }
    raw_kwargs = common | {
        "seed": 42,
        "min_pixels": base.min_pixels,
        "max_pixels": base.max_pixels,
    }
    assert mbq_statistics_cache_key(**raw_kwargs) == mbq_statistics_cache_key(**raw_kwargs)
    assert mbq_cache_key(config=base, **common) != mbq_cache_key(config=changed, **common)


def test_mbq_ablation_modes_are_cache_isolated():
    base = ExperimentConfig(MethodName.MBQ, scheme_id="base")
    changed = ExperimentConfig(
        MethodName.MBQ,
        mbq_gradient_aggregation="sample_mean",
        scheme_id="sample",
    )
    kwargs = {
        "model_revision": "model",
        "calibration_digest": "calibration",
        "selected_names": ["model.layers.0.self_attn.q_proj"],
        "reservoir_rows": 32,
        "attention_implementation": "sdpa",
    }
    assert mbq_cache_key(config=base, **kwargs) != mbq_cache_key(
        config=changed, **kwargs,
    )
    variants = ablation_configs(calibration_samples=128, seed=42)
    names = {item[0] for item in variants}
    assert {
        "mbq_current", "mbq_no_reweight", "mbq_sample_average",
        "mbq_modality_normalized", "mbq_no_median_floor",
        "mbq_block_reconstruction", "squeezellm_current",
        "squeezellm_no_sparse", "mbq_target_oracle",
    } <= names


def test_squeezellm_zero_sparse_fraction_has_no_corrections():
    source = torch.nn.Linear(8, 4, bias=False)
    reservoir = torch.randn(4, 8)
    stats = ActivationStats(
        abs_sum=reservoir.abs().sum(0),
        abs_max=reservoir.abs().amax(0),
        rows=4,
        reservoir=reservoir,
    )
    config = ExperimentConfig(
        MethodName.SQUEEZELLM,
        group_size=4,
        squeezellm_sparse_fraction=0.0,
    )
    wrapper, audit = squeezellm_linear(source, stats, config)
    assert audit["outlier_fraction"] == 0
    assert torch.isfinite(wrapper(torch.randn(2, 8))).all()


def test_sensitivity_summaries_and_fdr():
    rows = []
    for kind in ("calibration", "target"):
        for index in range(6):
            rows.append({
                "sample_kind": kind,
                "dataset": "target" if kind == "target" else "source",
                "id": f"{kind}:{index}",
                "layer": 0,
                "kind": "attention",
                "vision_to_answer_ratio": 0.1 + index * 0.01,
                "vision_to_prompt_ratio": 0.2,
                "vision_mass_share": 0.8,
            })
    summaries = summarize_sensitivity(rows, bootstrap_samples=50, seed=42)
    assert len(summaries) == 2
    features = sample_sensitivity_features(rows)
    assert len(features) == 6
    adjusted = _bh_fdr([0.01, 0.04, 0.03])
    assert adjusted[0] <= adjusted[2]
    assert all(0 <= value <= 1 for value in adjusted)


def test_build_quant_degradation_joins_exact_sample_ids(tmp_path):
    root = tmp_path / "confirm"
    for run in ("bf16_value_fp32_compute", "mbq_w4a16_g128_asym"):
        dataset = root / run / "MMMU_DEV_VAL"
        dataset.mkdir(parents=True)
        score = 1.0 if run.startswith("bf16") else 0.0
        (dataset / "official_samples.jsonl").write_text(
            '{"id":"MMMU_DEV_VAL:1","dataset":"MMMU_DEV_VAL",'
            f'"official_score":{score}}}\n'
        )
    rows = build_quant_degradation(root)
    mbq = next(row for row in rows if row["method"] == "mbq")
    assert mbq["score_delta_vs_bf16"] == -100


def test_promotion_ranking_is_balanced_reproducible_and_excludes_oracle():
    rows = []
    for dataset in ("a", "b", "c", "d"):
        for index in range(3):
            identifier = f"{dataset}:{index}"
            rows.extend([
                {"variant": "mbq_current", "dataset": dataset, "id": identifier,
                 "target_nll": 2.0, "evaluation_split": "ablation"},
                {"variant": "mbq_sample_average", "dataset": dataset, "id": identifier,
                 "target_nll": 1.8, "evaluation_split": "ablation"},
                {"variant": "mbq_no_reweight", "dataset": dataset, "id": identifier,
                 "target_nll": 1.9, "evaluation_split": "ablation"},
                {"variant": "mbq_target_oracle", "dataset": dataset, "id": identifier,
                 "target_nll": 1.0, "evaluation_split": "ablation"},
            ])
    first = rank_promotion_variants(rows, bootstrap_samples=50, seed=42)
    second = rank_promotion_variants(rows, bootstrap_samples=50, seed=42)
    assert first == second
    assert first[0]["variant"] == "mbq_sample_average"
    assert all(row["variant"] != "mbq_target_oracle" for row in first)


def test_attribution_keeps_indistinguishable_w4_conclusion():
    datasets = [
        {"dataset": name, "motivation_supported": True}
        for name in ("a", "b", "c", "d")
    ]
    perturbations = []
    for dataset in ("a", "b", "c", "d"):
        for index in range(5):
            common = {"dataset": dataset, "id": f"{dataset}:{index}", "layer": 0}
            perturbations.extend([
                common | {"modality": "vision", "delta_nll": 0.01},
                common | {"modality": "answer_language", "delta_nll": 0.1},
            ])
    decision = attribution_decision(
        datasets, perturbations, [], [],
        {"mean": 0.0, "ci_low": -0.2, "ci_high": 0.2},
        [],
        bootstrap_samples=50, seed=42,
    )
    assert decision["motivation_supported"]
    assert decision["w4_paired_ci_contains_zero"]
    assert decision["conclusion"] == "没有证据证明MBQ整体泛化更差"



def test_trimodal_formal_configs_share_raw_cache_but_isolate_search():
    configs = trimodal_configs(calibration_samples=128, seed=42)
    no_floor = configs["mbq_trimodal_no_floor"]
    floor = configs["mbq_trimodal_floor"]
    assert no_floor.mbq_modality_mode == "vision_prompt_answer"
    assert floor.mbq_algorithm_version == 5
    assert not no_floor.mbq_median_floor and floor.mbq_median_floor
    raw_kwargs = {
        "model_revision": "model",
        "calibration_digest": "calibration",
        "selected_names": ["model.layers.0.self_attn.q_proj"],
        "reservoir_rows": 32,
        "seed": 42,
        "min_pixels": no_floor.min_pixels,
        "max_pixels": no_floor.max_pixels,
        "attention_implementation": "sdpa",
        "modality_mode": no_floor.mbq_modality_mode,
        "algorithm_version": no_floor.mbq_algorithm_version,
    }
    assert mbq_statistics_cache_key(**raw_kwargs) == mbq_statistics_cache_key(**raw_kwargs)
    search_kwargs = {
        "model_revision": "model",
        "calibration_digest": "calibration",
        "selected_names": ["model.layers.0.self_attn.q_proj"],
        "reservoir_rows": 32,
        "attention_implementation": "sdpa",
    }
    assert mbq_cache_key(config=no_floor, **search_kwargs) != mbq_cache_key(
        config=floor, **search_kwargs,
    )


def test_trimodal_paired_effect_is_dataset_balanced_and_reproducible():
    baseline = {}
    rows = []
    for dataset in ("a", "b", "c", "d"):
        for index in range(2):
            key = (dataset, f"{dataset}:{index}")
            baseline[key] = 2.0 + index
            rows.append({
                "dataset": dataset,
                "id": key[1],
                "target_nll": baseline[key] - 0.1,
            })
    first = _paired_effect(
        baseline,
        rows,
        bootstrap_samples=50,
        seed=42,
        variant="tri",
        expected_per_dataset=2,
    )
    second = _paired_effect(
        baseline,
        rows,
        bootstrap_samples=50,
        seed=42,
        variant="tri",
        expected_per_dataset=2,
    )
    assert first == second
    assert first["dataset_balanced_nll_improvement"] == pytest.approx(0.1)
    assert first["paired_samples"] == 8



def test_factorial_configs_cover_exact_2x2_without_floor():
    configs = factorial_configs(calibration_samples=128, seed=42)
    assert {
        (
            config.effective_mbq_activation_mode,
            config.effective_mbq_objective_mode,
        )
        for config in configs.values()
    } == {
        ("vision_answer", "vision_answer"),
        ("vision_prompt_answer", "vision_answer"),
        ("vision_answer", "vision_prompt_answer"),
        ("vision_prompt_answer", "vision_prompt_answer"),
    }
    assert all(
        config.mbq_modality_mode == "vision_prompt_answer"
        and config.mbq_algorithm_version == 5
        and not config.mbq_median_floor
        for config in configs.values()
    )


def test_factorial_contrast_values_have_registered_sign_convention():
    effects = factorial_contrast_values({
        "A_VA__O_VA": 1.0,
        "A_VPA__O_VA": 2.0,
        "A_VA__O_VPA": 3.0,
        "A_VPA__O_VPA": 7.0,
    })
    assert effects == {
        "activation_main_effect": 2.5,
        "objective_main_effect": 3.5,
        "activation_objective_interaction": 3.0,
        "trimodal_endpoint_effect": 6.0,
    }


def test_prediction_change_summary_reports_kl_and_argmax_rate():
    reference = torch.tensor([[
        [3.0, 0.0], [0.0, 3.0], [2.0, 0.0], [0.0, 2.0],
    ]])
    identical = prediction_change_summary(
        reference, reference.clone(), torch.tensor([[False, False, True, True]]),
    )
    assert identical["output_kl"] == pytest.approx(0.0)
    assert identical["argmax_change_rate"] == pytest.approx(0.0)
    candidate = reference.clone()
    candidate[:, 1] = torch.tensor([4.0, 0.0])
    changed = prediction_change_summary(
        reference, candidate, torch.tensor([[False, False, True, True]]),
    )
    assert changed["output_kl"] > 0
    assert changed["argmax_change_rate"] > 0
