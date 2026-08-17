import json
import math

import pytest
import pandas as pd
import torch
from torch import nn

from experiments.vlm_quant_benchmark.calibration import (
    ActivationCollector,
    calibration_cache_key,
    collect_calibration_cached,
    load_calibration_cache,
)
from experiments.vlm_quant_benchmark.config import ExperimentConfig, MethodName
from experiments.vlm_quant_benchmark.corpus_evaluation import (
    evaluate_token_sequences_ppl,
    paired_window_delta,
)
from experiments.vlm_quant_benchmark.evaluation import target_token_span
from experiments.vlm_quant_benchmark.official_awq import (
    dequantize_awq_gemm,
    pack_awq_gemm,
    reference_model_name,
    unpack_awq_gemm,
)
from experiments.vlm_quant_benchmark.provenance import completed_run, mark_complete
from experiments.vlm_quant_benchmark.run_vlmeval_official import build_prompt, extract_scores, mathvista_auxeval_fixed
from experiments.vlm_quant_benchmark.result_analysis import (
    paired_score_difference,
    select_best_local_w4,
)
from experiments.vlm_quant_benchmark.selection import SelectedLinear


def test_target_mask_excludes_prompt_terminator_and_trailing_newline():
    prompt = torch.tensor([[10, 11, 12]])
    teacher = torch.tensor([[10, 11, 12, 20, 21, 13, 99, 13]])
    labels, positions = target_token_span(
        prompt,
        teacher,
        im_end_token_id=99,
        excluded_trailing_token_ids={13},
    )
    assert positions == [3, 4]
    assert labels.tolist() == [[-100, -100, -100, 20, 21, -100, -100, -100]]

    with pytest.raises(ValueError, match="prefix differs"):
        target_token_span(
            prompt,
            torch.tensor([[10, 77, 12, 20, 99]]),
            im_end_token_id=99,
        )


@pytest.mark.parametrize(
    "input_features,output_features,group_size",
    [(8, 16, 4), (12, 24, 4), (16, 8, 8)],
)
def test_awq_pack_unpack_and_dequantization(
    input_features, output_features, group_size,
):
    generator = torch.Generator().manual_seed(
        input_features * 100 + output_features
    )
    qweight_values = torch.randint(
        0, 16, (input_features, output_features),
        generator=generator, dtype=torch.int32,
    )
    qzero_values = torch.randint(
        0, 16, (input_features // group_size, output_features),
        generator=generator, dtype=torch.int32,
    )
    scales = torch.rand(
        input_features // group_size, output_features, generator=generator,
    ) + 0.01
    qweight = pack_awq_gemm(qweight_values)
    qzeros = pack_awq_gemm(qzero_values)
    torch.testing.assert_close(unpack_awq_gemm(qweight), qweight_values, rtol=0, atol=0)
    torch.testing.assert_close(unpack_awq_gemm(qzeros), qzero_values, rtol=0, atol=0)

    actual = dequantize_awq_gemm(
        qweight, qzeros, scales, group_size=group_size,
    )
    expected = (
        qweight_values.float()
        - qzero_values.float().repeat_interleave(group_size, dim=0)
    ) * scales.repeat_interleave(group_size, dim=0)
    torch.testing.assert_close(actual, expected.T.contiguous(), rtol=0, atol=0)


def test_official_awq_names_map_to_transformers_qwen25_vl_layout():
    mappings = {
        "model.layers.0.mlp.down_proj": "model.language_model.layers.0.mlp.down_proj",
        "model.embed_tokens.weight": "model.language_model.embed_tokens.weight",
        "model.norm.weight": "model.language_model.norm.weight",
        "visual.blocks.0.norm1.weight": "model.visual.blocks.0.norm1.weight",
        "lm_head.weight": "lm_head.weight",
    }
    assert {name: reference_model_name(name) for name in mappings} == mappings


class CorpusToy(nn.Module):
    def forward(self, input_ids, attention_mask=None, **_kwargs):
        vocabulary = 19
        values = torch.arange(vocabulary).reshape(1, 1, -1).float()
        logits = -(values - input_ids[..., None].float()).abs()
        return type("Output", (), {"logits": logits})


def test_corpus_batch_1_2_4_matches_without_cross_document_transitions():
    sequences = [
        torch.tensor([1, 2, 3, 4, 5]),
        torch.tensor([6, 7, 8]),
        torch.tensor([3, 2, 1, 0]),
    ]
    results = [
        evaluate_token_sequences_ppl(
            CorpusToy(), sequences, sequence_length=4, batch_size=batch,
        )
        for batch in (1, 2, 4)
    ]
    assert results[0]["tokens"] == 8
    assert [row["document_index"] for row in results[0]["window_records"]] == [0, 1, 2]
    for result in results[1:]:
        assert result["nll"] == pytest.approx(results[0]["nll"], abs=1e-7)
        assert result["ppl"] == pytest.approx(results[0]["ppl"], abs=1e-7)
        assert result["batch_validation_max_abs_token_nll_diff"] <= 1e-4

    candidate = json.loads(json.dumps(results[0]))
    for row in candidate["window_records"]:
        row["nll"] -= 0.1
        row["total_nll"] -= 0.1 * row["tokens"]
    delta = paired_window_delta(candidate, results[0])
    assert delta["conclusion"] == "improved"
    assert delta["bootstrap_95_ci"][1] < 0


class CollectorToy(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3, bias=False)

    def forward(self, input_ids, **_kwargs):
        return type("Output", (), {"logits": self.linear(input_ids.float())})


def test_deterministic_reservoir_cache_and_key_mismatch(tmp_path):
    model = CollectorToy()
    selected = [SelectedLinear("linear", model.linear, "decoder_ffn", 12)]
    rows = torch.arange(80, dtype=torch.float32).reshape(20, 4)
    with ActivationCollector(selected, reservoir_rows=5, seed=42) as first:
        model(rows)
    with ActivationCollector(selected, reservoir_rows=5, seed=42) as second:
        model(rows)
    torch.testing.assert_close(
        first.stats["linear"].reservoir,
        second.stats["linear"].reservoir,
        rtol=0,
        atol=0,
    )
    assert not torch.equal(first.stats["linear"].reservoir, rows[:5])

    key = calibration_cache_key(
        model_revision="model-a",
        calibration_digest="manifest-a",
        selected_names=["linear"],
        min_pixels=1,
        max_pixels=2,
        reservoir_rows=5,
        seed=42,
    )
    stats, cache_hit = collect_calibration_cached(
        model,
        selected,
        [{"input_ids": rows}],
        cache_dir=tmp_path,
        key=key,
        reservoir_rows=5,
        seed=42,
    )
    assert not cache_hit
    cached, cache_hit = collect_calibration_cached(
        model,
        selected,
        [],
        cache_dir=tmp_path,
        key=key,
        reservoir_rows=5,
        seed=42,
    )
    assert cache_hit
    torch.testing.assert_close(cached["linear"].reservoir, stats["linear"].reservoir)
    assert load_calibration_cache(
        tmp_path / f"{key}.pt", key="wrong", selected=selected,
    ) is None


def test_atomic_complete_requires_matching_digests_and_artifacts(tmp_path):
    metrics = tmp_path / "corpus_metrics.json"
    metrics.write_text("{}")
    digests = {"model": "a", "data": "b"}
    mark_complete(
        tmp_path,
        config_digest="config",
        input_digests=digests,
        required_artifacts=("corpus_metrics.json",),
    )
    assert completed_run(
        tmp_path,
        config_digest="config",
        input_digests=digests,
        required_artifacts=("corpus_metrics.json",),
    )
    assert not completed_run(
        tmp_path,
        config_digest="different",
        input_digests=digests,
        required_artifacts=("corpus_metrics.json",),
    )
    assert not completed_run(
        tmp_path,
        config_digest="config",
        input_digests={"model": "changed", "data": "b"},
        required_artifacts=("corpus_metrics.json",),
    )
    metrics.unlink()
    assert not completed_run(
        tmp_path,
        config_digest="config",
        input_digests=digests,
        required_artifacts=("corpus_metrics.json",),
    )
    with pytest.raises(FileNotFoundError):
        mark_complete(
            tmp_path,
            config_digest="config",
            input_digests=digests,
            required_artifacts=("missing.json",),
        )


def _official_rows(scores):
    datasets = ("DocVQA_VAL", "MathVista_MINI", "MMMU_DEV_VAL", "MMBench_DEV_EN")
    return [
        {
            "id": f"{dataset}:{index}",
            "dataset": dataset,
            "official_score": score,
        }
        for dataset, score in zip(datasets, scores)
        for index in (0, 1)
    ]


def test_paired_vlm_ci_and_best_w4_tie_break():
    reference = _official_rows([1.0, 1.0, 1.0, 1.0])
    improved = _official_rows([1.1, 1.1, 1.1, 1.1])
    difference = paired_score_difference(improved, reference)
    assert difference["conclusion"] == "improved"
    assert difference["bootstrap_95_ci"][0] > 0

    candidates = [
        {
            "run_id": "higher-bpw-full",
            "method": "awq",
            "scheme_id": "awq_w4",
            "scope": "full_vlm_extended",
            "implementation_tier": "local_algorithm_simulation",
            "logical_bits_per_weight": 4.5,
            "samples": _official_rows([0.9, 0.9, 0.9, 0.9]),
        },
        {
            "run_id": "lower-bpw-decoder",
            "method": "gptq",
            "scheme_id": "gptq_w4",
            "scope": "decoder_canonical",
            "implementation_tier": "local_algorithm_simulation",
            "logical_bits_per_weight": 4.25,
            "samples": _official_rows([0.9, 0.9, 0.9, 0.9]),
        },
    ]
    selection = select_best_local_w4(candidates, reference)
    assert selection["best_run_id"] == "lower-bpw-decoder"


def test_config_legacy_alias_and_scheme_digest_separation():
    legacy = ExperimentConfig.from_dict({
        "method": "bf16_reference",
        "scope": "decoder_canonical",
        "weight_bits": None,
    })
    assert legacy.method == MethodName.BF16_VALUE_FP32_COMPUTE
    w4 = ExperimentConfig(MethodName.RTN, weight_bits=4, scheme_id="rtn_w4")
    w8 = ExperimentConfig(MethodName.RTN, weight_bits=8, scheme_id="rtn_w8")
    assert w4.digest != w8.digest


def test_official_prompt_uses_injected_dataset_image_dumper():
    class PromptModel:
        def __init__(self):
            self.dump_image_func = None

        def set_dump_image(self, function):
            self.dump_image_func = function

        def build_prompt(self, line, dataset):
            return [
                {"type": "image", "value": self.dump_image_func(line)},
                {"type": "text", "value": line["question"]},
            ]

    model = PromptModel()
    model.set_dump_image(lambda line: line["image"])
    messages = build_prompt(
        model, {"image": "sample.png", "question": "What?"}, "DocVQA_VAL",
    )
    assert messages[0]["value"] == "sample.png"


def test_official_score_extraction_uses_dataset_specific_columns(tmp_path):
    doc = pd.DataFrame({"index": [1, 2], "eval_score": [0.75, 1.0], "res": ["x", "y"]})
    scores = extract_scores(
        "DocVQA_VAL", doc, tmp_path, ["1", "2"], lambda _path: None,
    )
    assert scores == {"1": 0.75, "2": 1.0}

    mcq = pd.DataFrame({"index": [3, 4], "hit": [1, 0], "score": [0.2, 0.8]})
    scores = extract_scores(
        "MMMU_DEV_VAL", mcq, tmp_path, ["3", "4"], lambda _path: None,
    )
    assert scores == {"3": 1.0, "4": 0.0}


def test_fixed_mathvista_judge_retries_three_times_at_zero(monkeypatch):
    import vlmeval.dataset.utils.mathvista as mathvista

    monkeypatch.setattr(mathvista, "post_check", lambda line, prefetch=False: False)
    monkeypatch.setattr(mathvista, "build_mathvista_gpt4_prompt", lambda line: "prompt")
    monkeypatch.setattr(mathvista, "FAIL_MSG", "FAIL")

    class Judge:
        def __init__(self):
            self.temperatures = []

        def generate(self, prompt, temperature):
            self.temperatures.append(temperature)
            return "FAIL"

    judge = Judge()
    result = mathvista_auxeval_fixed(judge, {})
    assert judge.temperatures == [0, 0, 0]
    assert result["res"] == ""
