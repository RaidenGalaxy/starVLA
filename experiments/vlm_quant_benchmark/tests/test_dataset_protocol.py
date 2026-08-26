import pytest

from experiments.vlm_quant_benchmark.run_top4_protocol import quant_configs
from experiments.vlm_quant_benchmark.dataset_protocol import (
    ACTIVE_DATASET_NAMES,
    docvqa_anls_score,
    normalize_vqa_answer,
    textvqa_consensus_score,
)


def test_active_protocol_has_only_the_new_four_datasets():
    assert ACTIVE_DATASET_NAMES == (
        "DocVQA_VAL", "MMMU_DEV_VAL", "ScienceQA_TEST", "TextVQA_VAL",
    )


def test_top4_matrix_is_exactly_bf16_plus_four_paired_w4_w3_runs():
    configs = quant_configs()
    assert len(configs) == 9
    assert sum(row["bits"] is None for row in configs) == 1
    assert sum(row["bits"] == 4 for row in configs) == 4
    assert sum(row["bits"] == 3 for row in configs) == 4
    assert {row["method"] for row in configs if row["bits"] == 3} == {
        "mbq", "rtn", "omniquant", "squeezellm",
    }
    assert all(
        row["scheme"] == f"{row['method']}_w{row['bits']}a16_g128_asym"
        for row in configs if row["bits"] is not None
    )


def test_docvqa_anls_exact_partial_threshold_and_multiple_references():
    assert docvqa_anls_score("Hello   World", " hello world ") == 1.0
    assert docvqa_anls_score("abcd", "abce") == pytest.approx(0.75)
    assert docvqa_anls_score("abcdefghij", "abcdezzzzz") == 0.5
    assert docvqa_anls_score("abcdefghij", "abcdzzzzzz") == 0.0
    assert docvqa_anls_score("['wrong', 'best answer']", "best answer") == 1.0


def test_textvqa_normalization_and_consensus_levels():
    assert normalize_vqa_answer(" The two, cats! ") == "2 cats"
    for matches, expected in ((0, 0.0), (1, 1 / 3), (2, 2 / 3), (3, 1.0), (10, 1.0)):
        answers = ["target"] * matches + [f"other-{index}" for index in range(10 - matches)]
        assert textvqa_consensus_score(answers, "TARGET") == pytest.approx(expected)
