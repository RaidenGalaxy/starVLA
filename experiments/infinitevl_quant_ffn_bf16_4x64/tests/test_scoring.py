from pathlib import Path

from experiments.infinitevl_quant_ffn_bf16_4x64.scoring import (
    extract_choice,
    prompt_choices,
    score_prediction,
)
from experiments.vlm_quant_benchmark.data import MultimodalSample


def _sample(target="A", prompt=None):
    return MultimodalSample(
        id="ScienceQA_TEST:1",
        dataset="ScienceQA_TEST",
        split="test",
        image=Path("unused.png"),
        prompt=prompt or "Question\nOptions:\nA. stable temperatures\nB. sunlight\nSelect one.",
        target=target,
    )


def test_choice_parser_preserves_answer_letter_with_explanation():
    sample = _sample("A")
    assert score_prediction(sample, "A").score == 1.0
    assert score_prediction(sample, "A. stable temperatures").score == 1.0
    assert score_prediction(sample, "(A) stable temperatures").score == 1.0
    assert score_prediction(sample, "Answer: A because it protects life").score == 1.0


def test_choice_parser_does_not_take_last_incidental_article():
    sample = _sample("B", "Question\nOptions:\nA. fins\nB. a tail\nSelect one.")
    scored = score_prediction(sample, "B. a tail")
    assert scored.score == 1.0
    assert scored.extracted == "B"


def test_choice_parser_handles_answer_prefix_and_unique_option_text():
    choices = {"A": "alpha", "B": "second image", "C": "gamma"}
    assert extract_choice("Answer:C", choices)[0] == "C"
    assert extract_choice("It is the second image", choices)[0] == "B"
    assert prompt_choices("Options:\nA. alpha\nB. beta") == {"A": "alpha", "B": "beta"}


def test_non_choice_numeric_stays_exact():
    sample = MultimodalSample(
        id="MMMU_DEV_VAL:1", dataset="MMMU_DEV_VAL", split="dev",
        image=Path("unused.png"), prompt="Determine the value", target="50",
    )
    assert score_prediction(sample, "50").score == 1.0
    assert score_prediction(sample, "50 V").score == 0.0

