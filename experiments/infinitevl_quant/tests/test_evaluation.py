from __future__ import annotations

from experiments.infinitevl_quant.evaluation import _prediction_score
from experiments.vlm_quant_benchmark.data import MultimodalSample


def _sample(dataset: str, target: str, official_index: str = "") -> MultimodalSample:
    return MultimodalSample(
        id=f"{dataset}:x", dataset=dataset, split="test",
        image=__file__, prompt="question", target=target,
        official_index=official_index,
    )


def test_multiple_choice_answer_extraction():
    sample = _sample("MMMU_DEV_VAL", "B")
    assert _prediction_score(sample, "The correct answer is B.") == 1.0
    assert _prediction_score(sample, "I select option C.") == 0.0


def test_textvqa_uses_ten_answer_consensus():
    sample = _sample("TextVQA_VAL", "dakota", "34602")
    assert _prediction_score(sample, "Dakota") == 1.0
    assert _prediction_score(sample, "nous les gosses") == 1 / 3
