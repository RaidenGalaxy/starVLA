from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from experiments.vlm_quant_benchmark.data import MultimodalSample
from experiments.vlm_quant_benchmark.dataset_protocol import (
    canonical_answers,
    docvqa_anls_score,
    normalize_vqa_answer,
    textvqa_consensus_score,
)


SCORER_ID = "local_deterministic_v2"
_LABEL = re.compile(r"^[A-I]$", flags=re.IGNORECASE)
_OPTION = re.compile(r"^\s*([A-I])[.)]\s+(.+?)\s*$", flags=re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class ScoreResult:
    score: float
    rule: str
    extracted: str


@lru_cache(maxsize=1)
def _textvqa_answers() -> dict[str, str]:
    path = (
        Path(__file__).resolve().parents[2]
        / "data/vlm_quant_benchmark/LMUData/TextVQA_VAL_local.tsv"
    )
    csv.field_size_limit(sys.maxsize)
    with path.open(newline="") as handle:
        return {
            str(row["index"]): row["answer"]
            for row in csv.DictReader(handle, delimiter="\t")
        }


def prompt_choices(prompt: str) -> dict[str, str]:
    return {label.upper(): text.strip() for label, text in _OPTION.findall(prompt)}


def extract_choice(prediction: str, choices: dict[str, str]) -> tuple[str | None, str]:
    labels = "".join(sorted(choices)) or "ABCDEFGHI"
    patterns = (
        (rf"^\s*[\[(]?([{labels}])[\])\s.:]*$", "exact_label"),
        (
            rf"(?:final\s+answer|answer|option|choice)\s*(?:is|:|=)?\s*"
            rf"[\[(]?([{labels}])[\])]?(?!\w)",
            "explicit_label",
        ),
        (rf"^\s*[\[(]?([{labels}])[\]).:](?:\s|$)", "leading_label"),
        (rf"[\[(]([{labels}])[\])]", "bracketed_label"),
    )
    for pattern, rule in patterns:
        matches = re.findall(pattern, prediction, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper(), rule
    normalized = normalize_vqa_answer(prediction)
    hits = [
        label for label, text in choices.items()
        if normalize_vqa_answer(text)
        and normalize_vqa_answer(text) in normalized
    ]
    return (hits[0], "unique_option_text") if len(hits) == 1 else (None, "unparsed")


def score_prediction(sample: MultimodalSample, prediction: str) -> ScoreResult:
    if sample.dataset == "DocVQA_VAL":
        return ScoreResult(
            docvqa_anls_score(sample.target, prediction), "docvqa_anls", prediction,
        )
    if sample.dataset == "TextVQA_VAL":
        answers = _textvqa_answers().get(sample.official_index, sample.target)
        return ScoreResult(
            textvqa_consensus_score(answers, prediction),
            "textvqa_consensus", prediction,
        )

    raw_answers = [answer.strip() for answer in canonical_answers(sample.target)]
    if raw_answers and all(_LABEL.fullmatch(answer) for answer in raw_answers):
        expected = {answer.upper() for answer in raw_answers}
        extracted, rule = extract_choice(prediction, prompt_choices(sample.prompt))
        return ScoreResult(float(extracted in expected), rule, extracted or "")

    normalized = normalize_vqa_answer(prediction)
    expected = {normalize_vqa_answer(answer) for answer in raw_answers}
    return ScoreResult(float(normalized in expected), "normalized_exact", normalized)


def legacy_score_prediction(sample: MultimodalSample, prediction: str) -> float:
    """Frozen copy of the v1 local scorer for audit-only comparisons."""
    if sample.dataset == "DocVQA_VAL":
        return docvqa_anls_score(sample.target, prediction)
    if sample.dataset == "TextVQA_VAL":
        answers = _textvqa_answers().get(sample.official_index, sample.target)
        return textvqa_consensus_score(answers, prediction)
    answers = {normalize_vqa_answer(item) for item in canonical_answers(sample.target)}
    normalized = normalize_vqa_answer(prediction)
    if normalized in answers:
        return 1.0
    expected = {answer.upper() for answer in answers}
    inferred = re.findall(r"\b([A-E])\b", prediction.upper())
    choice = inferred[-1] if inferred else normalized[:1].upper()
    return float(choice in expected)


def rescore_rows(
    rows: list[dict], samples: dict[str, MultimodalSample],
) -> list[dict]:
    result = []
    for row in rows:
        sample = samples[row["id"]]
        scored = score_prediction(sample, row["prediction"])
        copied = dict(row)
        copied.update({
            "score_legacy": float(row.get("score", legacy_score_prediction(sample, row["prediction"]))),
            "score_v2": scored.score,
            "score": scored.score,
            "scoring_v2": {
                "scorer": SCORER_ID,
                "rule": scored.rule,
                "extracted": scored.extracted,
            },
        })
        result.append(copied)
    return result

