"""Single source of truth for the active Top4 VLM evaluation protocol."""
from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source_split: str
    metric: str
    task_family: str
    max_score: float = 100.0
    higher_is_better: bool = True
    requires_image: bool = True
    local_scorer: bool = True


ACTIVE_DATASETS = (
    DatasetSpec("DocVQA_VAL", "validation", "ANLS", "ocr"),
    DatasetSpec("MMMU_DEV_VAL", "dev+validation", "Accuracy", "reasoning"),
    DatasetSpec("ScienceQA_TEST", "ScienceQA-IMG/test", "Accuracy", "reasoning"),
    DatasetSpec("TextVQA_VAL", "validation", "VQA Accuracy", "ocr"),
)
ACTIVE_DATASET_NAMES = tuple(item.name for item in ACTIVE_DATASETS)
DATASET_SPECS = {item.name: item for item in ACTIVE_DATASETS}
LEGACY_DATASET_NAMES = ("MathVista_MINI", "MMBench_DEV_EN")
PROTOCOL_VERSION = 3


def canonical_answers(value: Any) -> list[str]:
    """Return one or more reference answers without executing arbitrary input."""
    if isinstance(value, (list, tuple)):
        answers = list(value)
    elif isinstance(value, str) and value.lstrip().startswith(("[", "(")):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = value
        answers = list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
    else:
        answers = [value]
    result = [str(item).strip() for item in answers if str(item).strip()]
    if not result:
        raise ValueError("reference answer list is empty")
    return result


def normalized_levenshtein_distance(reference: str, prediction: str) -> float:
    reference = " ".join(str(reference).strip().lower().split())
    prediction = " ".join(str(prediction).strip().lower().split())
    if len(reference) > len(prediction):
        reference, prediction = prediction, reference
    distances = list(range(len(reference) + 1))
    for row, right in enumerate(prediction, 1):
        updated = [row]
        for column, left in enumerate(reference, 1):
            updated.append(
                distances[column - 1]
                if left == right
                else 1 + min(distances[column - 1], distances[column], updated[-1])
            )
        distances = updated
    denominator = max(len(reference), len(prediction))
    return 0.0 if denominator == 0 else distances[-1] / denominator


def docvqa_anls_score(answers: Any, prediction: Any, threshold: float = 0.5) -> float:
    """Official DocVQA ANLS sample score on the [0, 1] scale."""
    similarity = 1.0 - min(
        normalized_levenshtein_distance(answer, str(prediction))
        for answer in canonical_answers(answers)
    )
    return 0.0 if similarity < threshold else similarity


_VQA_PUNCTUATION = (
    ";", "/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\", "_",
    "-", ">", "<", "@", "`", ",", "?", "!",
)
_VQA_NUMBER_WORDS = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
    "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
    "nine": "9", "ten": "10",
}


def normalize_vqa_answer(value: Any) -> str:
    """EvalAI-style answer normalization used by the local TextVQA scorer."""
    text = str(value).replace("\n", " ").replace("\t", " ").strip()
    comma_in_number = re.search(r"(\d),(\d)", text) is not None
    for punctuation in _VQA_PUNCTUATION:
        if punctuation + " " in text or " " + punctuation in text or comma_in_number:
            text = text.replace(punctuation, "")
        else:
            text = text.replace(punctuation, " ")
    text = re.sub(r"(?<!\d)\.(?!\d)", "", text)
    words = []
    for word in text.lower().split():
        word = _VQA_NUMBER_WORDS.get(word, word)
        if word not in {"a", "an", "the"}:
            words.append(word)
    return " ".join(words)


def textvqa_consensus_score(answers: Any, prediction: Any) -> float:
    """TextVQA ten-answer consensus score on [0, 1]."""
    normalized_prediction = normalize_vqa_answer(prediction)
    matches = sum(
        normalize_vqa_answer(answer) == normalized_prediction
        for answer in canonical_answers(answers)
    )
    return min(1.0, matches / 3.0)


def finite_unit_score(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or result < 0 or result > 1:
        return None
    return result
