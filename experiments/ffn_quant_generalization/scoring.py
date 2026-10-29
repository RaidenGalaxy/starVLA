from __future__ import annotations

import math

SCORER_ID = "generalization_local_v1_chartqa_relaxed5pct_gqa_exact"


def normalize(text):
    return str(text).strip().lower()


def number(text):
    text = str(text).strip()
    try:
        value = float(text[:-1]) / 100 if text.endswith("%") else float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def score_prediction(sample, prediction):
    if sample.dataset == "ChartQA_TEST":
        target, guess = number(sample.target), number(prediction)
        if target is not None and target != 0 and guess is not None:
            return float(abs(guess - target) / abs(target) <= 0.05)
        return float(normalize(prediction) == normalize(sample.target))
    if sample.dataset == "GQA_TESTDEV_BALANCED":
        return float(normalize(prediction) == normalize(sample.target))
    raise ValueError(f"scorer received unregistered dataset {sample.dataset}")
