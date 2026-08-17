from __future__ import annotations

import math
import random
import re
import string
from collections.abc import Callable, Sequence

import torch


def normalize_answer(value: str) -> str:
    value = value.lower().strip().translate(str.maketrans("", "", string.punctuation))
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def exact_match(prediction: str, target: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(target))


def normalized_edit_similarity(prediction: str, target: str) -> float:
    left, right = normalize_answer(prediction), normalize_answer(target)
    if not left and not right:
        return 1.0
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return 1.0 - previous[-1] / max(len(left), len(right), 1)


def anls(prediction: str, target: str, threshold: float = 0.5) -> float:
    similarity = normalized_edit_similarity(prediction, target)
    return similarity if similarity >= threshold else 0.0


def tensor_error(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref, cand = reference.detach().float().reshape(-1), candidate.detach().float().reshape(-1)
    if ref.shape != cand.shape:
        raise ValueError(f"shape mismatch: {tuple(ref.shape)} != {tuple(cand.shape)}")
    if not torch.isfinite(ref).all() or not torch.isfinite(cand).all():
        raise ValueError("metric input contains NaN or Inf")
    error = cand - ref
    mse = error.square().mean().item() if error.numel() else 0.0
    power = ref.square().mean().item() if ref.numel() else 0.0
    return {
        "mse": mse,
        "nmse": mse / max(power, 1e-30),
        "cosine": torch.nn.functional.cosine_similarity(ref, cand, dim=0).item() if ref.numel() else 1.0,
        "sqnr_db": math.inf if mse == 0 else 10 * math.log10(max(power, 1e-30) / mse),
        "max_abs_error": error.abs().max().item() if error.numel() else 0.0,
    }


def bootstrap_ci(values: Sequence[float], *, statistic: Callable[[Sequence[float]], float] | None = None,
                 samples: int = 10_000, seed: int = 42) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    statistic = statistic or (lambda items: sum(items) / len(items))
    rng, estimates = random.Random(seed), []
    for _ in range(samples):
        estimates.append(statistic([values[rng.randrange(len(values))] for _ in values]))
    estimates.sort()
    return estimates[int(0.025 * samples)], estimates[min(int(0.975 * samples), samples - 1)]


class TokenNLL:
    """Token-weighted causal NLL; padding/prompt tokens use label -100."""

    def __init__(self) -> None:
        self.total_nll, self.tokens = 0.0, 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        shifted_logits = logits[..., :-1, :].float().reshape(-1, logits.shape[-1])
        shifted_labels = labels[..., 1:].reshape(-1)
        valid = shifted_labels != -100
        if not valid.any():
            return
        loss = torch.nn.functional.cross_entropy(shifted_logits[valid], shifted_labels[valid], reduction="sum")
        if not torch.isfinite(loss):
            raise ValueError("token NLL is non-finite")
        self.total_nll += float(loss.item())
        self.tokens += int(valid.sum().item())

    @property
    def mean_nll(self) -> float:
        return self.total_nll / self.tokens if self.tokens else math.nan

    @property
    def perplexity(self) -> float:
        return math.exp(self.mean_nll) if self.tokens else math.nan

    def to_dict(self) -> dict[str, float | int]:
        return {"total_nll": self.total_nll, "tokens": self.tokens,
                "nll": self.mean_nll, "ppl": self.perplexity}
