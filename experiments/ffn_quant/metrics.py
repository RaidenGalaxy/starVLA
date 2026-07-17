"""Streaming tensor, hidden-state and logits error metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class TensorErrorAccumulator:
    error_power: float = 0.0
    signal_power: float = 0.0
    candidate_power: float = 0.0
    dot: float = 0.0
    max_abs_error: float = 0.0
    count: int = 0

    @torch.no_grad()
    def update(self, reference: torch.Tensor, candidate: torch.Tensor) -> None:
        if reference.shape != candidate.shape:
            raise ValueError(f"metric shape mismatch: {reference.shape} != {candidate.shape}")
        ref = reference.detach().float().reshape(-1)
        cand = candidate.detach().float().reshape(-1)
        error = cand - ref
        self.error_power += float(torch.dot(error, error).item())
        self.signal_power += float(torch.dot(ref, ref).item())
        self.candidate_power += float(torch.dot(cand, cand).item())
        self.dot += float(torch.dot(ref, cand).item())
        self.max_abs_error = max(self.max_abs_error, float(error.abs().max().item()) if error.numel() else 0.0)
        self.count += error.numel()

    def compute(self) -> dict[str, float | int]:
        mse = self.error_power / max(self.count, 1)
        normalized_mse = self.error_power / max(self.signal_power, 1e-30)
        denominator = math.sqrt(self.signal_power * self.candidate_power)
        cosine = self.dot / denominator if denominator else 1.0
        sqnr = math.inf if self.error_power == 0 else 10.0 * math.log10(max(self.signal_power, 1e-30) / self.error_power)
        return {
            "mse": mse,
            "normalized_mse": normalized_mse,
            "cosine_similarity": cosine,
            "sqnr_db": sqnr,
            "max_abs_error": self.max_abs_error,
            "numel": self.count,
        }


@dataclass
class LogitsAccumulator:
    kl_sum: float = 0.0
    top1_matches: int = 0
    target_count: int = 0
    nll_sum: float = 0.0

    @torch.no_grad()
    def update(
        self,
        reference_logits: torch.Tensor,
        candidate_logits: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> None:
        reference_logits = reference_logits.float()
        candidate_logits = candidate_logits.float()
        target_ids = target_ids.long().reshape(-1)
        if reference_logits.shape != candidate_logits.shape:
            raise ValueError("reference and candidate logits must have the same shape")
        if reference_logits.shape[0] != target_ids.numel():
            raise ValueError("one target id is required per logits row")
        ref_log_probs = torch.log_softmax(reference_logits, dim=-1)
        candidate_log_probs = torch.log_softmax(candidate_logits, dim=-1)
        ref_probs = ref_log_probs.exp()
        self.kl_sum += float((ref_probs * (ref_log_probs - candidate_log_probs)).sum().item())
        self.top1_matches += int((reference_logits.argmax(-1) == candidate_logits.argmax(-1)).sum().item())
        self.nll_sum += float(-candidate_log_probs.gather(1, target_ids[:, None]).sum().item())
        self.target_count += target_ids.numel()

    def compute(self) -> dict[str, float | int]:
        count = max(self.target_count, 1)
        mean_nll = self.nll_sum / count
        return {
            "kl_divergence": self.kl_sum / count,
            "top1_agreement": self.top1_matches / count,
            "target_nll": mean_nll,
            "target_perplexity": math.exp(min(mean_nll, 80.0)),
            "target_tokens": self.target_count,
        }
