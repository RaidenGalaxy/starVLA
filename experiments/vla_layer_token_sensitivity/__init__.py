"""CPU-first diagnostics for token-conditioned VLA layer compression."""

from .allocation import AllocationResult, RiskReport, sequential_allocate
from .metrics import (
    action_error_by_phase,
    cosine_drift_by_group,
    cvar,
    linear_cka,
    linear_cka_by_group,
    non_additive_interaction,
    rank_stability,
    topk_overlap,
)

__all__ = [
    "AllocationResult",
    "RiskReport",
    "action_error_by_phase",
    "cosine_drift_by_group",
    "cvar",
    "linear_cka",
    "linear_cka_by_group",
    "non_additive_interaction",
    "rank_stability",
    "sequential_allocate",
    "topk_overlap",
]
