"""Metrics for diagnosing token-conditioned layer sensitivity on CPU.

The functions in this module operate on NumPy arrays and deliberately avoid
model-specific assumptions.  A future model adapter only needs to export
reference/perturbed hidden states, token-group masks, actions, and phase labels.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import numpy as np


Array = np.ndarray


def _as_float_array(value: Any, *, name: str) -> Array:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def cvar(values: Any, quantile: float = 0.9) -> float:
    """Return upper-tail conditional value at risk (larger means worse)."""

    array = _as_float_array(values, name="values").reshape(-1)
    if array.size == 0:
        raise ValueError("values must be non-empty")
    if not 0.0 <= quantile < 1.0:
        raise ValueError("quantile must satisfy 0 <= quantile < 1")
    threshold = np.quantile(array, quantile)
    tail = array[array >= threshold]
    return float(tail.mean())


def linear_cka(reference: Any, perturbed: Any) -> float:
    """Compute linear centered-kernel alignment for two 2-D matrices."""

    x = _as_float_array(reference, name="reference")
    y = _as_float_array(perturbed, name="perturbed")
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("linear_cka expects [observations, hidden_dim] arrays")
    if x.shape[0] != y.shape[0]:
        raise ValueError("reference and perturbed must share observation count")
    if x.shape[0] < 2:
        raise ValueError("linear_cka needs at least two observations")

    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    # ``||X.T @ Y||_F^2 == <X @ X.T, Y @ Y.T>``.  VLM hidden
    # dimensions are usually larger than the number of selected tokens, so
    # the Gram form avoids materialising several hidden_dim x hidden_dim
    # matrices for every layer perturbation.
    if x.shape[0] <= min(x.shape[1], y.shape[1]):
        x_gram = x @ x.T
        y_gram = y @ y.T
        cross = float(np.sum(x_gram * y_gram))
        x_norm = float(np.linalg.norm(x_gram, ord="fro"))
        y_norm = float(np.linalg.norm(y_gram, ord="fro"))
    else:
        cross = float(np.linalg.norm(x.T @ y, ord="fro") ** 2)
        x_norm = float(np.linalg.norm(x.T @ x, ord="fro"))
        y_norm = float(np.linalg.norm(y.T @ y, ord="fro"))
    denominator = x_norm * y_norm
    if denominator == 0.0:
        raise ValueError("linear_cka is undefined for a constant representation")
    return float(np.clip(cross / denominator, 0.0, 1.0))


def _validate_hidden_inputs(
    reference: Any,
    perturbed: Any,
    group_ids: Any,
) -> tuple[Array, Array, Array]:
    x = _as_float_array(reference, name="reference")
    y = _as_float_array(perturbed, name="perturbed")
    groups = np.asarray(group_ids)
    if x.ndim != 3:
        raise ValueError("hidden states must have shape [samples, tokens, hidden_dim]")
    if x.shape != y.shape:
        raise ValueError("reference and perturbed hidden states must share shape")
    if groups.shape != x.shape[:2]:
        raise ValueError("group_ids must have shape [samples, tokens]")
    return x, y, groups


def _resolve_group_names(groups: Array, group_names: Sequence[Hashable] | None) -> list[Hashable]:
    if group_names is not None:
        names = list(group_names)
    else:
        names = list(np.unique(groups))
    if not names:
        raise ValueError("at least one token group is required")
    return names


def cosine_drift_by_group(
    reference: Any,
    perturbed: Any,
    group_ids: Any,
    *,
    group_names: Sequence[Hashable] | None = None,
    epsilon: float = 1e-12,
) -> dict[Hashable, float]:
    """Mean ``1 - cosine`` per token group without token-count reweighting."""

    x, y, groups = _validate_hidden_inputs(reference, perturbed, group_ids)
    dot = np.sum(x * y, axis=-1)
    norm = np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1)
    cosine = dot / np.maximum(norm, epsilon)
    drift = 1.0 - np.clip(cosine, -1.0, 1.0)

    result: dict[Hashable, float] = {}
    for group in _resolve_group_names(groups, group_names):
        selected = drift[groups == group]
        if selected.size == 0:
            raise ValueError(f"token group {group!r} has no observations")
        result[group] = float(selected.mean())
    return result


def linear_cka_by_group(
    reference: Any,
    perturbed: Any,
    group_ids: Any,
    *,
    group_names: Sequence[Hashable] | None = None,
) -> dict[Hashable, float]:
    """Compute CKA independently after flattening sample/token rows per group."""

    x, y, groups = _validate_hidden_inputs(reference, perturbed, group_ids)
    result: dict[Hashable, float] = {}
    for group in _resolve_group_names(groups, group_names):
        mask = groups == group
        selected_x = x[mask]
        selected_y = y[mask]
        if selected_x.shape[0] < 2:
            raise ValueError(f"token group {group!r} needs at least two observations for CKA")
        result[group] = linear_cka(selected_x, selected_y)
    return result


def _normalized_action_error(reference: Any, perturbed: Any, scales: Any | None) -> Array:
    x = _as_float_array(reference, name="reference_actions")
    y = _as_float_array(perturbed, name="perturbed_actions")
    if x.shape != y.shape or x.ndim < 2:
        raise ValueError("actions must share shape [samples, ..., action_dim]")
    difference = y - x
    if scales is not None:
        divisor = _as_float_array(scales, name="action_scales")
        if divisor.ndim != 1 or divisor.shape[0] != x.shape[-1]:
            raise ValueError("action_scales must have shape [action_dim]")
        if np.any(divisor <= 0):
            raise ValueError("action_scales must be strictly positive")
        difference = difference / divisor
    axes = tuple(range(1, difference.ndim))
    return np.mean(np.square(difference), axis=axes)


def action_error_by_phase(
    reference: Any,
    perturbed: Any,
    phases: Any,
    *,
    action_scales: Any | None = None,
    quantile: float = 0.9,
) -> dict[Hashable, dict[str, float | int]]:
    """Summarize normalized per-sample action MSE by manipulation phase."""

    errors = _normalized_action_error(reference, perturbed, action_scales)
    phase_ids = np.asarray(phases)
    if phase_ids.shape != (errors.shape[0],):
        raise ValueError("phases must have shape [samples]")

    result: dict[Hashable, dict[str, float | int]] = {}
    for phase in np.unique(phase_ids):
        selected = errors[phase_ids == phase]
        result[phase] = {
            "count": int(selected.size),
            "mean": float(selected.mean()),
            "cvar": cvar(selected, quantile),
            "max": float(selected.max()),
        }
    return result


def _average_ranks(values: Array) -> Array:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = 0.5 * (start + end - 1)
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def rank_stability(reference_scores: Any, perturbed_scores: Any) -> float:
    """Spearman rank correlation with average ranks for tied values."""

    x = _as_float_array(reference_scores, name="reference_scores").reshape(-1)
    y = _as_float_array(perturbed_scores, name="perturbed_scores").reshape(-1)
    if x.shape != y.shape or x.size < 2:
        raise ValueError("score vectors must share length >= 2")
    x_rank = _average_ranks(x)
    y_rank = _average_ranks(y)
    x_rank -= x_rank.mean()
    y_rank -= y_rank.mean()
    denominator = np.linalg.norm(x_rank) * np.linalg.norm(y_rank)
    if denominator == 0.0:
        raise ValueError("rank correlation is undefined for a constant ranking")
    return float(np.clip(np.dot(x_rank, y_rank) / denominator, -1.0, 1.0))


def topk_overlap(reference_scores: Any, perturbed_scores: Any, k: int) -> float:
    """Fractional overlap between the indices of the k largest scores."""

    x = _as_float_array(reference_scores, name="reference_scores").reshape(-1)
    y = _as_float_array(perturbed_scores, name="perturbed_scores").reshape(-1)
    if x.shape != y.shape:
        raise ValueError("score vectors must share shape")
    if not 1 <= k <= x.size:
        raise ValueError("k must satisfy 1 <= k <= number of scores")
    x_top = set(np.argpartition(x, -k)[-k:].tolist())
    y_top = set(np.argpartition(y, -k)[-k:].tolist())
    return len(x_top & y_top) / k


def non_additive_interaction(joint_error: Any, skip_error: Any, quant_error: Any) -> Array:
    """Return ``joint - skip - quant`` elementwise."""

    joint = _as_float_array(joint_error, name="joint_error")
    skip = _as_float_array(skip_error, name="skip_error")
    quant = _as_float_array(quant_error, name="quant_error")
    if joint.shape != skip.shape or joint.shape != quant.shape:
        raise ValueError("joint, skip, and quant errors must share shape")
    return joint - skip - quant


def balanced_group_risk(
    group_risks: Mapping[Hashable, float],
    *,
    weights: Mapping[Hashable, float] | None = None,
) -> float:
    """Combine already aggregated group risks without using token counts."""

    if not group_risks:
        raise ValueError("group_risks must be non-empty")
    values = {group: float(risk) for group, risk in group_risks.items()}
    if not all(np.isfinite(risk) and risk >= 0.0 for risk in values.values()):
        raise ValueError("group risks must be finite and non-negative")
    if weights is None:
        return float(np.mean(list(values.values())))
    missing = set(values) - set(weights)
    if missing:
        raise ValueError(f"missing weights for groups: {sorted(map(str, missing))}")
    total_weight = sum(float(weights[group]) for group in values)
    if total_weight <= 0.0:
        raise ValueError("sum of selected group weights must be positive")
    return sum(values[group] * float(weights[group]) for group in values) / total_weight
