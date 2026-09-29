"""Sequential depth/precision allocation with recomputed marginal risk."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import TypeAlias


Plan: TypeAlias = dict[int, str]


@dataclass(frozen=True)
class RiskReport:
    total: float
    groups: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isfinite(self.total) or self.total < 0.0:
            raise ValueError("total risk must be finite and non-negative")
        if any(not isfinite(value) or value < 0.0 for value in self.groups.values()):
            raise ValueError("group risks must be finite and non-negative")


@dataclass(frozen=True)
class AllocationStep:
    layer: int
    previous: str
    selected: str
    saved_cost: float
    marginal_risk: float
    total_risk: float


@dataclass(frozen=True)
class AllocationResult:
    plan: Plan
    saved_cost: float
    risk: RiskReport
    steps: tuple[AllocationStep, ...]
    reached_target: bool


Evaluator: TypeAlias = Callable[[Mapping[int, str]], RiskReport]


def _validate_ladders(
    ladders: Mapping[int, Sequence[str]],
    costs: Mapping[str, float],
) -> None:
    if not ladders:
        raise ValueError("at least one layer ladder is required")
    for layer, ladder in ladders.items():
        if not ladder:
            raise ValueError(f"layer {layer} has an empty precision ladder")
        missing = [state for state in ladder if state not in costs]
        if missing:
            raise ValueError(f"missing costs for layer {layer} states: {missing}")
        state_costs = [float(costs[state]) for state in ladder]
        if any(not isfinite(cost) or cost < 0.0 for cost in state_costs):
            raise ValueError("state costs must be finite and non-negative")
        if any(later >= earlier for earlier, later in zip(state_costs, state_costs[1:])):
            raise ValueError("each ladder must move strictly from higher to lower cost")


def _is_feasible(report: RiskReport, limits: Mapping[str, float]) -> bool:
    return all(report.groups.get(group, float("inf")) <= limit for group, limit in limits.items())


def sequential_allocate(
    ladders: Mapping[int, Sequence[str]],
    costs: Mapping[str, float],
    *,
    target_saved_cost: float,
    evaluator: Evaluator,
    group_limits: Mapping[str, float] | None = None,
    max_total_risk: float | None = None,
    epsilon: float = 1e-12,
) -> AllocationResult:
    """Greedily lower precision/depth while re-evaluating every proposal.

    ``ladders[layer]`` is ordered from the dense state toward stronger
    compression, for example ``("FP16", "W8", "W4", "W3", "SKIP")``.
    The evaluator receives a complete plan.  Re-evaluating all one-step moves
    after every selection captures pairwise and higher-order interactions that
    a static additive score would miss.
    """

    _validate_ladders(ladders, costs)
    if not isfinite(target_saved_cost) or target_saved_cost < 0.0:
        raise ValueError("target_saved_cost must be finite and non-negative")
    if max_total_risk is not None and (not isfinite(max_total_risk) or max_total_risk < 0.0):
        raise ValueError("max_total_risk must be finite and non-negative")
    limits = dict(group_limits or {})
    if any(not isfinite(limit) or limit < 0.0 for limit in limits.values()):
        raise ValueError("group limits must be finite and non-negative")

    layers = sorted(ladders)
    positions = {layer: 0 for layer in layers}
    plan: Plan = {layer: ladders[layer][0] for layer in layers}
    initial_cost = sum(float(costs[plan[layer]]) for layer in layers)
    current_cost = initial_cost
    current_risk = evaluator(plan)
    if not _is_feasible(current_risk, limits):
        raise ValueError("initial plan violates a protected-group risk limit")
    if max_total_risk is not None and current_risk.total > max_total_risk:
        raise ValueError("initial plan violates max_total_risk")

    steps: list[AllocationStep] = []
    while initial_cost - current_cost + epsilon < target_saved_cost:
        proposals: list[tuple[float, float, int, str, float, RiskReport]] = []
        for layer in layers:
            next_position = positions[layer] + 1
            if next_position >= len(ladders[layer]):
                continue
            selected = ladders[layer][next_position]
            saved = float(costs[plan[layer]]) - float(costs[selected])
            candidate_plan = dict(plan)
            candidate_plan[layer] = selected
            report = evaluator(candidate_plan)
            if max_total_risk is not None and report.total > max_total_risk:
                continue
            if not _is_feasible(report, limits):
                continue
            marginal = report.total - current_risk.total
            # Negative marginal risk can happen when compression regularizes a
            # model.  Clamp only for ranking; retain the signed value in logs.
            score = saved / max(marginal, epsilon)
            proposals.append((score, -report.total, layer, selected, marginal, report))

        if not proposals:
            break
        _, _, layer, selected, marginal, report = max(proposals)
        previous = plan[layer]
        saved = float(costs[previous]) - float(costs[selected])
        plan[layer] = selected
        positions[layer] += 1
        current_cost -= saved
        current_risk = report
        steps.append(
            AllocationStep(
                layer=layer,
                previous=previous,
                selected=selected,
                saved_cost=saved,
                marginal_risk=marginal,
                total_risk=report.total,
            )
        )

    saved_cost = initial_cost - current_cost
    return AllocationResult(
        plan=plan,
        saved_cost=saved_cost,
        risk=current_risk,
        steps=tuple(steps),
        reached_target=saved_cost + epsilon >= target_saved_cost,
    )
