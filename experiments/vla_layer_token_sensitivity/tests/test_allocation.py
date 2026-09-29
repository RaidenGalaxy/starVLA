import pytest

from experiments.vla_layer_token_sensitivity.allocation import RiskReport, sequential_allocate


def test_allocator_recomputes_interaction_risk_and_respects_group_limit() -> None:
    ladders = {0: ("FP16", "W4", "SKIP"), 1: ("FP16", "W4", "SKIP")}
    costs = {"FP16": 1.0, "W4": 0.4, "SKIP": 0.0}

    def evaluator(plan):
        base = {"FP16": 0.0, "W4": 0.05, "SKIP": 0.18}
        total = base[plan[0]] + base[plan[1]]
        # Compressing both layers together creates a non-additive action-token
        # failure. The allocator must see it because it re-runs this evaluator.
        if plan[0] != "FP16" and plan[1] != "FP16":
            total += 0.5
        return RiskReport(total=total, groups={"action": total, "vision": total / 2})

    result = sequential_allocate(
        ladders,
        costs,
        target_saved_cost=1.0,
        evaluator=evaluator,
        group_limits={"action": 0.1},
    )

    assert not result.reached_target
    assert result.saved_cost == pytest.approx(0.6)
    assert sorted(result.plan.values()) == ["FP16", "W4"]
    assert result.risk.groups["action"] == 0.05


def test_allocator_reaches_budget_when_candidates_are_safe() -> None:
    ladders = {0: ("FP16", "W8", "W4"), 1: ("FP16", "W8", "W4")}
    costs = {"FP16": 1.0, "W8": 0.7, "W4": 0.4}

    def evaluator(plan):
        state_risk = {"FP16": 0.0, "W8": 0.02, "W4": 0.08}
        total = sum(state_risk[state] for state in plan.values())
        return RiskReport(total=total, groups={"action": total})

    result = sequential_allocate(
        ladders,
        costs,
        target_saved_cost=0.6,
        evaluator=evaluator,
        group_limits={"action": 0.2},
    )

    assert result.reached_target
    assert result.saved_cost >= 0.6
    assert len(result.steps) == 2
