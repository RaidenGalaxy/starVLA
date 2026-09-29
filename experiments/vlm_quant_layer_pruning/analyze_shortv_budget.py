from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .core import atomic_json
from .shortv_probe import SCHEMA_NAME


QUANT_SCHEMES = ("fp", "mbq_w4", "mbq_w3")
POINT_NLL_LIMIT = math.log(1.05)
UPPER_NLL_LIMIT = math.log(1.10)
POINT_KL_LIMIT = 0.02
UPPER_KL_LIMIT = 0.05


def read_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open() as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        raise ValueError("Q-ShortV budget input is empty")
    if any(row.get("schema_name") != SCHEMA_NAME for row in rows):
        raise ValueError("budget inputs do not use the Q-ShortV schema")
    return rows


def _bootstrap_indices(datasets: list[str], rng: np.random.Generator) -> np.ndarray:
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, dataset in enumerate(datasets):
        buckets[dataset].append(index)
    drawn = []
    for dataset in sorted(buckets):
        values = np.asarray(buckets[dataset], dtype=np.int64)
        drawn.extend(rng.choice(values, size=len(values), replace=True).tolist())
    return np.asarray(drawn, dtype=np.int64)


def _interval(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def analyze(
    rows: list[dict], ranking: dict, *, bootstrap_samples: int = 2000, seed: int = 42,
) -> dict:
    fixed_ranking = [int(value) for value in ranking["ranking"]]
    if sorted(fixed_ranking) != list(range(36)):
        raise ValueError("budget analysis requires a complete 36-layer FP ranking")
    samples = sorted({str(row["sample_id"]) for row in rows})
    sample_dataset = {}
    indexed = {}
    baseline = {}
    for row in rows:
        sample = str(row["sample_id"])
        state = str(row["quant_scheme"])
        if state not in QUANT_SCHEMES:
            continue
        sample_dataset.setdefault(sample, str(row["dataset"]))
        if row["operation"] == "baseline":
            key = (sample, state)
            value = float(row["metrics"]["target_nll"])
            # Separate CPU runs can differ at the low 1e-6 level because BLAS
            # reductions are not bitwise deterministic.  Budget deltas are
            # already computed against the baseline from the same probe run;
            # this check only guards against combining different artifacts.
            if key in baseline and not math.isclose(
                baseline[key], value, rel_tol=1e-6, abs_tol=1e-5,
            ):
                raise ValueError(f"inconsistent duplicate baseline row: {key}")
            baseline[key] = value
            continue
        if row["operation"] != "freeze_visual":
            continue
        layers = tuple(int(value) for value in row["layer_indices"])
        budget = len(layers)
        expected_layers = tuple(sorted(fixed_ranking[:budget]))
        if tuple(sorted(layers)) != expected_layers:
            raise ValueError(
                f"layer set is not the FP-ranking prefix for budget {budget}: {layers}"
            )
        key = (sample, state, budget)
        if key in indexed:
            raise ValueError(f"duplicate cumulative budget row: {key}")
        indexed[key] = row
    if set(sample_dataset) != set(samples):
        raise ValueError("dataset labels are incomplete")
    budgets_by_state = {
        state: sorted({
            budget for _, row_state, budget in indexed if row_state == state
        })
        for state in QUANT_SCHEMES
    }
    empty_states = [state for state, budgets in budgets_by_state.items() if not budgets]
    if empty_states:
        raise ValueError(f"states have no cumulative budget rows: {empty_states}")
    expected = {
        (sample, state, budget)
        for sample in samples
        for state in QUANT_SCHEMES
        for budget in budgets_by_state[state]
    }
    missing = expected - set(indexed)
    if missing:
        raise ValueError(f"missing {len(missing)} cumulative budget rows")
    missing_baselines = {
        (sample, state) for sample in samples for state in QUANT_SCHEMES
    } - set(baseline)
    if missing_baselines:
        raise ValueError(f"missing {len(missing_baselines)} baseline rows")

    datasets = [sample_dataset[sample] for sample in samples]
    positions = {name: [i for i, value in enumerate(datasets) if value == name] for name in set(datasets)}
    rng = np.random.default_rng(seed)
    draws = [_bootstrap_indices(datasets, rng) for _ in range(bootstrap_samples)]
    curves = {}
    frontiers = {}
    for state in QUANT_SCHEMES:
        state_rows = [{
            "budget": 0,
            "layers": [],
            "delta_nll_median": 0.0,
            "delta_nll_ci95": [0.0, 0.0],
            "answer_kl_median": 0.0,
            "answer_kl_ci95": [0.0, 0.0],
            "max_dataset_delta_nll_median": 0.0,
            "raw_safe": True,
            "frontier_safe": True,
        }]
        all_previous_safe = True
        for budget in budgets_by_state[state]:
            selected = [indexed[(sample, state, budget)] for sample in samples]
            delta = np.asarray([row["metrics"]["delta_target_nll"] for row in selected], dtype=np.float64)
            kl = np.asarray([max(0.0, row["metrics"]["answer_logit_kl"]) for row in selected], dtype=np.float64)
            if not np.isfinite(delta).all() or not np.isfinite(kl).all():
                raise ValueError("cumulative budget metrics contain non-finite values")
            delta_boot = np.asarray([np.median(delta[draw]) for draw in draws])
            kl_boot = np.asarray([np.median(kl[draw]) for draw in draws])
            dataset_medians = {
                name: float(np.median(delta[index])) for name, index in positions.items()
            }
            delta_point = float(np.median(delta))
            kl_point = float(np.median(kl))
            delta_ci = _interval(delta_boot)
            kl_ci = _interval(kl_boot)
            raw_safe = bool(
                delta_point <= POINT_NLL_LIMIT
                and delta_ci[1] <= UPPER_NLL_LIMIT
                and kl_point <= POINT_KL_LIMIT
                and kl_ci[1] <= UPPER_KL_LIMIT
                and max(dataset_medians.values()) <= UPPER_NLL_LIMIT
            )
            all_previous_safe = all_previous_safe and raw_safe
            state_rows.append({
                "budget": budget,
                "layers": fixed_ranking[:budget],
                "delta_nll_median": delta_point,
                "delta_nll_ci95": delta_ci,
                "answer_kl_median": kl_point,
                "answer_kl_ci95": kl_ci,
                "dataset_delta_nll_median": dict(sorted(dataset_medians.items())),
                "max_dataset_delta_nll_median": max(dataset_medians.values()),
                "raw_safe": raw_safe,
                "frontier_safe": all_previous_safe,
            })
        curves[state] = state_rows
        frontiers[state] = max(row["budget"] for row in state_rows if row["frontier_safe"])

    refinement_by_state = {state: [] for state in QUANT_SCHEMES}
    for state in QUANT_SCHEMES:
        rows_by_budget = curves[state]
        first_unsafe = next((row["budget"] for row in rows_by_budget if not row["frontier_safe"]), None)
        if first_unsafe is None:
            continue
        previous = max(row["budget"] for row in rows_by_budget if row["budget"] < first_unsafe)
        tested = set(budgets_by_state[state])
        refinement_by_state[state] = [
            budget for budget in range(previous + 1, first_unsafe)
            if budget not in tested
        ]
    refinement = sorted({
        budget for budgets in refinement_by_state.values() for budget in budgets
    })
    spread = max(frontiers.values()) - min(frontiers.values())
    return {
        "schema_name": "qshortv_budget_gate_v1",
        "schema_version": 1,
        "samples": len(samples),
        "datasets": dict(sorted({name: datasets.count(name) for name in set(datasets)}.items())),
        "ranking": fixed_ranking,
        "budgets": [0, *sorted({
            budget for budgets in budgets_by_state.values() for budget in budgets
        })],
        "budgets_by_state": {
            state: [0, *budgets_by_state[state]] for state in QUANT_SCHEMES
        },
        "curves": curves,
        "proxy_safe_frontier": frontiers,
        "frontier_spread_layers": spread,
        "refinement_budgets": refinement,
        "refinement_budgets_by_state": refinement_by_state,
        "proceed_refinement": bool(refinement),
        "proceed_generation_after_refinement_rule": "max frontier - min frontier >= 4 layers",
        "thresholds": {
            "point_delta_nll": POINT_NLL_LIMIT,
            "upper95_delta_nll": UPPER_NLL_LIMIT,
            "point_answer_kl": POINT_KL_LIMIT,
            "upper95_answer_kl": UPPER_KL_LIMIT,
            "max_dataset_delta_nll": UPPER_NLL_LIMIT,
        },
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
    }


def render_report(result: dict) -> str:
    lines = [
        "# Q-ShortV 位宽条件视觉冻结预算报告",
        "",
        f"样本：{result['samples']}；预算并集：{result['budgets']}；bootstrap：{result['bootstrap_samples']}。",
        "",
        f"各状态预算：`{result['budgets_by_state']}`。",
        "",
        "三个位宽共用 FP 风险上界排序；`frontier-safe` 要求当前及所有更小预算均通过代理安全门限。",
        "",
        "| 状态 | proxy-safe frontier |",
        "|---|---:|",
    ]
    for state in QUANT_SCHEMES:
        lines.append(f"| {state} | {result['proxy_safe_frontier'][state]} |")
    lines.extend([
        "",
        f"待补预算：`{result['refinement_budgets']}`。",
        "",
        "| 状态 | k | median ΔNLL | 95% CI | median KL | KL 95% CI | raw safe | frontier safe |",
        "|---|---:|---:|---|---:|---|---|---|",
    ])
    for state in QUANT_SCHEMES:
        for row in result["curves"][state]:
            lines.append(
                f"| {state} | {row['budget']} | {row['delta_nll_median']:.6f} "
                f"| {row['delta_nll_ci95']} | {row['answer_kl_median']:.6f} "
                f"| {row['answer_kl_ci95']} | {row['raw_safe']} | {row['frontier_safe']} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Q-ShortV cumulative visual budgets")
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ranking = json.loads(args.ranking.read_text())
    result = analyze(
        read_rows(args.input), ranking,
        bootstrap_samples=args.bootstrap_samples, seed=args.seed,
    )
    atomic_json(args.output_json, result)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_report.with_suffix(args.output_report.suffix + ".tmp")
    temporary.write_text(render_report(result))
    temporary.replace(args.output_report)


if __name__ == "__main__":
    main()
