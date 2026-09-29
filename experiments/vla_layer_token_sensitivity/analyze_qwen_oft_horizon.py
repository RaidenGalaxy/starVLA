"""Analyze schema-v2 Qwen-OFT horizon-conditioned scans and apply staged gates."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.vla_layer_token_sensitivity.metrics import rank_stability, topk_overlap
from experiments.vla_layer_token_sensitivity.qwen_oft_horizon_probe import STEP_GROUPS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--primary-layer", type=int, default=34)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260917)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("scan JSONL is empty")
    if any(int(row.get("schema_version", -1)) != 2 for row in rows):
        raise ValueError("horizon analyzer requires schema-version 2 rows")
    return rows


def _key(row: dict) -> tuple[str, str, str, int, str]:
    return (
        str(row["sample_id"]),
        str(row["input_condition"]),
        str(row["base_state"]),
        int(row["layer"]),
        str(row["operation"]),
    )


def _bootstrap_mean(
    values: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("bootstrap input must be a finite 1-D vector with at least two entries")
    estimates = np.asarray(
        [values[rng.integers(0, values.size, size=values.size)].mean() for _ in range(samples)]
    )
    return {
        "mean": float(values.mean()),
        "median": float(np.median(estimates)),
        "ci95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))],
        "positive_fraction": float(np.mean(values > 0.0)),
    }


def _safe_rank(left: np.ndarray, right: np.ndarray) -> float:
    try:
        return rank_stability(left, right)
    except ValueError:
        return float("nan")


def _rank_bootstrap(
    fp: np.ndarray,
    w4: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> dict:
    if fp.shape != w4.shape or fp.ndim != 2:
        raise ValueError("rank matrices must share [episodes,layers] shape")
    k = min(2, fp.shape[1])
    point_fp = fp.mean(axis=0)
    point_w4 = w4.mean(axis=0)
    correlations: list[float] = []
    overlaps: list[float] = []
    for _ in range(samples):
        indices = rng.integers(0, fp.shape[0], size=fp.shape[0])
        left = fp[indices].mean(axis=0)
        right = w4[indices].mean(axis=0)
        correlations.append(_safe_rank(left, right))
        overlaps.append(topk_overlap(left, right, k))
    finite = np.asarray([value for value in correlations if math.isfinite(value)])
    return {
        "point_spearman": _safe_rank(point_fp, point_w4),
        "bootstrap_spearman_median": None if finite.size == 0 else float(np.median(finite)),
        "bootstrap_spearman_ci95": (
            [None, None]
            if finite.size == 0
            else [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))]
        ),
        "point_top2_overlap": float(topk_overlap(point_fp, point_w4, k)),
        "bootstrap_top2_overlap_median": float(np.median(overlaps)),
    }


def _bh_adjust(p_values: dict[int, float]) -> dict[int, float]:
    if not p_values:
        return {}
    ordered = sorted(p_values, key=p_values.get)
    count = len(ordered)
    adjusted: dict[int, float] = {}
    running = 1.0
    for reverse_position in range(count - 1, -1, -1):
        layer = ordered[reverse_position]
        rank = reverse_position + 1
        running = min(running, p_values[layer] * count / rank)
        adjusted[layer] = float(min(1.0, running))
    return adjusted


def _sign_flip_pvalue(values: np.ndarray, samples: int, rng: np.random.Generator) -> float:
    observed = float(np.mean(values))
    signs = rng.choice((-1.0, 1.0), size=(samples, values.size))
    null = np.mean(signs * values[None, :], axis=1)
    return float((1 + np.count_nonzero(null >= observed)) / (samples + 1))


def _step_values(row: dict, comparison: str) -> np.ndarray:
    values = np.asarray(row["metrics"][comparison]["action_step_mse"], dtype=np.float64)
    if values.shape != (16,) or not np.all(np.isfinite(values)):
        raise ValueError("action_step_mse must contain 16 finite values")
    return values


def _interaction_by_episode(
    indexed: dict,
    sample_rows: list[dict],
    *,
    condition: str,
    layer: int,
) -> tuple[list[int], np.ndarray]:
    episodes: list[int] = []
    values: list[np.ndarray] = []
    for reference in sample_rows:
        sample = str(reference["sample_id"])
        quant = indexed[(sample, condition, "w4", -1, "reference")]
        skip = indexed[(sample, condition, "fp", layer, "skip")]
        joint = indexed[(sample, condition, "w4", layer, "skip")]
        interaction = (
            _step_values(joint, "vs_fp")
            - _step_values(quant, "vs_fp")
            - _step_values(skip, "vs_fp")
        )
        episodes.append(int(reference["episode_index"]))
        values.append(interaction)
    return episodes, np.stack(values)


def _condition_layer_complete(indexed: dict, samples: list[str], condition: str, layer: int) -> bool:
    expected = {
        (sample, condition, base, candidate_layer, operation)
        for sample in samples
        for base, candidate_layer, operation in (
            ("fp", -1, "reference"),
            ("w4", -1, "reference"),
            ("fp", layer, "skip"),
            ("w4", layer, "skip"),
        )
    }
    return expected <= set(indexed)


def analyze_horizon(
    rows: list[dict],
    *,
    primary_layer: int,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    indexed = {_key(row): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("scan contains duplicate schema-v2 keys")
    state_references = sorted(
        [
            row
            for row in rows
            if row["input_condition"] == "state"
            and row["base_state"] == "fp"
            and int(row["layer"]) == -1
            and row["operation"] == "reference"
        ],
        key=lambda row: int(row["episode_index"]),
    )
    if len(state_references) < 2:
        raise ValueError("at least two state-conditioned episode references are required")
    if len({int(row["episode_index"]) for row in state_references}) != len(state_references):
        raise ValueError("confirmation analysis requires one state sample per episode")
    samples = [str(row["sample_id"]) for row in state_references]
    if not _condition_layer_complete(indexed, samples, "state", primary_layer):
        raise ValueError(f"state-conditioned primary layer {primary_layer} scan is incomplete")

    rng = np.random.default_rng(seed)
    _, primary_steps = _interaction_by_episode(
        indexed, state_references, condition="state", layer=primary_layer
    )
    early = primary_steps[:, list(STEP_GROUPS["early"])].mean(axis=1)
    late = primary_steps[:, list(STEP_GROUPS["late"])].mean(axis=1)
    late_summary = _bootstrap_mean(late, samples=bootstrap_samples, rng=rng)
    difference_summary = _bootstrap_mean(late - early, samples=bootstrap_samples, rng=rng)

    additive: list[float] = []
    fp_gt: list[float] = []
    baseline_gt: list[float] = []
    for reference in state_references:
        sample = str(reference["sample_id"])
        quant = indexed[(sample, "state", "w4", -1, "reference")]
        skip = indexed[(sample, "state", "fp", primary_layer, "skip")]
        quant_late = _step_values(quant, "vs_fp")[list(STEP_GROUPS["late"])].mean()
        skip_late = _step_values(skip, "vs_fp")[list(STEP_GROUPS["late"])].mean()
        additive.append(float(quant_late + skip_late))
        fp_gt.append(float(reference["metrics"]["vs_ground_truth"]["action_mse"]))
        baseline_gt.append(float(reference["baseline_vs_ground_truth"]["action_mse"]))
    additive_mean = float(np.mean(additive))
    relative = float(late.mean() / additive_mean) if additive_mean > 0.0 else float("inf")
    fp_gt_mean = float(np.mean(fp_gt))
    baseline_gt_mean = float(np.mean(baseline_gt))
    fidelity_improvement = 1.0 - fp_gt_mean / baseline_gt_mean if baseline_gt_mean > 0 else -math.inf

    primary_checks = {
        "late_ci_positive": late_summary["ci95"][0] > 0.0,
        "relative_superadditivity_gte_0_25": relative >= 0.25,
        "positive_episode_fraction_gte_0_70": late_summary["positive_fraction"] >= 0.70,
        "late_minus_early_ci_positive": difference_summary["ci95"][0] > 0.0,
        "fp_gt_improvement_gte_0_10": fidelity_improvement >= 0.10,
    }
    confirmation_go = all(primary_checks.values())

    no_state = None
    per_task_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"state": [], "no_state": []}
    )
    for reference, value in zip(state_references, late):
        task = str(reference.get("instruction", f"task-{reference.get('task_index', 'unknown')}"))
        per_task_values[task]["state"].append(float(value))
    if _condition_layer_complete(indexed, samples, "no_state", primary_layer):
        _, no_state_steps = _interaction_by_episode(
            indexed, state_references, condition="no_state", layer=primary_layer
        )
        no_state_late = no_state_steps[:, list(STEP_GROUPS["late"])].mean(axis=1)
        no_state = _bootstrap_mean(no_state_late, samples=bootstrap_samples, rng=rng)
        no_state_gt = [
            float(indexed[(sample, "no_state", "fp", -1, "reference")]["metrics"]["vs_ground_truth"]["action_mse"])
            for sample in samples
        ]
        no_state["fp_ground_truth_mse"] = float(np.mean(no_state_gt))
        for reference, value in zip(state_references, no_state_late):
            task = str(reference.get("instruction", f"task-{reference.get('task_index', 'unknown')}"))
            per_task_values[task]["no_state"].append(float(value))

    per_task = {
        task: {
            condition: {
                "count": len(values),
                "mean_late_interaction": None if not values else float(np.mean(values)),
                "positive_fraction": None
                if not values
                else float(np.mean(np.asarray(values) > 0.0)),
            }
            for condition, values in conditions.items()
        }
        for task, conditions in sorted(per_task_values.items())
    }

    complete_layers = sorted(
        layer
        for layer in {
            int(row["layer"])
            for row in rows
            if row["input_condition"] == "state" and int(row["layer"]) >= 0
        }
        if _condition_layer_complete(indexed, samples, "state", layer)
    )
    rank_results: dict[str, dict] = {}
    rank_gate = False
    if len(complete_layers) >= 2:
        for step in (11, 14):
            fp = np.empty((len(samples), len(complete_layers)), dtype=np.float64)
            w4 = np.empty_like(fp)
            for sample_position, sample in enumerate(samples):
                for layer_position, layer in enumerate(complete_layers):
                    fp[sample_position, layer_position] = _step_values(
                        indexed[(sample, "state", "fp", layer, "skip")], "vs_fp"
                    )[step]
                    w4[sample_position, layer_position] = _step_values(
                        indexed[(sample, "state", "w4", layer, "skip")], "vs_base"
                    )[step]
            section = _rank_bootstrap(fp, w4, samples=bootstrap_samples, rng=rng)
            section["gate"] = bool(
                section["point_spearman"] < 0.7
                and section["bootstrap_spearman_median"] is not None
                and section["bootstrap_spearman_median"] < 0.7
                and section["point_top2_overlap"] <= 0.5
            )
            rank_results[str(step)] = section
            rank_gate = rank_gate or section["gate"]

    late_block_layers = [layer for layer in range(28, 36) if layer in complete_layers]
    late_block: dict[str, dict] = {}
    cluster_gate: bool | None = None
    if len(late_block_layers) == 8:
        p_values: dict[int, float] = {}
        layer_values: dict[int, np.ndarray] = {}
        for layer in late_block_layers:
            _, steps = _interaction_by_episode(indexed, state_references, condition="state", layer=layer)
            values = steps[:, list(STEP_GROUPS["late"])].mean(axis=1)
            layer_values[layer] = values
            p_values[layer] = _sign_flip_pvalue(values, bootstrap_samples, rng)
        q_values = _bh_adjust(p_values)
        positive_layers: list[int] = []
        for layer in late_block_layers:
            summary = _bootstrap_mean(layer_values[layer], samples=bootstrap_samples, rng=rng)
            qualifies = bool(
                summary["mean"] > 0.0
                and q_values[layer] < 0.05
                and summary["positive_fraction"] >= 0.65
            )
            if qualifies:
                positive_layers.append(layer)
            late_block[str(layer)] = {
                **summary,
                "one_sided_sign_flip_p": p_values[layer],
                "bh_q": q_values[layer],
                "qualifies": qualifies,
            }
        cluster_gate = any(right == left + 1 for left, right in zip(positive_layers, positive_layers[1:]))

    if not confirmation_go:
        full_scan_go = False
        full_scan_reason = "primary confirmation failed"
    elif cluster_gate is None:
        full_scan_go = None
        full_scan_reason = "primary passed; anchor/late-block evidence is incomplete"
    else:
        full_scan_go = bool(rank_gate or cluster_gate)
        full_scan_reason = "anchor/late-block gate evaluated"
    return {
        "status": "complete",
        "schema_version": 2,
        "sample_count": len(samples),
        "episode_count": len(state_references),
        "episode_indices": [int(row["episode_index"]) for row in state_references],
        "layers_complete": complete_layers,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "primary": {
            "layer": primary_layer,
            "late_interaction": late_summary,
            "early_interaction": _bootstrap_mean(early, samples=bootstrap_samples, rng=rng),
            "late_minus_early": difference_summary,
            "additive_late_mean": additive_mean,
            "relative_superadditivity": relative,
            "fp_ground_truth_mse": fp_gt_mean,
            "persistence_baseline_ground_truth_mse": baseline_gt_mean,
            "ground_truth_improvement": fidelity_improvement,
            "checks": primary_checks,
            "decision": "CONFIRM_GO" if confirmation_go else "CONFIRM_NO_GO",
        },
        "no_state_replication": no_state,
        "per_task_late_interaction": per_task,
        "anchor_rank": {
            "per_step": rank_results,
            "gate": rank_gate if rank_results else None,
        },
        "late_block": {
            "per_layer": late_block,
            "adjacent_positive_cluster_gate": cluster_gate,
        },
        "full_36_layer_gate": {
            "decision": (
                "PENDING"
                if full_scan_go is None
                else ("GO" if full_scan_go else "NO_GO")
            ),
            "rule": "primary confirmation and (step-11/14 rank gate or adjacent late-layer cluster)",
            "reason": full_scan_reason,
        },
        "limitations": [
            "open-loop normalized-action analysis, not robot success",
            "W4 uses dequantized float CPU weights, not packed INT4",
            "single early 10K-step Qwen-OFT checkpoint",
        ],
    }


def _render_report(result: dict) -> str:
    primary = result["primary"]
    lines = [
        "# Qwen-OFT Horizon-Conditioned 确认实验报告",
        "",
        f"主要确认判定：**{primary['decision']}**。全 36 层门控："
        f"**{result['full_36_layer_gate']['decision']}**。",
        "",
        f"样本：{result['sample_count']} 个未见 episode；主检验层：{primary['layer']}；"
        f"bootstrap：{result['bootstrap_samples']} 次。",
        "",
        "## 主检验",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| late interaction mean | {primary['late_interaction']['mean']:.8f} |",
        f"| late interaction 95% CI | [{primary['late_interaction']['ci95'][0]:.8f}, {primary['late_interaction']['ci95'][1]:.8f}] |",
        f"| positive episode fraction | {primary['late_interaction']['positive_fraction']:.3f} |",
        f"| relative superadditivity | {primary['relative_superadditivity']:.3f} |",
        f"| late - early 95% CI | [{primary['late_minus_early']['ci95'][0]:.8f}, {primary['late_minus_early']['ci95'][1]:.8f}] |",
        f"| FP GT MSE | {primary['fp_ground_truth_mse']:.6f} |",
        f"| persistence baseline GT MSE | {primary['persistence_baseline_ground_truth_mse']:.6f} |",
        f"| FP improvement | {primary['ground_truth_improvement']:.3%} |",
        "",
        "## 预注册条件",
        "",
    ]
    for name, passed in primary["checks"].items():
        lines.append(f"- `{name}`: **{passed}**")
    no_state = result["no_state_replication"]
    if no_state is not None:
        lines.extend(
            [
                "",
                "## 无 state 对照",
                "",
                f"无 state 的 late interaction mean 为 `{no_state['mean']:.8f}`，95% CI 为 "
                f"`[{no_state['ci95'][0]:.8f}, {no_state['ci95'][1]:.8f}]`，"
                f"正值 episode 比例为 `{no_state['positive_fraction']:.3f}`。",
                "这说明 discovery 信号在无 state 条件下复现，但不能推广到训练一致的 state 条件。",
                "",
                "| 任务 | state mean | no-state mean |",
                "|---|---:|---:|",
            ]
        )
        for task, values in result["per_task_late_interaction"].items():
            state_mean = values["state"]["mean_late_interaction"]
            no_state_mean = values["no_state"]["mean_late_interaction"]
            lines.append(f"| {task} | {state_mean:.8f} | {no_state_mean:.8f} |")
    lines.extend(
        [
            "",
            "## 后续门控",
            "",
            f"- 完整 state 层：{result['layers_complete']}",
            f"- Step 11/14 排名门控：{result['anchor_rank']['gate']}",
            f"- 28–35 相邻正交互簇：{result['late_block']['adjacent_positive_cluster_gate']}",
            "",
            "## 边界",
            "",
            "该结果是 CPU 开环、fake-W4、单个早期 checkpoint 的机制确认，不能解释为闭环成功率或真实 INT4 性能。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    result = analyze_horizon(
        _read_jsonl(args.input),
        primary_layer=args.primary_layer,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = args.output_json.with_suffix(args.output_json.suffix + ".tmp")
    temporary_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_json.replace(args.output_json)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = args.output_report.with_suffix(args.output_report.suffix + ".tmp")
    temporary_report.write_text(_render_report(result), encoding="utf-8")
    temporary_report.replace(args.output_report)
    print(json.dumps({"status": "complete", "primary": result["primary"]["decision"], "full36": result["full_36_layer_gate"]["decision"]}))


if __name__ == "__main__":
    main()
