"""Aggregate CPU-pilot predictions with source-group paired bootstrap intervals."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .protocol import atomic_json, file_sha256


def paired_group_bootstrap(
    differences: dict[str, list[float]],
    *,
    samples: int = 10_000,
    seed: int = 42,
) -> dict[str, float | int | None]:
    group_values = np.asarray([np.mean(values) for _, values in sorted(differences.items())], dtype=float)
    if len(group_values) == 0:
        return {"groups": 0, "mean": None, "ci95_low": None, "ci95_high": None, "p_le_zero": None}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(group_values), size=(samples, len(group_values)))
    boot = group_values[indices].mean(axis=1)
    return {
        "groups": len(group_values),
        "mean": float(group_values.mean()),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "p_le_zero": float(np.mean(boot <= 0.0)),
    }


def _load(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status", "ok") != "ok":
                    continue
                identity = row.get("inference_cache_key", f"{path}:{len(rows)}")
                if identity not in seen:
                    rows.append(row)
                    seen.add(identity)
    return rows


def _condition(row: dict[str, Any]) -> str:
    condition = row.get("condition", {})
    return f"delay={condition.get('appended_delay_sec', condition.get('delay_sec'))}|profile={condition.get('distractor_profile')}"


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _auc(points: list[tuple[int, float]]) -> float | None:
    unique = sorted(dict(points).items())
    if len(unique) < 2 or unique[-1][0] == unique[0][0]:
        return None
    x = np.asarray([item[0] for item in unique], dtype=float)
    y = np.asarray([item[1] for item in unique], dtype=float)
    return float(np.trapz(y, x) / (x[-1] - x[0]))


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate", default="future_utility")
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["uniform_fifo", "recent_fifo", "reservoir", "motion_topk", "novelty_topk"],
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = _load(args.results)
    if not rows:
        raise ValueError("no successful prediction rows")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                row["method"],
                int(row["memory_frames"]),
                float(row["encode_fps"]),
                row.get("stage", "final_memory"),
                _condition(row),
            )
        ].append(row)
    configurations = []
    for key, items in sorted(groups.items()):
        scores = [float(item["prediction"]["score"]) for item in items]
        configurations.append(
            {
                "method": key[0],
                "memory_frames": key[1],
                "encode_fps": key[2],
                "stage": key[3],
                "condition": key[4],
                "samples": len(items),
                "source_groups": len({item["source_group"] for item in items}),
                "accuracy": _mean(scores),
                "capture_span_recall": _mean(
                    [float(item["metrics"]["capture_span_recall"]) for item in items if item["metrics"].get("capture_span_recall") is not None]
                ),
                "retention_span_recall": _mean(
                    [float(item["metrics"]["retention_span_recall"]) for item in items if item["metrics"].get("retention_span_recall") is not None]
                ),
                "mean_frames_queried": _mean([float(item.get("frames_queried", 0)) for item in items]),
                "mean_query_latency_sec": _mean([float(item["query_latency_sec"]) for item in items]),
                "mean_retained_rgb_bytes": _mean(
                    [float(item["metrics"].get("retained_rgb_bytes", 0)) for item in items]
                ),
            }
        )

    final = [row for row in rows if row.get("stage", "final_memory") == "final_memory"]
    score_map = {
        (
            row["sample_id"],
            _condition(row),
            row["method"],
            int(row["memory_frames"]),
            float(row["encode_fps"]),
        ): row
        for row in final
    }
    comparisons = []
    candidate_configs = sorted(
        {
            (int(row["memory_frames"]), float(row["encode_fps"]))
            for row in final
            if row["method"] == args.candidate
        }
    )
    for memory, fps in candidate_configs:
        for baseline in args.baselines:
            differences: dict[str, list[float]] = defaultdict(list)
            for row in final:
                if row["method"] != args.candidate or int(row["memory_frames"]) != memory or float(row["encode_fps"]) != fps:
                    continue
                paired = score_map.get((row["sample_id"], _condition(row), baseline, memory, fps))
                if paired is not None:
                    differences[row["source_group"]].append(
                        float(row["prediction"]["score"]) - float(paired["prediction"]["score"])
                    )
            comparison = paired_group_bootstrap(
                differences,
                samples=args.bootstrap_samples,
                seed=args.seed + memory + int(10 * fps),
            )
            comparisons.append(
                {"candidate": args.candidate, "baseline": baseline, "memory_frames": memory, "encode_fps": fps, **comparison}
            )

    interference = []
    by_config_sample: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in final:
        key = (row["method"], int(row["memory_frames"]), float(row["encode_fps"]), row["sample_id"])
        by_config_sample[key][_condition(row)] = row
    config_diffs: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for (method, memory, fps, _sample), conditions in by_config_sample.items():
        clean = next(
            (
                row
                for name, row in conditions.items()
                if "profile=clean" in name and float(row.get("condition", {}).get("appended_delay_sec", 0) or 0) == 0
            ),
            None,
        )
        if clean is None:
            continue
        for name, row in conditions.items():
            if row is clean:
                continue
            gap = float(clean["prediction"]["score"]) - float(row["prediction"]["score"])
            config_diffs[(method, memory, fps, name)].append(gap)
    for key, values in sorted(config_diffs.items()):
        interference.append(
            {
                "method": key[0],
                "memory_frames": key[1],
                "encode_fps": key[2],
                "condition": key[3],
                "interference_gap": float(np.mean(values)),
                "paired_samples": len(values),
            }
        )

    delay_slopes = []
    delay_groups: dict[tuple[str, int, float], list[tuple[float, float]]] = defaultdict(list)
    for row in final:
        appended = row.get("condition", {}).get("appended_delay_sec")
        if appended is not None:
            delay_groups[(row["method"], int(row["memory_frames"]), float(row["encode_fps"]))].append(
                (float(appended), float(row["prediction"]["score"]))
            )
    for key, values in sorted(delay_groups.items()):
        slope = None
        if len(values) >= 2 and len({item[0] for item in values}) >= 2:
            slope = float(np.polyfit(np.asarray([item[0] for item in values]), np.asarray([item[1] for item in values]), 1)[0] * 60.0)
        delay_slopes.append(
            {
                "method": key[0],
                "memory_frames": key[1],
                "encode_fps": key[2],
                "accuracy_change_per_delay_minute": slope,
                "samples": len(values),
            }
        )

    auc_rows = []
    method_fps = sorted({(row["method"], float(row["encode_fps"])) for row in final})
    for method, fps in method_fps:
        points = []
        for memory in sorted({int(row["memory_frames"]) for row in final if row["method"] == method and float(row["encode_fps"]) == fps}):
            values = [
                float(row["prediction"]["score"])
                for row in final
                if row["method"] == method and float(row["encode_fps"]) == fps and int(row["memory_frames"]) == memory
            ]
            if values:
                points.append((memory, float(np.mean(values))))
        auc_rows.append({"method": method, "encode_fps": fps, "points": points, "accuracy_budget_auc": _auc(points)})

    critical_scores = [
        float(row["prediction"]["score"])
        for row in rows
        if row.get("stage") == "critical_only"
    ]
    oracle_accuracy = _mean(critical_scores)
    baseline_accuracies = []
    for method in args.baselines:
        values = [float(row["prediction"]["score"]) for row in final if row["method"] == method]
        if values:
            baseline_accuracies.append((method, float(np.mean(values))))
    best_baseline = max(baseline_accuracies, key=lambda pair: pair[1]) if baseline_accuracies else None
    oracle_gap = oracle_accuracy - best_baseline[1] if oracle_accuracy is not None and best_baseline else None
    candidate_worst = min(
        (item["accuracy"] for item in configurations if item["method"] == args.candidate and item["stage"] == "final_memory"),
        default=None,
    )
    baseline_worst_by_method = []
    for method in args.baselines:
        values = [item["accuracy"] for item in configurations if item["method"] == method and item["stage"] == "final_memory"]
        if values:
            baseline_worst_by_method.append((method, min(values)))
    best_baseline_worst = max(baseline_worst_by_method, key=lambda pair: pair[1]) if baseline_worst_by_method else None
    worst_gain = candidate_worst - best_baseline_worst[1] if candidate_worst is not None and best_baseline_worst else None
    candidate_aucs = [item["accuracy_budget_auc"] for item in auc_rows if item["method"] == args.candidate and item["accuracy_budget_auc"] is not None]
    baseline_aucs = [item["accuracy_budget_auc"] for item in auc_rows if item["method"] in args.baselines and item["accuracy_budget_auc"] is not None]
    best_candidate_auc = max(candidate_aucs, default=None)
    best_baseline_auc = max(baseline_aucs, default=None)
    auc_relative = (
        (best_candidate_auc - best_baseline_auc) / best_baseline_auc
        if best_candidate_auc is not None and best_baseline_auc not in {None, 0.0}
        else None
    )
    clean_rows = [
        row
        for row in final
        if row.get("condition", {}).get("distractor_profile") == "clean"
        and float(row.get("condition", {}).get("appended_delay_sec", 0.0) or 0.0) == 0.0
    ]
    candidate_clean_values = [
        float(row["prediction"]["score"]) for row in clean_rows if row["method"] == args.candidate
    ]
    baseline_clean = []
    for method in args.baselines:
        values = [float(row["prediction"]["score"]) for row in clean_rows if row["method"] == method]
        if values:
            baseline_clean.append((method, float(np.mean(values))))
    candidate_clean = _mean(candidate_clean_values)
    best_baseline_clean = max(baseline_clean, key=lambda pair: pair[1]) if baseline_clean else None
    clean_delta = candidate_clean - best_baseline_clean[1] if candidate_clean is not None and best_baseline_clean else None
    gates = {
        "critical_oracle_accuracy": oracle_accuracy,
        "best_baseline": best_baseline,
        "oracle_gap": oracle_gap,
        "oracle_gap_ge_10pp": None if oracle_gap is None else oracle_gap >= 0.10,
        "candidate_worst_group": candidate_worst,
        "best_baseline_worst_group": best_baseline_worst,
        "worst_group_gain": worst_gain,
        "worst_group_gain_ge_5pp": None if worst_gain is None else worst_gain >= 0.05,
        "best_candidate_auc": best_candidate_auc,
        "best_baseline_auc": best_baseline_auc,
        "auc_relative_gain": auc_relative,
        "auc_gain_ge_10pct": None if auc_relative is None else auc_relative >= 0.10,
        "candidate_clean_delay0": candidate_clean,
        "best_baseline_clean_delay0": best_baseline_clean,
        "clean_delay0_delta": clean_delta,
        "clean_delay0_not_worse_than_1pp": None if clean_delta is None else clean_delta >= -0.01,
        "all_budgets_respected": all(bool(row["metrics"].get("budget_respected")) for row in rows),
        "all_queries_no_video_replay": all(row.get("raw_video_replay_at_query") is False for row in rows),
    }
    report = {
        "format_version": 1,
        "inputs": [{"path": str(path.resolve()), "sha256": file_sha256(path)} for path in args.results],
        "rows": len(rows),
        "configurations": configurations,
        "paired_bootstrap": comparisons,
        "interference": interference,
        "delay_slopes": delay_slopes,
        "accuracy_budget_auc": auc_rows,
        "research_gates": gates,
    }
    atomic_json(args.output_dir / "analysis.json", report)
    lines = [
        "# CPU 流式 VLM Pilot 分析",
        "",
        f"有效预测行：{len(rows)}；source-group paired bootstrap：{args.bootstrap_samples} 次。",
        "",
        "## Research gates",
        "",
        f"- Critical-only accuracy：{_percent(oracle_accuracy)}。",
        f"- 相对最佳固定预算 baseline 的 oracle gap：{_percent(oracle_gap)}；10 pp gate：{gates['oracle_gap_ge_10pp']}。",
        f"- Future-utility 最差组增益：{_percent(worst_gain)}；5 pp gate：{gates['worst_group_gain_ge_5pp']}。",
        f"- Accuracy-budget AUC 相对增益：{_percent(auc_relative)}；10% gate：{gates['auc_gain_ge_10pct']}。",
        f"- Clean delay0 相对最佳 baseline：{_percent(clean_delta)}；不劣于 1 pp gate：{gates['clean_delay0_not_worse_than_1pp']}。",
        f"- 全部预算合规：{gates['all_budgets_respected']}；全部查询无原视频回放：{gates['all_queries_no_video_replay']}。",
        "",
        "完整配置、干扰差值和 bootstrap 置信区间见 `analysis.json`。",
        "",
    ]
    (args.output_dir / "report_zh.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote analysis for {len(rows)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()
