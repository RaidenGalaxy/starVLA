from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr, wilcoxon

from .core import TOKEN_GROUPS, atomic_json


def read_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError("probe result is empty")
    return rows


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return np.zeros_like(values)
    return (rankdata(values, method="average") - 1.0) / (values.size - 1.0)


def composite_score(delta_nll: np.ndarray, drifts: dict[str, np.ndarray]) -> np.ndarray:
    score = 0.5 * percentile_ranks(delta_nll)
    for group in TOKEN_GROUPS:
        score += percentile_ranks(drifts[group]) / 6.0
    return np.asarray(score, dtype=np.float64)


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = float(spearmanr(left, right).statistic)
    return value if math.isfinite(value) else 1.0


def topk_overlap(left: np.ndarray, right: np.ndarray, k: int = 3) -> float:
    # Higher score means more important, so compare the top-risk layers.
    left_top = set(np.argsort(left)[-k:].tolist())
    right_top = set(np.argsort(right)[-k:].tolist())
    return len(left_top & right_top) / k


def bh_adjust(pvalues: list[float]) -> list[float]:
    count = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.ones(count, dtype=np.float64)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], 1):
        rank = count - reverse_rank + 1
        running = min(running, float(pvalues[index]) * count / rank)
        adjusted[index] = running
    return adjusted.tolist()


def _single_layer_rows(rows: list[dict]) -> tuple[list[str], list[int], dict]:
    indexed = {}
    samples = sorted({str(row["sample_id"]) for row in rows})
    layers = sorted({
        int(row["layer_indices"][0]) for row in rows
        if row["operation"] == "skip" and len(row["layer_indices"]) == 1
    })
    if len(layers) < 3:
        raise ValueError("analysis requires at least three single-layer probes")
    for row in rows:
        if row["operation"] == "skip" and len(row["layer_indices"]) == 1:
            key = (str(row["sample_id"]), str(row["quant_scheme"]), int(row["layer_indices"][0]))
            if key in indexed:
                raise ValueError(f"duplicate single-layer row: {key}")
            indexed[key] = row
    expected = {
        (sample, state, layer)
        for sample in samples for state in ("fp", "mbq_w4") for layer in layers
    }
    missing = expected - set(indexed)
    if missing:
        raise ValueError(f"missing {len(missing)} FP/W4 single-layer rows")
    return samples, layers, indexed


def _matrices(samples: list[str], layers: list[int], indexed: dict, state: str):
    delta = np.empty((len(samples), len(layers)), dtype=np.float64)
    drifts = {
        group: np.empty_like(delta) for group in TOKEN_GROUPS
    }
    for sample_position, sample in enumerate(samples):
        for layer_position, layer in enumerate(layers):
            metrics = indexed[(sample, state, layer)]["metrics"]
            delta[sample_position, layer_position] = metrics["delta_target_nll"]
            for group in TOKEN_GROUPS:
                drifts[group][sample_position, layer_position] = (
                    metrics["final_hidden_cosine_drift"][group]
                )
    if not np.isfinite(delta).all() or any(
        not np.isfinite(values).all() for values in drifts.values()
    ):
        raise ValueError("analysis matrices contain non-finite values")
    return delta, drifts


def _aggregate_score(delta: np.ndarray, drifts: dict[str, np.ndarray], indices=None):
    if indices is None:
        indices = np.arange(delta.shape[0])
    aggregate_delta = np.median(delta[indices], axis=0)
    aggregate_drifts = {
        group: np.median(values[indices], axis=0) for group, values in drifts.items()
    }
    return composite_score(aggregate_delta, aggregate_drifts), aggregate_delta, aggregate_drifts


def _bootstrap_indices(datasets: list[str], rng: np.random.Generator) -> np.ndarray:
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, dataset in enumerate(datasets):
        buckets[dataset].append(index)
    drawn = []
    for dataset in sorted(buckets):
        values = np.asarray(buckets[dataset], dtype=np.int64)
        drawn.extend(rng.choice(values, size=len(values), replace=True).tolist())
    return np.asarray(drawn, dtype=np.int64)


def _interval(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "ci95": [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))],
    }


def analyze(
    rows: list[dict], *, bootstrap_samples: int = 2000, seed: int = 42,
) -> dict:
    samples, layers, indexed = _single_layer_rows(rows)
    sample_dataset = {}
    for row in rows:
        sample_dataset.setdefault(str(row["sample_id"]), str(row["dataset"]))
    datasets = [sample_dataset[sample] for sample in samples]
    fp_delta, fp_drifts = _matrices(samples, layers, indexed, "fp")
    w4_delta, w4_drifts = _matrices(samples, layers, indexed, "mbq_w4")
    fp_score, fp_delta_point, fp_drift_point = _aggregate_score(fp_delta, fp_drifts)
    w4_score, w4_delta_point, w4_drift_point = _aggregate_score(w4_delta, w4_drifts)
    point_rho = safe_spearman(fp_score, w4_score)
    point_overlap = topk_overlap(fp_score, w4_score, min(3, len(layers)))

    group_pairs = list(itertools.combinations(TOKEN_GROUPS, 2))
    token_point = {
        f"{left}_vs_{right}": safe_spearman(
            w4_drift_point[left], w4_drift_point[right],
        )
        for left, right in group_pairs
    }
    bootstrap_rho: list[float] = []
    bootstrap_overlap: list[float] = []
    token_bootstrap = {name: [] for name in token_point}
    interaction_bootstrap = {layer: [] for layer in layers}
    rng = np.random.default_rng(seed)
    interaction = w4_delta - fp_delta
    for _ in range(bootstrap_samples):
        indices = _bootstrap_indices(datasets, rng)
        fp_current, _, _ = _aggregate_score(fp_delta, fp_drifts, indices)
        w4_current, _, w4_group_current = _aggregate_score(w4_delta, w4_drifts, indices)
        bootstrap_rho.append(safe_spearman(fp_current, w4_current))
        bootstrap_overlap.append(
            topk_overlap(fp_current, w4_current, min(3, len(layers)))
        )
        for left, right in group_pairs:
            name = f"{left}_vs_{right}"
            token_bootstrap[name].append(safe_spearman(
                w4_group_current[left], w4_group_current[right],
            ))
        for position, layer in enumerate(layers):
            interaction_bootstrap[layer].append(
                float(np.median(interaction[indices, position]))
            )

    raw_pvalues = []
    interaction_rows = []
    for position, layer in enumerate(layers):
        values = interaction[:, position]
        try:
            pvalue = float(wilcoxon(values, alternative="greater").pvalue)
        except ValueError:
            pvalue = 1.0
        raw_pvalues.append(pvalue)
        denominator = (
            float(np.median(np.abs(fp_delta[:, position])))
            + float(np.median(np.abs(w4_delta[:, position])))
            + 1e-12
        )
        interval = _interval(interaction_bootstrap[layer])
        interaction_rows.append({
            "layer": layer,
            "median": float(np.median(values)),
            "ci95": interval["ci95"],
            "positive_fraction": float(np.mean(values > 0)),
            "standardized_ratio": float(np.median(values)) / denominator,
            "pvalue": pvalue,
        })
    for row, qvalue in zip(interaction_rows, bh_adjust(raw_pvalues)):
        row["bh_qvalue"] = qvalue
        row["passes"] = bool(
            row["ci95"][0] > 0
            and qvalue < 0.05
            and row["standardized_ratio"] >= 0.25
        )

    rank_boot = _interval(bootstrap_rho)
    overlap_boot = _interval(bootstrap_overlap)
    rank_gate = bool(
        (point_rho < 0.7 and rank_boot["ci95"][1] < 0.85)
        or (
            point_overlap <= 1 / 3
            and float(np.mean(np.asarray(bootstrap_overlap) <= 1 / 3)) >= 0.9
        )
    )
    token_results = {}
    token_gate = False
    for name, point in token_point.items():
        interval = _interval(token_bootstrap[name])
        passes = bool(point < 0.6 and interval["ci95"][1] < 0.8)
        token_gate |= passes
        token_results[name] = {"point": point, "bootstrap": interval, "passes": passes}
    interaction_gate = sum(row["passes"] for row in interaction_rows) >= 2
    go = bool(token_gate and (rank_gate or interaction_gate))

    return {
        "schema_version": 1,
        "samples": len(samples),
        "datasets": dict(sorted({name: datasets.count(name) for name in set(datasets)}.items())),
        "layers": layers,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "scores": {
            "fp": {str(layer): float(value) for layer, value in zip(layers, fp_score)},
            "mbq_w4": {str(layer): float(value) for layer, value in zip(layers, w4_score)},
            "fp_delta_nll": {
                str(layer): float(value) for layer, value in zip(layers, fp_delta_point)
            },
            "mbq_w4_delta_nll": {
                str(layer): float(value) for layer, value in zip(layers, w4_delta_point)
            },
        },
        "Q_rank_change": {
            "spearman_point": point_rho,
            "spearman_bootstrap": rank_boot,
            "top3_overlap_point": point_overlap,
            "top3_overlap_bootstrap": overlap_boot,
            "bootstrap_probability_overlap_leq_one_third": float(
                np.mean(np.asarray(bootstrap_overlap) <= 1 / 3)
            ),
            "passes": rank_gate,
        },
        "T_token_divergence": {"comparisons": token_results, "passes": token_gate},
        "I_positive_interaction": {
            "layers": interaction_rows,
            "passing_layers": [row["layer"] for row in interaction_rows if row["passes"]],
            "passes": interaction_gate,
        },
        "gate_rule": "T and (Q or I)",
        "go": go,
    }


def render_report(result: dict, phase: str) -> str:
    q = result["Q_rank_change"]
    t = result["T_token_divergence"]
    interaction = result["I_positive_interaction"]
    decision = (
        "PROCEED_CONFIRMATION" if phase == "discovery" and result["go"]
        else "STOP_AFTER_DISCOVERY" if phase == "discovery"
        else "GO_FULL_36" if result["go"] else "NO_GO"
    )
    lines = [
        f"# QCLP-VLM {phase} 分析报告",
        "",
        f"样本：{result['samples']}；层：{result['layers']}；bootstrap：{result['bootstrap_samples']}。",
        "",
        "## 预注册门限",
        "",
        "| 判据 | 结果 | 通过 |",
        "|---|---:|---|",
        (
            f"| Q：FP/W4 层排序变化 | rho={q['spearman_point']:.4f}, "
            f"95% CI={q['spearman_bootstrap']['ci95']}, top3={q['top3_overlap_point']:.3f} "
            f"| {q['passes']} |"
        ),
        f"| T：token 组排序分歧 | - | {t['passes']} |",
        (
            f"| I：至少两个正交互层 | {interaction['passing_layers']} "
            f"| {interaction['passes']} |"
        ),
        f"| `T and (Q or I)` | **{decision}** | {result['go']} |",
        "",
        "## Token 组比较",
        "",
        "| 比较 | rho | bootstrap 95% CI | 通过 |",
        "|---|---:|---|---|",
    ]
    for name, row in t["comparisons"].items():
        lines.append(
            f"| {name} | {row['point']:.4f} | {row['bootstrap']['ci95']} | {row['passes']} |"
        )
    lines.extend([
        "",
        "## 逐层量化×剪层交互",
        "",
        "| 层 | median I | 95% CI | 正样本比例 | 标准化比例 | BH q | 通过 |",
        "|---:|---:|---|---:|---:|---:|---|",
    ])
    for row in interaction["layers"]:
        lines.append(
            f"| {row['layer']} | {row['median']:.8f} | {row['ci95']} | "
            f"{row['positive_fraction']:.3f} | {row['standardized_ratio']:.3f} | "
            f"{row['bh_qvalue']:.6f} | {row['passes']} |"
        )
    lines.extend([
        "",
        "Discovery 阶段只用于决定是否进入独立确认；只有 confirmatory 阶段的门限结果决定是否扫描全部 36 层。",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze QCLP-VLM layer probes")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--phase", choices=("discovery", "confirmatory"), required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyze(
        read_rows(args.input),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    result["phase"] = args.phase
    atomic_json(args.output_json, result)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(render_report(result, args.phase))


if __name__ == "__main__":
    main()

