from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

from .analyze import bh_adjust
from .core import atomic_json
from .shortv_probe import SCHEMA_NAME


def read_rows(paths: Path | list[Path]) -> list[dict]:
    if isinstance(paths, Path):
        paths = [paths]
    rows = []
    for path in paths:
        with path.open() as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if not rows:
        raise ValueError("Q-ShortV probe result is empty")
    invalid = [row.get("schema_name") for row in rows if row.get("schema_name") != SCHEMA_NAME]
    if invalid:
        raise ValueError("probe rows do not use the Q-ShortV schema")
    return rows


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    value = float(spearmanr(left, right).statistic)
    return value if math.isfinite(value) else 1.0


def bottomk_overlap(left: np.ndarray, right: np.ndarray, k: int = 3) -> float:
    left_bottom = set(np.argsort(left)[:k].tolist())
    right_bottom = set(np.argsort(right)[:k].tolist())
    return len(left_bottom & right_bottom) / k


def _interval(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "ci95": [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))],
    }


def _bootstrap_indices(datasets: list[str], rng: np.random.Generator) -> np.ndarray:
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, dataset in enumerate(datasets):
        buckets[dataset].append(index)
    drawn = []
    for dataset in sorted(buckets):
        values = np.asarray(buckets[dataset], dtype=np.int64)
        drawn.extend(rng.choice(values, size=len(values), replace=True).tolist())
    return np.asarray(drawn, dtype=np.int64)


def _probe_matrices(rows: list[dict], candidate_scheme: str):
    samples = sorted({str(row["sample_id"]) for row in rows})
    layers = sorted({
        int(row["layer_indices"][0])
        for row in rows
        if row["operation"] == "freeze_visual" and len(row["layer_indices"]) == 1
    })
    if len(layers) < 3:
        raise ValueError("Q-ShortV analysis requires at least three single-layer probes")
    indexed = {}
    sample_dataset = {}
    for row in rows:
        sample_dataset.setdefault(str(row["sample_id"]), str(row["dataset"]))
        if row["operation"] != "freeze_visual" or len(row["layer_indices"]) != 1:
            continue
        key = (
            str(row["sample_id"]),
            str(row["quant_scheme"]),
            int(row["layer_indices"][0]),
        )
        if key in indexed:
            raise ValueError(f"duplicate visual-freeze row: {key}")
        indexed[key] = row
    expected = {
        (sample, state, layer)
        for sample in samples for state in ("fp", candidate_scheme) for layer in layers
    }
    missing = expected - set(indexed)
    if missing:
        raise ValueError(
            f"missing {len(missing)} FP/{candidate_scheme} visual-freeze rows"
        )

    matrices = {}
    for state in ("fp", candidate_scheme):
        lc = np.empty((len(samples), len(layers)), dtype=np.float64)
        delta_nll = np.empty_like(lc)
        for sample_position, sample in enumerate(samples):
            for layer_position, layer in enumerate(layers):
                metrics = indexed[(sample, state, layer)]["metrics"]
                lc[sample_position, layer_position] = metrics["answer_logit_kl"]
                delta_nll[sample_position, layer_position] = metrics["delta_target_nll"]
        if not np.isfinite(lc).all() or not np.isfinite(delta_nll).all():
            raise ValueError("Q-ShortV matrices contain non-finite values")
        if (lc < -1e-8).any():
            raise ValueError("answer KL contains a materially negative value")
        matrices[state] = {"lc": np.maximum(lc, 0.0), "delta_nll": delta_nll}
    datasets = [sample_dataset[sample] for sample in samples]
    return samples, layers, datasets, matrices


def _baseline_delta_nll(
    rows: list[dict], samples: list[str], candidate_scheme: str,
) -> np.ndarray:
    indexed = {}
    relevant = {"fp", candidate_scheme}
    for row in rows:
        if row["operation"] != "baseline" or row["quant_scheme"] not in relevant:
            continue
        key = (str(row["sample_id"]), str(row["quant_scheme"]))
        if key in indexed:
            raise ValueError(f"duplicate baseline row: {key}")
        indexed[key] = float(row["metrics"]["target_nll"])
    expected = {(sample, state) for sample in samples for state in relevant}
    missing = expected - set(indexed)
    if missing:
        raise ValueError(f"missing {len(missing)} FP/{candidate_scheme} baseline rows")
    delta = np.asarray([
        indexed[(sample, candidate_scheme)] - indexed[(sample, "fp")]
        for sample in samples
    ], dtype=np.float64)
    if not np.isfinite(delta).all():
        raise ValueError("baseline target-NLL deltas contain non-finite values")
    return delta


def analyze(
    rows: list[dict], *, candidate_scheme: str = "mbq_w4",
    bootstrap_samples: int = 2000, seed: int = 42,
) -> dict:
    if candidate_scheme not in {"mbq_w3", "mbq_w4"}:
        raise ValueError("candidate_scheme must be mbq_w3 or mbq_w4")
    samples, layers, datasets, matrices = _probe_matrices(rows, candidate_scheme)
    fp_lc = matrices["fp"]["lc"]
    quant_lc = matrices[candidate_scheme]["lc"]
    fp_delta = matrices["fp"]["delta_nll"]
    quant_delta = matrices[candidate_scheme]["delta_nll"]
    baseline_delta = _baseline_delta_nll(rows, samples, candidate_scheme)
    fp_point = np.median(fp_lc, axis=0)
    quant_point = np.median(quant_lc, axis=0)
    point_rho = safe_spearman(fp_point, quant_point)
    k = min(3, len(layers))
    point_overlap = bottomk_overlap(fp_point, quant_point, k)

    rng = np.random.default_rng(seed)
    bootstrap_rho: list[float] = []
    bootstrap_overlap: list[float] = []
    baseline_delta_bootstrap: list[float] = []
    interaction = quant_delta - fp_delta
    interaction_bootstrap = {layer: [] for layer in layers}
    for _ in range(bootstrap_samples):
        indices = _bootstrap_indices(datasets, rng)
        fp_current = np.median(fp_lc[indices], axis=0)
        quant_current = np.median(quant_lc[indices], axis=0)
        bootstrap_rho.append(safe_spearman(fp_current, quant_current))
        bootstrap_overlap.append(bottomk_overlap(fp_current, quant_current, k))
        baseline_delta_bootstrap.append(float(np.median(baseline_delta[indices])))
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
            + float(np.median(np.abs(quant_delta[:, position])))
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

    rank_bootstrap = _interval(bootstrap_rho)
    overlap_bootstrap = _interval(bootstrap_overlap)
    overlap_probability = float(
        np.mean(np.asarray(bootstrap_overlap) <= 1 / 3)
    )
    rank_gate = bool(
        (point_rho < 0.7 and rank_bootstrap["ci95"][1] < 0.85)
        or (point_overlap <= 1 / 3 and overlap_probability >= 0.9)
    )
    passing_layers = [row["layer"] for row in interaction_rows if row["passes"]]
    interaction_gate = len(passing_layers) >= 2
    return {
        "schema_name": "qshortv_gate_v1",
        "schema_version": 1,
        "candidate_scheme": candidate_scheme,
        "samples": len(samples),
        "datasets": dict(sorted({name: datasets.count(name) for name in set(datasets)}.items())),
        "layers": layers,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "visual_lc": {
            "fp": {str(layer): float(value) for layer, value in zip(layers, fp_point)},
            candidate_scheme: {
                str(layer): float(value) for layer, value in zip(layers, quant_point)
            },
        },
        "quantization_baseline": {
            "delta_target_nll_median": float(np.median(baseline_delta)),
            "delta_target_nll_bootstrap": _interval(baseline_delta_bootstrap),
            "positive_fraction": float(np.mean(baseline_delta > 0)),
            "interpretation": "diagnostic_only_not_part_of_Q_or_I_gate",
        },
        "Q_visual_rank_change": {
            "spearman_point": point_rho,
            "spearman_bootstrap": rank_bootstrap,
            "bottom3_overlap_point": point_overlap,
            "bottom3_overlap_bootstrap": overlap_bootstrap,
            "bootstrap_probability_overlap_leq_one_third": overlap_probability,
            "passes": rank_gate,
        },
        "I_visual_positive_interaction": {
            "layers": interaction_rows,
            "passing_layers": passing_layers,
            "passes": interaction_gate,
        },
        "gate_rule": "Q_visual_rank_change or I_visual_positive_interaction",
        "go": bool(rank_gate or interaction_gate),
    }


def render_report(result: dict, phase: str) -> str:
    q = result["Q_visual_rank_change"]
    interaction = result["I_visual_positive_interaction"]
    candidate_scheme = result.get("candidate_scheme", "mbq_w4")
    candidate_label = candidate_scheme.upper().replace("MBQ_", "MBQ-")
    decision = (
        "PROCEED_FULL_36" if phase == "discovery" and result["go"]
        else "STOP_AFTER_DISCOVERY" if phase == "discovery"
        else "GO_GPU_VALIDATION" if result["go"] else "NO_GO"
    )
    lines = [
        f"# Q-ShortV {phase} 分析报告",
        "",
        (
            f"样本：{result['samples']}；层：{result['layers']}；"
            f"候选量化：{candidate_label}；bootstrap：{result['bootstrap_samples']}。"
        ),
        "",
        "## 预注册门限",
        "",
        "| 判据 | 结果 | 通过 |",
        "|---|---:|---|",
        (
            f"| Q：FP/{candidate_label} visual-LC 排序变化 | rho={q['spearman_point']:.4f}, "
            f"95% CI={q['spearman_bootstrap']['ci95']}, "
            f"bottom3={q['bottom3_overlap_point']:.3f} | {q['passes']} |"
        ),
        (
            f"| I：至少两个视觉冻结正交互层 | {interaction['passing_layers']} "
            f"| {interaction['passes']} |"
        ),
        f"| `Q or I` | **{decision}** | {result['go']} |",
        "",
        "## 候选量化基线质量诊断",
        "",
        (
            f"相对 FP 的 baseline target-NLL 中位增量为 "
            f"`{result['quantization_baseline']['delta_target_nll_median']:.6f}`，"
            f"bootstrap 95% CI 为 "
            f"`{result['quantization_baseline']['delta_target_nll_bootstrap']['ci95']}`；"
            "该项只用于区分整体量化退化，不参与 Q/I Gate。"
        ),
        "",
        "## 逐层视觉 Layer Contribution",
        "",
        f"| 层 | FP LC | {candidate_label} LC |",
        "|---:|---:|---:|",
    ]
    for layer in result["layers"]:
        lines.append(
            f"| {layer} | {result['visual_lc']['fp'][str(layer)]:.8f} "
            f"| {result['visual_lc'][candidate_scheme][str(layer)]:.8f} |"
        )
    lines.extend([
        "",
        "## 逐层量化×视觉冻结交互",
        "",
        "| 层 | median I | 95% CI | 正样本比例 | 标准化比例 | BH q | 通过 |",
        "|---:|---:|---|---:|---:|---:|---|",
    ])
    for row in interaction["layers"]:
        lines.append(
            f"| {row['layer']} | {row['median']:.8f} | {row['ci95']} "
            f"| {row['positive_fraction']:.3f} | {row['standardized_ratio']:.3f} "
            f"| {row['bh_qvalue']:.6f} | {row['passes']} |"
        )
    lines.extend([
        "",
        "Discovery 失败时不得补扫 36 层；通过后才构造 FP-ShortV 与 Q-ShortV 固定策略。",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Q-ShortV visual-freeze probes")
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--phase", choices=("discovery", "confirmatory"), required=True)
    parser.add_argument(
        "--quant-scheme", choices=("mbq_w3", "mbq_w4"), default="mbq_w4",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    result = analyze(
        read_rows(args.input),
        candidate_scheme=args.quant_scheme,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    result["phase"] = args.phase
    atomic_json(args.output_json, result)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_report.with_suffix(args.output_report.suffix + ".tmp")
    temporary.write_text(render_report(result, args.phase))
    temporary.replace(args.output_report)


if __name__ == "__main__":
    main()
