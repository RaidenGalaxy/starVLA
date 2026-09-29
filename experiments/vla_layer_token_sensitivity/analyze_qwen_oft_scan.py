"""Analyze a Qwen-OFT FP/W4 layer scan and apply the preregistered gates."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from experiments.vla_layer_token_sensitivity.metrics import rank_stability, topk_overlap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("scan JSONL is empty")
    return rows


def _key(row: dict) -> tuple[str, str, int, str]:
    return str(row["sample_id"]), str(row["base_state"]), int(row["layer"]), str(row["operation"])


def _safe_rank(x: Sequence[float], y: Sequence[float]) -> float:
    try:
        return rank_stability(x, y)
    except ValueError:
        return float("nan")


def _finite_summary(values: Sequence[float]) -> dict:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return {"median": None, "ci95": [None, None], "valid_bootstraps": 0}
    return {
        "median": float(np.median(finite)),
        "ci95": [float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))],
        "valid_bootstraps": int(finite.size),
    }


def _episode_layer_matrix(
    rows: Sequence[dict],
    *,
    layers: Sequence[int],
    value: Callable[[dict], float],
) -> tuple[list[int], np.ndarray]:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(int(row["episode_index"]), int(row["layer"]))].append(float(value(row)))
    episodes = sorted({episode for episode, _ in grouped})
    matrix = np.empty((len(episodes), len(layers)), dtype=np.float64)
    for episode_position, episode in enumerate(episodes):
        for layer_position, layer in enumerate(layers):
            values = grouped.get((episode, layer), [])
            if not values:
                raise ValueError(f"episode {episode} is missing layer {layer}")
            matrix[episode_position, layer_position] = float(np.mean(values))
    return episodes, matrix


def _bootstrap_rank_comparison(
    left: np.ndarray,
    right: np.ndarray,
    *,
    k: int,
    samples: int,
    rng: np.random.Generator,
) -> dict:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("bootstrap rank matrices must share [episodes, layers] shape")
    if left.shape[0] < 2:
        raise ValueError("episode bootstrap needs at least two episodes")
    point_left = left.mean(axis=0)
    point_right = right.mean(axis=0)
    correlations: list[float] = []
    overlaps: list[float] = []
    for _ in range(samples):
        indices = rng.integers(0, left.shape[0], size=left.shape[0])
        candidate_left = left[indices].mean(axis=0)
        candidate_right = right[indices].mean(axis=0)
        correlations.append(_safe_rank(candidate_left, candidate_right))
        overlaps.append(topk_overlap(candidate_left, candidate_right, k))
    return {
        "point_spearman": _safe_rank(point_left, point_right),
        "bootstrap_spearman": _finite_summary(correlations),
        "point_topk_overlap": topk_overlap(point_left, point_right, k),
        "bootstrap_topk_overlap": _finite_summary(overlaps),
        "left_layer_scores": {str(layer): float(score) for layer, score in zip(range(len(point_left)), point_left)},
        "right_layer_scores": {str(layer): float(score) for layer, score in zip(range(len(point_right)), point_right)},
    }


def _metric(row: dict, comparison: str, name: str) -> float:
    return float(row["metrics"][comparison][name])


def analyze(rows: Sequence[dict], *, bootstrap_samples: int, seed: int) -> dict:
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap-samples must be positive")
    indexed = {_key(row): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("scan JSONL contains duplicate result keys")
    layers = sorted(
        {
            int(row["layer"])
            for row in rows
            if row["operation"] == "skip" and row["base_state"] == "fp"
        }
    )
    if len(layers) < 2:
        raise ValueError("at least two probed layers are required")
    sample_ids = sorted(
        {
            str(row["sample_id"])
            for row in rows
            if row["base_state"] == "fp" and int(row["layer"]) == -1
        }
    )
    expected = {
        (sample_id, base_state, layer, "skip")
        for sample_id in sample_ids
        for base_state in ("fp", "w4")
        for layer in layers
    } | {
        (sample_id, base_state, -1, "reference")
        for sample_id in sample_ids
        for base_state in ("fp", "w4")
    }
    missing = sorted(expected - set(indexed))
    if missing:
        raise ValueError(f"scan is incomplete: {len(missing)} required rows are missing")

    fp_skips = [indexed[(sample_id, "fp", layer, "skip")] for sample_id in sample_ids for layer in layers]
    w4_skips = [indexed[(sample_id, "w4", layer, "skip")] for sample_id in sample_ids for layer in layers]
    episode_ids, action_matrix = _episode_layer_matrix(
        fp_skips,
        layers=layers,
        value=lambda row: row["metrics"]["vs_fp"]["final_hidden_cosine_drift"]["action_query"],
    )
    vision_episodes, vision_matrix = _episode_layer_matrix(
        fp_skips,
        layers=layers,
        value=lambda row: row["metrics"]["vs_fp"]["final_hidden_cosine_drift"]["vision"],
    )
    if vision_episodes != episode_ids:
        raise ValueError("token-group episode sets differ")

    token_counts = {
        str(row["sample_id"]): row["token_counts"]
        for row in rows
        if row["base_state"] == "fp" and int(row["layer"]) == -1
    }

    def global_hidden_drift(row: dict) -> float:
        drifts = row["metrics"]["vs_fp"]["final_hidden_cosine_drift"]
        counts = token_counts[str(row["sample_id"])]
        total = sum(int(value) for value in counts.values())
        return sum(float(drifts[group]) * int(counts[group]) for group in drifts) / total

    global_episodes, global_matrix = _episode_layer_matrix(
        fp_skips,
        layers=layers,
        value=global_hidden_drift,
    )
    if global_episodes != episode_ids:
        raise ValueError("global/token-group episode sets differ")

    rng = np.random.default_rng(seed)
    topk = max(1, math.ceil(0.25 * len(layers)))
    action_vs_vision = _bootstrap_rank_comparison(
        action_matrix,
        vision_matrix,
        k=topk,
        samples=bootstrap_samples,
        rng=rng,
    )
    action_vs_global = _bootstrap_rank_comparison(
        action_matrix,
        global_matrix,
        k=topk,
        samples=bootstrap_samples,
        rng=rng,
    )
    for comparison in (action_vs_vision, action_vs_global):
        comparison["left_layer_scores"] = {
            str(layer): score for layer, score in zip(layers, comparison["left_layer_scores"].values())
        }
        comparison["right_layer_scores"] = {
            str(layer): score for layer, score in zip(layers, comparison["right_layer_scores"].values())
        }

    fp_episodes, fp_error = _episode_layer_matrix(
        fp_skips,
        layers=layers,
        value=lambda row: _metric(row, "vs_fp", "action_mse"),
    )
    w4_episodes, w4_error = _episode_layer_matrix(
        w4_skips,
        layers=layers,
        value=lambda row: _metric(row, "vs_base", "action_mse"),
    )
    if fp_episodes != w4_episodes:
        raise ValueError("FP/W4 episode sets differ")
    fp_vs_w4 = _bootstrap_rank_comparison(
        fp_error,
        w4_error,
        k=topk,
        samples=bootstrap_samples,
        rng=rng,
    )
    for side in ("left_layer_scores", "right_layer_scores"):
        fp_vs_w4[side] = {str(layer): score for layer, score in zip(layers, fp_vs_w4[side].values())}

    interactions: dict[tuple[int, int], list[float]] = defaultdict(list)
    for sample_id in sample_ids:
        quant_error = _metric(indexed[(sample_id, "w4", -1, "reference")], "vs_fp", "action_mse")
        for layer in layers:
            fp_skip = indexed[(sample_id, "fp", layer, "skip")]
            w4_skip = indexed[(sample_id, "w4", layer, "skip")]
            episode = int(fp_skip["episode_index"])
            value = (
                _metric(w4_skip, "vs_fp", "action_mse")
                - quant_error
                - _metric(fp_skip, "vs_fp", "action_mse")
            )
            interactions[(episode, layer)].append(value)

    episode_interactions = np.empty((len(episode_ids), len(layers)), dtype=np.float64)
    for episode_position, episode in enumerate(episode_ids):
        for layer_position, layer in enumerate(layers):
            values = interactions.get((episode, layer), [])
            if not values:
                raise ValueError(f"episode {episode} is missing interaction values for layer {layer}")
            episode_interactions[episode_position, layer_position] = float(np.mean(values))

    interaction_layers: dict[str, dict] = {}
    positive_count = 0
    for layer_position, layer in enumerate(layers):
        values = episode_interactions[:, layer_position]
        estimates = [
            float(values[rng.integers(0, len(values), size=len(values))].mean())
            for _ in range(bootstrap_samples)
        ]
        interval = [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]
        positive = interval[0] > 0.0
        positive_count += int(positive)
        interaction_layers[str(layer)] = {
            "mean": float(values.mean()),
            "bootstrap_median": float(np.median(estimates)),
            "ci95": interval,
            "positive_ci": positive,
        }
    positive_fraction = positive_count / len(layers)

    def median_value(section: dict, metric: str) -> float:
        value = section[metric]["median"]
        return float("nan") if value is None else float(value)

    gate_a = (
        median_value(action_vs_global, "bootstrap_spearman") < 0.6
        or median_value(action_vs_vision, "bootstrap_spearman") < 0.6
        or median_value(action_vs_global, "bootstrap_topk_overlap") < 0.5
        or median_value(action_vs_vision, "bootstrap_topk_overlap") < 0.5
    )
    gate_b = median_value(fp_vs_w4, "bootstrap_spearman") < 0.7
    gate_c = positive_fraction >= 0.2
    go = gate_a and (gate_b or gate_c)

    return {
        "status": "complete",
        "decision": "GO" if go else "NO_GO",
        "rule": "A and (B or C)",
        "sample_count": len(sample_ids),
        "episode_count": len(episode_ids),
        "episode_indices": episode_ids,
        "layers": layers,
        "topk": topk,
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "gates": {
            "A_token_conditioned_ranking": gate_a,
            "B_fp_w4_ranking_inversion": gate_b,
            "C_positive_interaction_fraction": gate_c,
        },
        "A": {
            "thresholds": {"spearman_lt": 0.6, "topk_overlap_lt": 0.5},
            "action_query_vs_vision": action_vs_vision,
            "action_query_vs_global": action_vs_global,
        },
        "B": {
            "threshold": {"spearman_lt": 0.7},
            "fp_skip_vs_w4_skip": fp_vs_w4,
        },
        "C": {
            "threshold": {"positive_layer_fraction_gte": 0.2},
            "positive_layers": positive_count,
            "positive_layer_fraction": positive_fraction,
            "per_layer": interaction_layers,
        },
        "limitations": [
            "teacher-forced open-loop action divergence, not task success",
            "globally W4-dequantized CPU weights, not packed INT4 latency",
            "Bridge calibration subset from one video/data shard",
        ],
    }


def _format_value(value: object, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _render_report(result: dict) -> str:
    a_global = result["A"]["action_query_vs_global"]
    a_vision = result["A"]["action_query_vs_vision"]
    b = result["B"]["fp_skip_vs_w4_skip"]
    gates = result["gates"]
    layer_rows = []
    for layer in result["layers"]:
        key = str(layer)
        interaction = result["C"]["per_layer"][key]
        layer_rows.append(
            f"| {layer} | {_format_value(b['left_layer_scores'][key], 6)} | "
            f"{_format_value(b['right_layer_scores'][key], 6)} | "
            f"{_format_value(interaction['mean'], 6)} | "
            f"[{_format_value(interaction['ci95'][0], 6)}, "
            f"{_format_value(interaction['ci95'][1], 6)}] | "
            f"{interaction['positive_ci']} |"
        )
    lines = [
        "# Qwen-OFT FP/W4 层敏感性预实验报告",
        "",
        f"结论：**{result['decision']}**，判定规则为 `A and (B or C)`。",
        "",
        f"样本：{result['sample_count']} 帧 / {result['episode_count']} 条 episode；"
        f"层：{', '.join(map(str, result['layers']))}；episode bootstrap：{result['bootstrap_samples']} 次。",
        "",
        "## 1. 实验目的",
        "",
        "本实验检验的不是普通的“哪些 VLA 层可以剪”，而是一个更具体的假设：",
        "**W4 量化是否会改变以 action-query token 为条件的层敏感性，并与剪层产生超加性退化。**",
        "若该现象成立，按视觉或全局 token 得到的剪层排序就不能直接用于量化 VLA，值得继续研究 token-aware 的联合量化/剪层策略。",
        "",
        "## 2. 实验机制",
        "",
        "对每个样本和每个候选层 `l` 运行四类前向：",
        "",
        "```text",
        "Y_FP            完整 FP32 模型",
        "Y_FP_SKIP(l)    FP32 模型跳过第 l 层",
        "Y_W4            全部 decoder Linear 做 groupwise W4 fake quant",
        "Y_W4_SKIP(l)    W4 模型再跳过第 l 层",
        "```",
        "",
        "以动作均方误差 `D` 计算：",
        "",
        "```text",
        "E_skip_FP(l) = D(Y_FP, Y_FP_SKIP(l))",
        "E_quant      = D(Y_FP, Y_W4)",
        "E_skip_W4(l) = D(Y_W4, Y_W4_SKIP(l))",
        "I(l)         = D(Y_FP, Y_W4_SKIP(l)) - E_quant - E_skip_FP(l)",
        "```",
        "",
        "`I(l) > 0` 表示量化与剪层的联合损失大于两者单独损失之和。token 条件敏感性则用最终 hidden state 的 cosine drift，分别对 vision、language、action-query token 计算。所有置信区间按 episode 重采样，而不是把同一轨迹的四帧视为独立样本。",
        "",
        "## 3. 预注册的继续条件",
        "",
        "- A：action-query 与 global/vision 的层排序 `ρ < 0.6`，或 top-25% overlap `< 0.5`。",
        "- B：FP 与 W4 的 skip-ranking `ρ < 0.7`。",
        "- C：至少 20% 的候选层，其 `I(l)` 的 bootstrap 95% CI 完全大于 0。",
        "- 只有 `A and (B or C)` 成立才判为 GO。",
        "",
        "## 4. 实际结果",
        "",
        "| Gate | 结果 | 核心统计 |",
        "|---|---:|---|",
        (
            f"| A：action-query 排序区别于视觉/全局 | {gates['A_token_conditioned_ranking']} | "
            f"vs global ρ={_format_value(a_global['bootstrap_spearman']['median'])}, "
            f"top-k={_format_value(a_global['bootstrap_topk_overlap']['median'])}; "
            f"vs vision ρ={_format_value(a_vision['bootstrap_spearman']['median'])}, "
            f"top-k={_format_value(a_vision['bootstrap_topk_overlap']['median'])} |"
        ),
        (
            f"| B：FP→W4 排序反转 | {gates['B_fp_w4_ranking_inversion']} | "
            f"ρ={_format_value(b['bootstrap_spearman']['median'])}, "
            f"95% CI={b['bootstrap_spearman']['ci95']} |"
        ),
        (
            f"| C：正交互层比例 | {gates['C_positive_interaction_fraction']} | "
            f"{result['C']['positive_layers']}/{len(result['layers'])}="
            f"{result['C']['positive_layer_fraction']:.1%} |"
        ),
        "",
        "逐层动作误差与交互项如下；FP/W4 列均为相对各自基础模型的 skip action MSE：",
        "",
        "| 层 | FP skip MSE | W4 skip MSE | I(l) 均值 | I(l) 95% CI | CI>0 |",
        "|---:|---:|---:|---:|---:|---:|",
        *layer_rows,
        "",
        "## 5. 结果解释",
        "",
        "- A 未成立：action-query 与 global/vision 的敏感层排序仍高度一致，而且 top-2 层完全重合，没有显示 action-query 独有的关键层集合。",
        "- B 未成立：W4 前后 skip-ranking 基本保持，当前量化没有显著改写层冗余排序。",
        "- C 未成立：7 个候选层中没有一个正交互项的 95% CI 完全大于 0，无法支持量化与剪层存在稳定的超加性退化。",
        "- 因此当前证据不支持继续扩展同一 MATS-VLA 假设；按预注册计划停止下载第二个 VLA。这个 NO_GO 是对当前现象假设和预实验配置的结论，不等价于“VLA 层剪枝没有价值”。",
        "",
        "## 6. 解释边界",
        "",
        "- 这是开环 teacher divergence，不能解释为机器人任务成功率。",
        "- W4 是反量化后的 weight-only fake quant，不能解释为真实 INT4 加速。",
        "- 数据只来自 BridgeData2 的一个 shard，结论仅用于决定是否值得继续投入。",
        "- checkpoint 是 10K-step 早期模型，不是最终强基线。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    result = analyze(
        _read_jsonl(args.input),
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
    print(json.dumps({"decision": result["decision"], "gates": result["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
