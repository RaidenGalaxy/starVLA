from __future__ import annotations

import json
import math
from pathlib import Path

from .progress import atomic_json, atomic_text
from .statistics import retention


def _load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _fmt(value, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{value:.{digits}f}" if math.isfinite(value) else "—"


def _metric_table(metrics: dict[str, dict]) -> list[str]:
    datasets = ("DocVQA_VAL", "MMMU_DEV_VAL", "ScienceQA_TEST", "TextVQA_VAL")
    lines = [
        "| 方案 | DocVQA | MMMU | ScienceQA | TextVQA | 平均 | NLL | PPL | KL | 输出一致率 | TF 秒 | Gen 秒 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in metrics.items():
        scores = value.get("datasets", {})
        lines.append(
            f"| {name} | "
            + " | ".join(_fmt(scores.get(dataset, {}).get("score"), 2) for dataset in datasets)
            + f" | {_fmt(value.get('mean_score'), 2)} | {_fmt(value.get('target_nll'))} "
            f"| {_fmt(value.get('target_ppl'))} | {_fmt(value.get('mean_logits_kl'))} "
            f"| {_fmt(value.get('output_consistency'))} | "
            f"{_fmt(value.get('teacher_seconds'), 1)} | "
            f"{_fmt(value.get('generation_seconds'), 1)} |"
        )
    return lines


def _plot_state_drift(output_dir: Path, state: dict) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return _plot_state_drift_svg(output_dir, state)
    variants = state.get("aggregate", {})
    if not variants:
        return None
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    for name, payload in variants.items():
        by_prefix = payload.get("by_prefix", {})
        x = sorted(int(value) for value in by_prefix)
        y = [by_prefix[str(value)]["normalized_frobenius"] for value in x]
        if x:
            axis.plot(x, y, marker="o", label=name)
    axis.set_xscale("log", base=2)
    axis.set_xlabel("Prefix tokens")
    axis.set_ylabel("Normalized Frobenius state error")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path = output_dir / "STATE_DRIFT_CURVES.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return str(path)


def _plot_state_drift_svg(output_dir: Path, state: dict) -> str | None:
    variants = state.get("aggregate", {})
    series = {}
    for name, payload in variants.items():
        by_prefix = payload.get("by_prefix", {})
        points = sorted(
            (int(prefix), float(values["normalized_frobenius"]))
            for prefix, values in by_prefix.items()
        )
        if points:
            series[name] = points
    if not series:
        return None
    width, height = 760, 460
    left, top, plot_width, plot_height = 80, 30, 640, 350
    all_y = [value for points in series.values() for _, value in points]
    ymax = max(max(all_y), 1e-8) * 1.08
    prefixes = sorted({prefix for points in series.values() for prefix, _ in points})
    xmin, xmax = math.log2(min(prefixes)), math.log2(max(prefixes))

    def xy(prefix, value):
        x = left + (math.log2(prefix) - xmin) / max(xmax - xmin, 1.0) * plot_width
        y = top + plot_height - value / ymax * plot_height
        return x, y

    colors = ("#3366cc", "#dc3912", "#109618", "#ff9900")
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top+plot_height}" x2="{left+plot_width}" y2="{top+plot_height}" stroke="#333"/>',
    ]
    for prefix in prefixes:
        x, _ = xy(prefix, 0)
        elements.extend([
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_height}" stroke="#ddd"/>',
            f'<text x="{x:.1f}" y="{top+plot_height+24}" text-anchor="middle" font-size="13">{prefix}</text>',
        ])
    for index in range(6):
        value = ymax * index / 5
        _, y = xy(prefixes[0], value)
        elements.extend([
            f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_width}" y2="{y:.1f}" stroke="#eee"/>',
            f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="12">{value:.3g}</text>',
        ])
    for index, (name, points) in enumerate(sorted(series.items())):
        color = colors[index % len(colors)]
        coordinates = [xy(prefix, value) for prefix, value in points]
        joined = " ".join(f"{x:.1f},{y:.1f}" for x, y in coordinates)
        elements.append(f'<polyline points="{joined}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x, y in coordinates:
            elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        legend_y = 42 + index * 22
        elements.extend([
            f'<line x1="{left+plot_width-115}" y1="{legend_y}" '
            f'x2="{left+plot_width-90}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>',
            f'<text x="{left+plot_width-82}" y="{legend_y+4}" font-size="13">{name}</text>',
        ])
    elements.extend([
        f'<text x="{left+plot_width/2}" y="{height-25}" text-anchor="middle" font-size="14">Prefix tokens (log2)</text>',
        f'<text x="18" y="{top+plot_height/2}" text-anchor="middle" font-size="14" '
        f'transform="rotate(-90 18 {top+plot_height/2})">Normalized Frobenius state error</text>',
        '</svg>',
    ])
    path = output_dir / "STATE_DRIFT_CURVES.svg"
    atomic_text(path, "\n".join(elements) + "\n")
    return str(path)


def build_report(output_dir: Path) -> dict:
    preflight = _load(output_dir / "preflight" / "preflight.json")
    preflight_diagnostics = _load(output_dir / "preflight" / "diagnostics.json")
    smoke = _load(output_dir / "smoke" / "smoke_summary.json")
    scope = _load(output_dir / "scope" / "scope_sensitivity.json")
    screen = _load(output_dir / "screen" / "screen_summary.json")
    qig = _load(output_dir / "qig" / "qig_summary.json")
    confirm = _load(output_dir / "confirm" / "confirm_summary.json")
    state = _load(output_dir / "confirm" / "state_probe.json")
    missing = [
        name for name, value in (
            ("preflight", preflight), ("smoke", smoke), ("scope", scope), ("screen", screen),
            ("qig", qig), ("confirm", confirm), ("state_probe", state),
        ) if value is None
    ]
    decision = "NO_GO"
    gates = {"complete": not missing}
    if not missing:
        metrics = confirm["metrics"]
        candidate_name = confirm["candidate"]
        candidate_retention = retention(metrics["vg_qig"], metrics["bf16"])
        mean_gain = metrics["vg_qig"]["mean_score"] - metrics["mbq"]["mean_score"]
        nll_gain = 1.0 - metrics["vg_qig"]["target_nll"] / metrics["mbq"]["target_nll"]
        dataset_drops = {
            dataset: metrics["vg_qig"]["datasets"][dataset]["score"]
            - metrics["mbq"]["datasets"][dataset]["score"]
            for dataset in metrics["mbq"]["datasets"]
        }
        mbq_auc = state["aggregate"]["mbq"]["state_drift_auc"]
        qig_auc = state["aggregate"]["vg_qig"]["state_drift_auc"]
        auc_reduction = 1.0 - qig_auc / mbq_auc if mbq_auc > 0 else 0.0
        gates.update({
            "candidate": candidate_name,
            "mean_bf16_retention_at_least_90pct": candidate_retention["mean"] >= 0.90,
            "qig_score_plus_1_or_nll_minus_1pct": mean_gain >= 1.0 or nll_gain >= 0.01,
            "no_dataset_drop_over_2_points": min(dataset_drops.values()) >= -2.0,
            "state_drift_auc_reduction_at_least_10pct": auc_reduction >= 0.10,
            "mean_retention": candidate_retention["mean"],
            "qig_minus_mbq_mean_score": mean_gain,
            "qig_nll_relative_reduction": nll_gain,
            "dataset_score_deltas": dataset_drops,
            "state_drift_auc_relative_reduction": auc_reduction,
        })
        decision = "GO" if all(
            gates[key] for key in (
                "mean_bf16_retention_at_least_90pct",
                "qig_score_plus_1_or_nll_minus_1pct",
                "no_dataset_drop_over_2_points",
                "state_drift_auc_reduction_at_least_10pct",
            )
        ) else "NO_GO"

    plot = _plot_state_drift(output_dir, state or {})
    lines = [
        "# InfiniteVL-4B 纯 CPU 量化探索报告",
        "",
        "> 结果标签：`cpu_reference_backend`、`fake_quantized`、"
        "`no_speed_or_memory_claim`。所有低比特结果均为 FP32 CPU 上的 QDQ 质量模拟，"
        "不能解释为真实低比特内存、延迟或吞吐收益。",
        "",
        f"## 结论：{decision}",
        "",
    ]
    if missing:
        lines.extend([
            f"当前缺少阶段：{', '.join(missing)}。在独立确认与状态门槛齐全前，"
            "LongSFT/流式实验默认 NO_GO。",
            "",
        ])
    else:
        lines.extend([
            "| 门槛 | 结果 |",
            "|---|---|",
            f"| 候选相对 BF16 平均保留率 ≥90% | {gates['mean_bf16_retention_at_least_90pct']} "
            f"({_fmt(100*gates['mean_retention'], 2)}%) |",
            f"| QIG 比 MBQ +1 分或 NLL -1% | {gates['qig_score_plus_1_or_nll_minus_1pct']} |",
            f"| 任一数据集不下降超过 2 分 | {gates['no_dataset_drop_over_2_points']} |",
            f"| state-drift AUC 下降 ≥10% | {gates['state_drift_auc_reduction_at_least_10pct']} "
            f"({_fmt(100*gates['state_drift_auc_relative_reduction'], 2)}%) |",
            "",
        ])
    if preflight is None and preflight_diagnostics is not None:
        failure = preflight_diagnostics.get("failure_message", "尚未完成")
        lines.extend([
            "## CPU 后端预检诊断",
            "",
            f"状态：`{preflight_diagnostics.get('status', 'unknown')}`；最后完成步骤："
            f"`{preflight_diagnostics.get('last_completed_step', 'unknown')}`。",
            "",
            f"失败/暂停原因：`{failure}`。完整可机器读取诊断见 "
            "`preflight/diagnostics.json`。",
            "",
        ])
    if preflight:
        counts = preflight["inventory"]["counts"]
        parity = preflight["backend_parity"]
        lines.extend([
            "## CPU 后端与模块清单",
            "",
            f"固定模型 revision：`{preflight['model_revision']}`；Torch threads："
            f"{preflight['threads']['selected']}；decoder Linear：{counts['decoder_total']}；"
            f"正式 decoder_safe：{counts['decoder_safe']}。",
            "",
            "| CPU parity 项 | 最大误差 | 门槛 |",
            "|---|---:|---:|",
            f"| Gated Delta recurrent/chunk | "
            f"{_fmt(parity['gated_delta_recurrent_chunk_max_error'], 8)} | 1e-4 |",
            f"| ShortConvolution one-shot/cache | "
            f"{_fmt(parity['short_convolution_one_shot_cached_max_error'], 8)} | 1e-5 |",
            f"| RMSNorm × SiLU(gate) | {_fmt(parity['rmsnorm_gate_max_error'], 8)} | 1e-6 |",
            "",
            "| 范围 | Linear 数 | 参数量 | 占全模型参数 |",
            "|---|---:|---:|---:|",
        ])
        labels = {
            "decoder_ffn": "FFN", "swa_attention": "SWA",
            "gdn_main": "GDN main", "gdn_dynamics": "GDN a/b",
        }
        for name, label in labels.items():
            value = preflight["inventory"]["by_scope"][name]
            lines.append(
                f"| {label} | {value['linears']} | {value['parameters']} | "
                f"{_fmt(100 * value['model_parameter_fraction'], 2)}% |"
            )
        lines.append("")
    if smoke:
        lines.extend(["## BF16/FP16 smoke", ""])
        lines.extend(_metric_table(smoke["metrics"]))
        lines.append("")
    if scope:
        lines.extend([
            "## 模块脆弱性",
            "",
            "| 范围 | 参数占比 | NLL 相对增幅 | logits KL | block NMSE |",
            "|---|---:|---:|---:|---:|",
        ])
        for name, value in scope["variants"].items():
            lines.append(
                f"| {name} | {_fmt(100*value.get('parameter_fraction', 0), 2)}% | "
                f"{_fmt(100*value.get('nll_relative_increase', 0), 2)}% | "
                f"{_fmt(value.get('mean_logits_kl'))} | {_fmt(value.get('mean_block_nmse'))} |"
            )
        lines.append("")
    if screen:
        lines.extend(["## 六格式筛选", ""])
        lines.extend(_metric_table(screen["metrics"]))
        lines.extend([
            "",
            f"进入 QIG 的 frontier：`{screen['candidate']}`；低位宽门槛失败回退："
            f"`{screen['fallback_to_w4']}`。",
            "",
        ])
    if confirm:
        lines.extend(["## 独立确认", ""])
        lines.extend(_metric_table(confirm["metrics"]))
        lines.extend([
            "",
            "| 方案 | 改善 | 退化 | 不变 | 预测改变 |",
            "|---|---:|---:|---:|---:|",
        ])
        for name, comparison in confirm.get("comparisons_to_bf16", {}).items():
            counts = comparison["prediction_transitions"]["counts"]
            lines.append(
                f"| {name} | {counts['improved']} | {counts['regressed']} | "
                f"{counts['unchanged']} | {counts['prediction_changed']} |"
            )
        lines.extend(["", "paired bootstrap 使用 2,000 次重采样；逐数据集区间保存在 "
                      "`confirm/confirm_summary.json`。", ""])
    if state:
        lines.extend([
            "## Gated DeltaNet 状态漂移",
            "",
            "| 方案 | state-drift AUC |",
            "|---|---:|",
        ])
        for name, value in state.get("aggregate", {}).items():
            lines.append(f"| {name} | {_fmt(value['state_drift_auc'])} |")
        if plot:
            lines.extend(["", f"![prefix-state drift]({Path(plot).name})"])
        lines.append("")
    lines.extend([
        "## 解释边界",
        "",
        "- FP16 是官方 BF16 值经 FP16 舍入后再扩展至 FP32 计算，不是原生 FP16 checkpoint。",
        "- W4A16/W3A16/W2A16 为 asymmetric group-wise g128；W4A8 为 symmetric "
        "per-output-channel W4 加 per-token dynamic A8。",
        "- 视觉编码器、merger、embedding、LM head、norm、short convolution 未量化；"
        "GDN a/b dynamics 仅出现在 scope 脆弱性实验。",
        "- 本轮未运行 LongSFT、真实长视频或 packed CPU kernel 基准。",
        "",
    ])
    atomic_text(output_dir / "INFINITEVL_CPU_QUANT_REPORT_ZH.md", "\n".join(lines))
    result = {"decision": decision, "gates": gates, "missing_stages": missing, "plot": plot}
    atomic_json(output_dir / "decision.json", result)
    return result
