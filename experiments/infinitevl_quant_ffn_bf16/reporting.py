from __future__ import annotations

import math


def _fmt(value: float, digits: int = 4) -> str:
    return "nan" if not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def render_report(audit: dict, screen: dict, confirm: dict) -> str:
    lines = [
        "# InfiniteVL FFN 参考精度注意力量化报告",
        "",
        "标签：`cpu_reference_backend`、`fake_quantized`、"
        "`ffn_reference_value_fp32_compute`、`no_speed_or_memory_claim`。",
        "",
        "## A–D 复核",
        "",
        f"- 原始指标文件复算：{audit['source_audit']['metrics_checked']} 个，"
        f"聚合错误 {audit['source_audit']['aggregate_errors']} 个，"
        f"非有限值 {audit['source_audit']['nonfinite_metrics']} 个。",
        "- BF16 指官方 BF16 checkpoint 值扩展至 FP32 存储和计算。",
        "- 原 Layer/Projection 排名使用的 scan 8 条属于 search 32 条子集；"
        "本报告的 confirm 32 条为独立未使用集合。",
        "- A–D 的 Shapley、Layer 与 Projection 主要归因 Target NLL，不等同于独立 QA 显著性。",
        "",
        "## Screen 复现",
        "",
    ]
    for key, row in sorted(screen["reproduction"].items()):
        lines.append(
            f"- {key}: 新 NLL {_fmt(row['observed'])}，原 NLL {_fmt(row['expected'])}，"
            f"绝对差 {_fmt(row['absolute_error'], 8)}。"
        )
    lines.extend([
        "",
        "## 独立 Confirm",
        "",
        "| 变体 | 平均分 | Target NLL | NLL增幅 | 平均保留率 | 最差保留率 | 判定 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    baseline = confirm["metrics"]["bf16"]
    lines.append(
        f"| BF16 reference | {_fmt(baseline['mean_score'], 2)} | "
        f"{_fmt(baseline['target_nll'])} | 0.00% | 100.00% | 100.00% | reference |"
    )
    for variant, metrics in confirm["metrics"].items():
        if variant == "bf16":
            continue
        gate = confirm["comparisons"][variant]["gate"]
        lines.append(
            f"| {variant} | {_fmt(metrics['mean_score'], 2)} | {_fmt(metrics['target_nll'])} | "
            f"{100*gate['target_nll_relative_increase']:.2f}% | "
            f"{100*gate['mean_retention']:.2f}% | {100*gate['worst_retention']:.2f}% | "
            f"{'USABLE' if gate['passed'] else 'FAIL'} |"
        )
    selection = confirm["selection"]
    lines.extend([
        "",
        "## 结论",
        "",
        f"- 最终候选：`{selection['candidate'] or 'none'}`。",
        f"- 判定：`{selection['decision']}`。",
        "- 量化覆盖仅为模型参数的一部分，不作真实内存、速度或吞吐收益声明。",
        "",
    ])
    return "\n".join(lines)

