from __future__ import annotations

import math


def _fmt(value: float, digits: int = 2) -> str:
    return "nan" if not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def render_report(rescore: dict, analysis: dict) -> str:
    lines = [
        "# InfiniteVL FFN 参考精度 4×64 扩展评测",
        "",
        "标签：`cpu_reference_backend`、`fake_quantized`、",
        "`ffn_reference_value_fp32_compute`、`no_speed_or_memory_claim`。",
        "",
        "## 评分器复核",
        "",
        "- QA 主结果使用 `local_deterministic_v2`；旧 scorer 只用于审计对照。",
        "- v2 不再把选项 A 当作英文冠词删除，也不再用回答中的最后一个孤立字母猜选项。",
        f"- 历史 64 条/配置中，BF16 被重评分修改 {rescore['variants']['bf16']['changed_scores']} 条。",
        "",
    ]
    for view, title in (("full256", "完整 256"), ("unseen_holdout128", "独立 holdout 128")):
        lines.extend([
            f"## {title}",
            "",
            "| 变体 | DocVQA | MMMU | ScienceQA | TextVQA | 平均分 | NLL | NLL增幅 | gate概率 | 状态 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ])
        baseline = analysis["metrics"][view]["bf16"]
        dataset_order = ("DocVQA_VAL", "MMMU_DEV_VAL", "ScienceQA_TEST", "TextVQA_VAL")
        lines.append(
            "| BF16 reference | "
            + " | ".join(_fmt(baseline["datasets"][name]["score"]) for name in dataset_order)
            + f" | {_fmt(baseline['mean_score'])} | {_fmt(baseline['target_nll'], 4)} | 0.00% | — | reference |"
        )
        for variant in analysis["variant_order"]:
            metrics = analysis["metrics"][view][variant]
            comparison = analysis["comparisons"][view][variant]
            gate = comparison["gate"]
            lines.append(
                f"| {variant} | "
                + " | ".join(_fmt(metrics["datasets"][name]["score"]) for name in dataset_order)
                + f" | {_fmt(metrics['mean_score'])} | {_fmt(metrics['target_nll'], 4)} | "
                f"{100*gate['target_nll_relative_increase']:.2f}% | "
                f"{100*comparison['bootstrap']['gate_pass_probability']:.1f}% | "
                f"{analysis['classification'][variant]} |"
            )
        lines.append("")
    lines.extend([
        "## 结论",
        "",
        f"- 最终候选：`{analysis['selection']['candidate'] or 'none'}`。",
        f"- 判定：`{analysis['selection']['decision']}`。",
        "- `ROBUST_USABLE` 要求完整集与独立 holdout 的点门槛均通过，且 bootstrap 门槛通过概率均不低于 95%。",
        "- 本实验是 QDQ/fake quantization 质量实验，不作真实内存、速度或吞吐收益声明。",
        "",
    ])
    return "\n".join(lines)

