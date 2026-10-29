from __future__ import annotations

# ruff: noqa: RUF001
import csv
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.infinitevl_quant.progress import atomic_json, atomic_text, file_sha256

from .config import DATASETS, POLICIES

NAMES = {"infinitevl": "InfiniteVL", "qwen25vl_3b": "Qwen2.5-VL-3B", "qwen25vl_7b": "Qwen2.5-VL-7B"}
COLORS = ("#C44E52", "#4C72B0")
LEGENDS = ("Sensitive first", "Sensitive protected")


def save(fig, root, name):
    for extension in ("svg", "png"):
        fig.savefig(root / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def report(config, summary, design):
    root = config.output_dir / "report"
    root.mkdir(parents=True, exist_ok=True)
    partial = " [PARTIAL: incomplete cells excluded]" if summary["partial"] else ""
    csv_rows = []
    for split, filename, title in (
        ("screen32", "nll_layer_sweep", "Layer-by-layer NLL | new screen32"),
        ("confirm128", "nll_independent_confirmation", "Independent NLL confirmation | confirm128"),
    ):
        fig, axes = plt.subplots(3, 2, figsize=(13, 11), squeeze=False)
        for i, (kind, model) in enumerate(summary["models"].items()):
            cohort = model["cohorts"].get(split, {})
            for j, dataset in enumerate(DATASETS):
                ax = axes[i, j]
                scope = cohort.get("scopes", {}).get(dataset, {})
                for p, color, label in zip(POLICIES, COLORS, LEGENDS, strict=True):
                    curve = scope.get("curves", {}).get(p, {})
                    values = curve.get("points", [])
                    if not values:
                        continue
                    x = [r["coverage_percent"] for r in values]
                    # Missing positions are never connected across an unmeasured gap.
                    full_q = (
                        list(range(model["layers"] + 1)) if split == "screen32" else design["models"][kind]["confirm_q"]
                    )
                    byq = {r["q"]: r for r in values}
                    ax.plot(
                        [100 * q / model["layers"] for q in full_q],
                        [byq[q]["metrics"]["nll"] if q in byq else np.nan for q in full_q],
                        color=color,
                        marker="o",
                        ms=3,
                        lw=1.3,
                        label=label,
                    )
                    ci = np.array([r["metrics"]["nll_ci95"] for r in values])
                    ax.vlines(x, ci[:, 0], ci[:, 1], color=color, alpha=0.25, lw=0.7)
                    for row in values:
                        m = row["metrics"]
                        csv_rows.append(
                            {
                                "model": kind,
                                "split": split,
                                "dataset": dataset,
                                "strategy": p,
                                "q": row["q"],
                                "coverage_percent": row["coverage_percent"],
                                "bf16_saving_percent": row["bf16_saving_percent"],
                                "mask_id": row["id"],
                                "samples": m["samples"],
                                "target_tokens": m["target_tokens"],
                                "total_nll": m["total_nll"],
                                "nll": m["nll"],
                                "delta_nll": m["delta_nll"],
                                "nll_ci95_low": m["nll_ci95"][0],
                                "nll_ci95_high": m["nll_ci95"][1],
                            }
                        )
                baseline = scope.get("points", {}).get("B")
                if baseline:
                    ax.axhline(baseline["nll"], color="black", ls=":", lw=1, label="Reference")
                for item in scope.get("random", {}).values():
                    ax.plot(item["coverage_percent"], item["mean"], "s", color="#777777", ms=5)
                    ax.vlines(item["coverage_percent"], *item["seed_range"], color="#777777", lw=3)
                ax.set(
                    xlim=(-1, 101),
                    xlabel="FFN quantization coverage (%)",
                    ylabel="Target NLL (nats/token)",
                    title=f"{NAMES[kind]} | {dataset}",
                )
                ax.grid(alpha=0.2)
                if i == 0 and j == 0:
                    ax.legend(fontsize=8)
        fig.suptitle(title + partial + "\n95% pointwise paired bootstrap CI; y scales vary by panel", fontsize=12)
        fig.text(
            0.5,
            0.005,
            "W4 asymmetric RTN g128 | CPU FP32 fake quant | gray bars: 3-seed range (not sample CI)",
            ha="center",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.025, 1, 0.95))
        save(fig, root, filename)
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), squeeze=False)
    for i, (kind, model) in enumerate(summary["models"].items()):
        for j, dataset in enumerate(DATASETS):
            ax = axes[i, j]
            scope = model["cohorts"].get("screen32", {}).get("scopes", {}).get(dataset, {})
            for p, color, label in zip(POLICIES, COLORS, LEGENDS, strict=True):
                values = scope.get("curves", {}).get(p, {}).get("marginal", [])
                if values:
                    ax.plot(
                        [100 * r["q"] / model["layers"] for r in values],
                        [r["delta"] for r in values],
                        color=color,
                        marker=".",
                        lw=1,
                        label=label,
                    )
                    ci = np.array([r["ci95"] for r in values])
                    ax.vlines(
                        [100 * r["q"] / model["layers"] for r in values],
                        ci[:, 0],
                        ci[:, 1],
                        color=color,
                        alpha=0.2,
                        lw=0.6,
                    )
            ax.axhline(0, color="black", lw=0.7)
            ax.set(
                title=f"{NAMES[kind]} | {dataset}",
                xlabel="FFN quantization coverage (%)",
                ylabel="NLL(q) - NLL(q-1)",
                xlim=(-1, 101),
            )
            ax.grid(alpha=0.2)
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Marginal damage of each newly quantized layer | screen32" + partial)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    save(fig, root, "marginal_layer_damage")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), squeeze=False)
    for ax, (kind, model) in zip(axes[0], summary["models"].items(), strict=True):
        labels, values, lows, highs = [], [], [], []
        for key in design["models"][kind]["qa"]:
            item = model["qa"].get(key)
            if not item:
                continue
            q = item["condition"]["q"]
            label = (
                "Reference"
                if q == 0
                else "All W4"
                if q == model["layers"]
                else "50% sensitive"
                if key == design["models"][kind]["aliases"][POLICIES[0]][str(model["layers"] // 2)]
                else "50% protect"
                if q == model["layers"] // 2
                else "Protect top1"
            )
            labels.append(label)
            m = item["metrics"]
            values.append(m["qa_macro"])
            lows.append(m["qa_ci95"][0])
            highs.append(m["qa_ci95"][1])
        if values:
            ax.bar(np.arange(len(values)), values, color="#4C72B0", alpha=0.8)
            ax.vlines(np.arange(len(values)), lows, highs, color="black", lw=1)
            ax.set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right")
        else:
            ax.text(0.5, 0.5, "No complete QA condition", transform=ax.transAxes, ha="center")
        ax.set(title=NAMES[kind], ylabel="Two-dataset QA macro score (%)", ylim=(0, 100))
    fig.suptitle("QA32 directional check | greedy, max 32 new tokens | pointwise 95% CI")
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    save(fig, root, "qa_sanity_check")
    random_rows, marginal_rows = [], []
    for kind, model in summary["models"].items():
        for split, cohort in model["cohorts"].items():
            for scope, result in cohort["scopes"].items():
                for policy, curve in result["curves"].items():
                    for item in curve.get("marginal", []):
                        marginal_rows.append({"model": kind, "scope": scope, "strategy": policy, **item})
                    if scope == "overall":
                        for row in curve["points"]:
                            m = row["metrics"]
                            csv_rows.append(
                                {
                                    "model": kind,
                                    "split": split,
                                    "dataset": scope,
                                    "strategy": policy,
                                    "q": row["q"],
                                    "coverage_percent": row["coverage_percent"],
                                    "bf16_saving_percent": row["bf16_saving_percent"],
                                    "mask_id": row["id"],
                                    "samples": m["samples"],
                                    "target_tokens": m["target_tokens"],
                                    "total_nll": m["total_nll"],
                                    "nll": m["nll"],
                                    "delta_nll": m["delta_nll"],
                                    "nll_ci95_low": m["nll_ci95"][0],
                                    "nll_ci95_high": m["nll_ci95"][1],
                                }
                            )
                for q, item in result["random"].items():
                    random_rows.append({"model": kind, "scope": scope, "q": int(q), **item})
    for name, values in (
        ("random_controls.csv", random_rows),
        ("marginal_points.csv", marginal_rows),
        ("old_bridge_points.csv", summary.get("old_bridge", [])),
        ("allocation_tests.csv", list(summary["tests"].values())),
    ):
        if values:
            stream = io.StringIO()
            columns = list(dict.fromkeys(k for value in values for k in value))
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(values)
            atomic_text(root / name, stream.getvalue())
    if csv_rows:
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
        atomic_text(root / "curve_points.csv", stream.getvalue())
    lines = [
        "# 三模型W4逐层扫描与跨数据集验证",
        "",
        f"核心实验状态：{'完整' if summary['core_complete'] else '未完整；只展示完整条件'}。",
        "",
        "横轴为FFN量化覆盖率0–100%；理论BF16权重节省率为其0.75倍。",
        "计算限定为CPU FP32 fake-quant，不代表实际低比特内存或速度。",
        "",
        "| 模型 | 数据集 | 策略 | 逐层NLL Spearman | 95%区间 |",
        "|---|---|---|---:|---|",
    ]
    for kind, model in summary["models"].items():
        for dataset in DATASETS:
            scope = model["cohorts"].get("screen32", {}).get("scopes", {}).get(dataset, {})
            for policy in POLICIES:
                stat = scope.get("curves", {}).get(policy, {}).get("spearman")
                if stat:
                    lines.append(f"| {NAMES[kind]} | {dataset} | {policy} | {stat['rho']} | {stat['ci95']} |")
    lines += ["", "| 模型 | 数据集 | 参考NLL → 全W4 NLL | 确认集中保护优势通过Holm的比例数 |", "|---|---|---|---:|"]
    for kind, model in summary["models"].items():
        for dataset in DATASETS:
            points = model["cohorts"].get("confirm128", {}).get("scopes", {}).get(dataset, {}).get("points", {})
            if "B" not in points or "W" not in points:
                continue
            significant = sum(
                t["model"] == kind
                and t["scope"] == dataset
                and t.get("p_holm") is not None
                and t["p_holm"] < 0.05
                and t["protection_advantage_nll"] > 0
                for t in summary["tests"].values()
            )
            lines.append(
                f"| {NAMES[kind]} | {dataset} | {points['B']['nll']:.6f} → {points['W']['nll']:.6f} | {significant}/4 |"
            )
    lines += [
        "",
        f"独立确认比较：{len(summary['tests'])}/36；Holm校正完整：{summary.get('holm_family_complete', False)}。",
        "",
        "旧screen32包含历史排序样本；新confirm128按图像隔离，承担独立确认。",
        "ChartQA训练集曾用于历史校准，本轮测试图像已排除与历史样本重复。",
        "随机种子范围不是样本置信区间；QA32仅用于方向核验，未显著不代表等效。",
        "局部边际损伤依赖当前量化背景，不代表单层固有敏感度。",
        "",
        "完整统计与逐数据集结果见analysis.json；图表、CSV及原始逐样本文件共同交付。",
    ]
    atomic_text(root / "report_zh.md", "\n".join(lines) + "\n")
    atomic_json(root / "analysis.json", summary)
    artifacts = {p.name: file_sha256(p) for p in sorted(root.iterdir()) if p.is_file() and p.name != "artifacts.json"}
    atomic_json(root / "artifacts.json", artifacts)
    return {"report": str(root / "report_zh.md"), "artifacts": artifacts}
