# ruff: noqa: RUF001
from __future__ import annotations

import csv
import json

from experiments.infinitevl_quant.progress import atomic_json

from .config import MODELS, POLICIES, QA_COUNTS

NAMES = {"sensitive_first": "Sensitive first", "sensitive_protected": "Sensitive protected"}


def plots(output, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    for split, metric, filename, ylabel in (
        ("confirm192", "qa_macro", "qa_vs_compression", "Macro QA score (%)"),
        ("screen32", "nll", "nll_vs_compression", "Token-weighted NLL"),
    ):
        if not any(split in v["cohorts"] for v in summary["models"].values()):
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=metric == "qa_macro", layout="constrained")
        for axis, kind in zip(axes, MODELS, strict=True):
            cohort = summary["models"][kind]["cohorts"].get(split)
            axis.set_title("InfiniteVL" if kind == "infinitevl" else "Qwen")
            axis.set_xlabel("FFN theoretical weight savings vs BF16 (%)")
            axis.set_ylabel(ylabel)
            axis.set_xlim(-1, 76)
            axis.grid(alpha=0.2)
            if metric == "qa_macro":
                axis.set_ylim(0, 100)
            top = axis.secondary_xaxis("top", functions=(lambda c: c * 0.48, lambda q: q / 0.48))
            top.set_xlabel("FFN layers at W4 (of 36)")
            top.set_xticks(QA_COUNTS)
            if cohort is None:
                axis.text(0.5, 0.5, "Pending", transform=axis.transAxes, ha="center")
                continue
            for policy, color in zip(POLICIES, ("#C44E52", "#4C72B0"), strict=True):
                sequence = cohort["policy_points"][policy]
                for i, row in enumerate(sequence):
                    point = row["metrics"]
                    if point is None or point.get(metric) is None:
                        continue
                    x, y = row["compression_percent"], point[metric]
                    bounds = point["qa_ci95" if metric == "qa_macro" else "nll_ci95"]
                    if metric == "qa_macro":
                        # Percentile CI can exclude an unusual point estimate; draw its actual limits.
                        axis.vlines(x, bounds[0], bounds[1], color=color, alpha=0.6)
                        axis.plot(x, y, "o", color=color, markersize=4)
                    else:
                        axis.plot(x, y, ".", color=color, markersize=3)
                    if i and sequence[i - 1]["metrics"] is not None:
                        prior = sequence[i - 1]
                        axis.plot([prior["compression_percent"], x], [prior["metrics"][metric], y], color=color, lw=1.5)
                axis.plot([], [], color=color, marker="o", label=NAMES[policy])
            reference = cohort["points"].get("B")
            if reference:
                axis.axhline(reference[metric], color="black", linestyle=":", lw=1, label="Reference")
            if metric == "nll" and cohort.get("random_points"):
                rows = cohort["random_points"]
                x = [r["compression_percent"] for r in rows]
                axis.plot(x, [r["nll_mean"] for r in rows], "--x", color="gray", label="Random order mean")
                axis.fill_between(
                    x,
                    [r["nll_min"] for r in rows],
                    [r["nll_max"] for r in rows],
                    color="gray",
                    alpha=0.15,
                    label="3-seed range (not CI)",
                )
            axis.legend(fontsize=8, loc="best")
        fig.suptitle(
            "Historical confirm192; pointwise sample-bootstrap 95% CI"
            if metric == "qa_macro"
            else "screen32; two dense allocation curves and sparse random controls",
            fontsize=11,
        )
        fig.supxlabel("CPU FP32 fake-quant; theoretical FFN weight bits only; packing metadata excluded", fontsize=9)
        for suffix in ("svg", "png"):
            path = output / f"{filename}.{suffix}"
            fig.savefig(path, dpi=220)
            files.append(str(path))
        plt.close(fig)
    return files


def report(config, summary=None):
    output = config.output_dir / "report"
    output.mkdir(parents=True, exist_ok=True)
    status_path = config.output_dir / "status.json"
    state = json.loads(status_path.read_text()) if status_path.exists() else {"status": "pending"}
    if summary is None:
        path = config.output_dir / "analyze/summary.json"
        summary = json.loads(path.read_text()) if path.exists() else None
    if summary is None:
        text = (
            "# FFN 压缩比例—质量曲线\n\n当前状态："
            + state["status"]
            + "。正式条件尚未完成，图表将在完整条件保存后更新。\n"
        )
        (output / "report_zh.md").write_text(text)
        return {"experiment_complete": False, "report": str(output / "report_zh.md")}
    atomic_json(output / "analysis_snapshot.json", summary)
    rows = []
    for kind, model in summary["models"].items():
        for split, cohort in model["cohorts"].items():
            for policy, points in {**cohort["policy_points"], **cohort.get("random_seed_points", {})}.items():
                for row in points:
                    m = row["metrics"]
                    if m is None:
                        continue
                    rows.append(
                        {
                            "model": kind,
                            "cohort": split,
                            "policy": policy,
                            "q": row["q"],
                            "compression_percent": row["compression_percent"],
                            "mask": m["condition"]["mask_hex"],
                            "mixed_weight_bits": m["condition"]["mixed_weight_bits"],
                            "samples": m["samples"],
                            "target_tokens": m["target_tokens"],
                            "total_nll": m["total_nll"],
                            "nll": m["nll"],
                            "nll_ci_low": m["nll_ci95"][0],
                            "nll_ci_high": m["nll_ci95"][1],
                            "qa_macro": m["qa_macro"],
                            "qa_ci_low": m["qa_ci95"][0] if m["qa_ci95"] else None,
                            "qa_ci_high": m["qa_ci95"][1] if m["qa_ci95"] else None,
                            "reused_rows": m["reused_rows"],
                        }
                    )
    with (output / "curve_points.csv").open("w", newline="") as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    pictures = plots(output, summary)
    complete = not summary["partial"] and summary.get("verification", {}).get("status") == "passed"
    lines = [
        "# FFN 压缩比例—质量曲线",
        "",
        f"运行状态：{state['status']}；完整核算：{complete}。",
        "",
        "横轴为相对 BF16 的 FFN 理论权重节省率 C=q/48，范围0–75%，不含元数据。"
        "CPU FP32 fake-quant，无原生低比特内存或速度声明。",
        "",
        "QA 使用历史 confirm192、本轮重新推理或显式核验复用；NLL 扫描使用 screen32。"
        "两种样本集分别报告，未完成点不参与曲线统计。",
        "",
        "![QA主图](qa_vs_compression.svg)",
        "",
        "![NLL辅图](nll_vs_compression.svg)",
        "",
        "## 曲线相关性",
        "",
        "区间为按样本配对、按数据集分层的bootstrap95%区间。相关系数属于描述性趋势，横轴点和随机种子不作为独立样本。",
        "",
        "| 模型 | 样本集 | 策略与指标 | Spearman rho | 95% CI |",
        "|---|---|---|---:|---|",
    ]
    for kind, model in summary["models"].items():
        for split, cohort in model["cohorts"].items():
            for key, entry in cohort["correlations"].items():
                lines.append(f"| {kind} | {split} | {key} | {entry['rho']} | {entry['ci95']} |")
    lines.extend(
        [
            "",
            "## 同压缩率的层分配效应",
            "",
            "正值表示保护敏感层更好：NLL为先压缩减先保护，QA为先保护减先压缩。16项双侧配对符号检验统一Holm校正；区间为点对点95%区间。",
            "",
            "| 模型 | W4层数 | 指标 | 保护优势 | 95% CI | Holm p |",
            "|---|---:|---|---:|---|---:|",
        ]
    )
    for entry in summary["tests"].values():
        lines.append(
            f"| {entry['model']} | {entry['q']} | {entry['metric']} | "
            f"{entry['advantage_of_protection']:.6f} | {entry['ci95']} | {entry['p_holm']} |"
        )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "相关趋势与相同压缩率下的分配效应分别解释；未显著不视为等效，不宣称全局最优混合精度。",
            "",
            f"预算省略项：{summary['budgeted_omissions']}。",
            "",
            f"独立核算：{summary.get('verification', {'status': 'pending'})}。",
            "",
            "[逐点CSV](curve_points.csv) · [完整统计](analysis_snapshot.json)",
        ]
    )
    if complete:
        lines.extend(
            [
                "",
                "## 一页结论表",
                "",
                "相关系数区间只描述样本重采样稳定性；正式层分配差异使用16项Holm校正。",
                "",
                "| 模型 | confirm192 QA趋势：先压缩 / 先保护 | 同压缩率NLL分配差异 | 同压缩率QA分配差异 |",
                "|---|---|---|---|",
            ]
        )
        for kind in MODELS:
            cohort = summary["models"][kind]["cohorts"]["confirm192"]
            trends = [cohort["correlations"][p + ":qa_macro"] for p in POLICIES]
            counts = {
                metric: sum(
                    e["model"] == kind and e["metric"] == metric and e["p_holm"] <= 0.05
                    for e in summary["tests"].values()
                )
                for metric in ("nll", "qa_macro")
            }
            trend_text = " / ".join(f"rho={e['rho']}, CI={e['ci95']}" for e in trends)
            effects = {}
            for metric in ("nll", "qa_macro"):
                entries = [e for e in summary["tests"].values() if e["model"] == kind and e["metric"] == metric]
                values = [e["advantage_of_protection"] for e in entries]
                extreme = max(entries, key=lambda e: abs(e["advantage_of_protection"]))
                unit = "百分点" if metric == "qa_macro" else "NLL"
                effects[metric] = (
                    f"{min(values):.4f}至{max(values):.4f} {unit}；{counts[metric]}/4点Holm显著；"
                    f"最大绝对差q={extreme['q']}，CI={extreme['ci95']}"
                )
            lines.append(f"| {kind} | {trend_text} | {effects['nll']} | {effects['qa_macro']} |")
        lines.extend(["", "两模型趋势方向的描述性比较（不替代模型间差异检验）：", ""])
        for split, metric in (("confirm192", "qa_macro"), ("confirm192", "nll"), ("screen32", "nll")):
            for policy in POLICIES:
                entries = [
                    summary["models"][kind]["cohorts"][split]["correlations"][policy + ":" + metric] for kind in MODELS
                ]
                directions = [0 if e["rho"] is None or e["rho"] == 0 else (1 if e["rho"] > 0 else -1) for e in entries]
                match = "方向一致" if directions[0] == directions[1] and directions[0] != 0 else "方向不一致或不可定义"
                stable = all(e["ci95"] is not None and e["ci95"][0] * e["ci95"][1] > 0 for e in entries)
                uncertainty = "两模型区间均不跨零" if stable else "至少一个模型区间跨零或不可定义"
                lines.append(f"- {split} / {metric} / {policy}：{match}；{uncertainty}。")
    (output / "report_zh.md").write_text("\n".join(lines) + "\n")
    return {"experiment_complete": complete, "report": str(output / "report_zh.md"), "figures": pictures}
