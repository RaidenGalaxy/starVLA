# Chinese report text intentionally uses native punctuation.
# ruff: noqa: RUF001
from __future__ import annotations

import csv
import json

from experiments.infinitevl_quant.progress import atomic_json, atomic_text, file_sha256

from .config import STAGES


def verified_summary(root):
    marker, summary, digest = root / ".complete", root / "summary.json", root / "summary.sha256.json"
    if not all(p.is_file() for p in (marker, summary, digest)):
        return None
    identity = json.loads((root / "identity.json").read_text())
    completion = json.loads(marker.read_text())
    if any(completion.get(k) != v for k, v in identity.items()):
        raise RuntimeError(f"invalid report completion identity: {root}")
    if completion.get("required") != ["summary.json", "summary.sha256.json"]:
        raise RuntimeError("unexpected completion artifact declarations")
    manifest = {str(p.relative_to(root)): file_sha256(p) for p in sorted(root.glob("*/samples/*.json"))}
    if json.loads(digest.read_text()) != {"summary.json": file_sha256(summary), "sample_rows": manifest}:
        raise RuntimeError(f"report refused a modified summary: {summary}")
    return json.loads(summary.read_text())


def ratio_artifacts(config, summary):
    report = config.output_dir / "report"
    rows = summary["aggregated"]
    path = report / "ratio_by_layer.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "trajectory",
                "layer",
                "group",
                "rho_sample_median",
                "eta_f_sample_median",
                "eta_r_sample_median",
                "eta_y_sample_median",
                "r_rms_pooled",
                "f_rms_pooled",
                "error_rms_pooled",
                "merged_rms_pooled",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["model_kind"],
                    row["trajectory"],
                    row["layer"],
                    row["group"],
                    *[row["sample_medians"][k] for k in ("rho", "eta_f", "eta_r", "eta_y")],
                    *[row["pooled_rms"][k] for k in ("residual", "ffn", "error", "merged")],
                ]
            )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"infinitevl": "#c23431", "qwen": "#2459a6"}
    for group in ("all_valid", "visual", "text_prompt", "answer_prediction"):
        figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
        for axis, metric in zip(axes.flat, ("rho", "eta_f", "eta_r", "eta_y"), strict=True):
            for kind, color in colors.items():
                for trajectory, style in (("reference", "-"), ("quantized", "--")):
                    selected = sorted(
                        [
                            r
                            for r in rows
                            if r["group"] == group and r["model_kind"] == kind and r["trajectory"] == trajectory
                        ],
                        key=lambda r: r["layer"],
                    )
                    axis.plot(
                        [r["layer"] for r in selected],
                        [
                            float("nan") if r["sample_medians"][metric] is None else r["sample_medians"][metric]
                            for r in selected
                        ],
                        color=color,
                        linestyle=style,
                        label=f"{kind} / {trajectory}",
                    )
            axis.set(xlabel="Layer index (zero based)", ylabel=metric, title=metric)
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
        figure.suptitle(f"Calibration32: sample-median ratios / {group}")
        name = "ratio_by_layer.svg" if group == "all_valid" else f"ratio_{group}.svg"
        figure.savefig(report / name, metadata={"Date": None})
        plt.close(figure)


def report(config):
    root = config.output_dir / "report"
    root.mkdir(parents=True, exist_ok=True)
    summaries = {stage: verified_summary(config.output_dir / stage) for stage in STAGES if stage not in ("report",)}
    status_path = config.output_dir / "status.json"
    status = json.loads(status_path.read_text()) if status_path.is_file() else {"status": "unknown"}
    lines = [
        "# InfiniteVL FFN 量化根因验证",
        "",
        "标签：`cpu_reference_backend`、`fake_quantized`、`historically_reused_evaluation`、`no_speed_or_memory_claim`。",
        "",
        "本报告仅使用完成标记、身份与摘要校验通过的阶段。未完成矩阵中的样本只用于进度展示，不参与机制结论。",
        "",
        f"当前运行状态：`{status.get('status')}`；阶段：`{status.get('stage', 'unknown')}`。",
        "",
        "| 阶段 | 状态 |",
        "|---|---|",
    ]
    for stage, summary in summaries.items():
        sample_files = list((config.output_dir / stage).glob("*/samples/*.json"))
        lines.append(
            f"| {stage} | {'已完成并校验' if summary is not None else f'未完成；已保存 {len(sample_files)} 个单元'} |"
        )
    ratio = summaries.get("ratio")
    if ratio:
        ratio_artifacts(config, ratio)
        lines.extend(
            [
                "",
                "## 比值核算",
                "",
                "四个指标分别为 rho=RMS(f)/RMS(r)、eta_f=RMS(delta)/RMS(f)、"
                "eta_r=RMS(delta)/RMS(r)、eta_y=RMS(delta)/RMS(r+f)。",
                "",
                "恒等式在单元内核验；能量池化 RMS 与样本比值中位数分别报告。零分母标为不可定义。"
                "旧 trace 仅代表最多 256 个均匀抽样 token。",
                "",
                "[逐层完整数值](ratio_by_layer.csv)",
                "",
                "![逐层比值](ratio_by_layer.svg)",
                "",
                "分组曲线：[视觉](ratio_visual.svg)、[文本提示](ratio_text_prompt.svg)、"
                "[答案预测位置](ratio_answer_prediction.svg)。实线为参考轨迹，虚线为量化轨迹。",
            ]
        )
        history = ratio["historical_fp64"]
        differences = [abs(r["rho_fp32_minus_fp64"]) for r in history if r["rho_fp32_minus_fp64"] is not None]
        lines.append(
            f"\n历史张量 FP64 重算：{len(history)} 个单元；"
            f"rho 的 FP32/FP64 最大绝对差为 {max(differences, default=0):.6g}。"
        )
        for kind, modes in ratio["transport_scales"]["gammas"].items():
            import statistics

            values = list(modes["relative"].values())
            lines.append(
                f"\n{kind} 向另一模型匹配相对误差的 gamma：逐层中位数 {statistics.median(values):.4f}，"
                f"范围 {min(values):.4f}–{max(values):.4f}。系数来自校准集，不含确认集。"
            )
    local = summaries.get("local")
    if local:
        lines.extend(["", "## W3 gate 与 W4 层定位", ""])
        for kind, result in local["models"].items():
            lines.append(f"{kind}：W4 恢复排序前三层为 {result['top3_layers']}（零起始）。")
            lines.extend(["", "| W3 条件 | token-weighted NLL |", "|---|---:|"])
            for condition, value in result["w3_nll"].items():
                lines.append(f"| {condition} | {value:.6f} |")
            lines.extend(
                ["", "| layer 1 位宽 | gate NMSE | SiLU NMSE | product NMSE | down NMSE |", "|---|---:|---:|---:|---:|"]
            )
            for row in result["decomposition_aggregates"]:
                if row["layer"] == 1:
                    errors = row["stage_nmse"]
                    lines.append(
                        f"| W{row['bits']} | {errors['gate']} | {errors['silu']} | "
                        f"{errors['product']} | {errors['down']} |"
                    )
    dose = summaries.get("dose")
    if dose:
        lines.extend(
            [
                "",
                "## 幅度与方向证据",
                "",
                "误差缩放只能检验幅度的解释力，不能单独识别 residual 尺度的因果作用。"
                "随机方向与真实误差逐 token 匹配 RMS；各条件的实际误差分布保存在 `__achieved` 样本产物中。",
                "",
                "| 模型 | 条件 | NLL |",
                "|---|---|---:|",
            ]
        )
        for kind, result in dose["models"].items():
            for condition, value in result["nll"].items():
                lines.append(f"| {kind} | {condition} | {value:.6f} |")
            if result.get("budgeted_omissions"):
                lines.append(
                    f"\n{kind} 为保护必需比较而整组延后的探索条件：{list(result['budgeted_omissions'])}。"
                    "这些条件不用于机制结论。\n"
                )
    propagation = summaries.get("propagation")
    if propagation:
        lines.extend(
            [
                "",
                "## GDN 传播路径",
                "",
                "状态修复不以局部增益比为门槛。recurrent/chunk 对齐与参考模型自修复均须通过；"
                "表中为 screen32 探索性结果，确认结果另列。",
                "",
                "| 修复条件 | 绝对 NLL 恢复 | 95% CI |",
                "|---|---:|---|",
            ]
        )
        for label, effect in propagation["models"]["infinitevl"].get("effects", {}).items():
            lines.append(
                f"| {label} | {effect['point']['recovery_absolute']:.6f} | {effect['ci95']['recovery_absolute']} |"
            )
    confirm = summaries.get("confirm")
    if confirm:
        lines.extend(
            [
                "",
                "## 192 条历史复用确认",
                "",
                "| 模型 | 干预 | 绝对 NLL 恢复 | 95% CI | 检验方向 | Holm p |",
                "|---|---|---:|---|---|---:|",
            ]
        )
        for kind, effects in confirm["effects"].items():
            for label, effect in effects.items():
                lines.append(
                    f"| {kind} | {label} | {effect['point']['recovery_absolute']:.6f} | "
                    f"{effect['ci95']['recovery_absolute']} | {effect['primary_direction']} | "
                    f"{effect['p_primary_holm']:.6g} |"
                )
        lines.extend(
            [
                "",
                "干预已在确认集推理前锁定。解释比例保留原始数值与置信区间，不截断到 0–100%。"
                "GDN oracle 修复仅报告 teacher-forcing 结果。",
                "",
                "## 机制结论边界",
                "",
                "确认的恢复效应不自动构成单一根因证明。幅度主导还要求双向匹配、剂量曲线及同能量方向对照闭合；"
                "结构主导要求目标修复优于匹配对照；传播参与要求状态修复及无效应对照同时通过。",
                "",
                "最终机制判定：需结合完整分解、双向与匹配对照审阅；当前报告不会根据单个显著恢复效应自动命名根因。",
            ]
        )
        evidence = confirm["mechanism_evidence"]
        lines.extend(["", "已确认的机制组成证据：", ""])
        for label, component in evidence["components"].items():
            lines.append(
                f"- `{label}`：`{component['interpretation']}`；解释比例 {component['explained_fraction']}，"
                f"95% CI {component['explained_fraction_ci95']}。"
            )
        minimum = evidence["minimum_among_confirmed_mixed_precision_candidates"]
        lines.append(f"\n已确认候选中保留参考精度层数最少的混合精度条件：{minimum}。它不是全局最优性证明。")
    else:
        lines.extend(
            [
                "",
                "## 尚未验证的结论",
                "",
                "确认阶段尚未完成，不能宣称幅度、gate 或 GDN 是最终根因，也不能宣称存在已验证的最小恢复方案。",
            ]
        )
    atomic_text(root / "report_zh.md", "\n".join(lines) + "\n")
    atomic_json(
        root / "analyses.json",
        {
            "mechanism_evidence": confirm["mechanism_evidence"] if confirm else None,
            "confirmation_complete": confirm is not None,
            "history": "historically_reused_evaluation",
        },
    )
    atomic_json(
        root / "evidence_index.json",
        {
            "stages": {
                stage: {
                    "complete": value is not None,
                    "summary_sha256": file_sha256(config.output_dir / stage / "summary.json")
                    if value is not None
                    else None,
                }
                for stage, value in summaries.items()
            },
            "status": status,
        },
    )
    return {
        "completed_stages": [s for s, v in summaries.items() if v is not None],
        "experiment_complete": all(summaries.values()),
        "report": str(root / "report_zh.md"),
    }
