from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from experiments.infinitevl_quant.progress import atomic_json, atomic_text

from .config import MechanismConfig
from .protocol import load_audit, stage_progress


def _read(path: Path, *, required: bool = True) -> dict[str, Any] | None:
    if not path.is_file():
        if required:
            raise FileNotFoundError(path)
        return None
    return json.loads(path.read_text())


def _percent(value: float | None) -> str:
    return "—" if value is None or not math.isfinite(float(value)) else f"{100.0 * float(value):.2f}%"


def _number(value: float | None, digits: int = 4) -> str:
    return "—" if value is None or not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def run_report(config: MechanismConfig) -> dict[str, Any]:
    audit = load_audit(config)
    calibration = _read(config.output_dir / "calibrate/error_scales.json")
    localization = _read(config.output_dir / "localize/layer_pair_summary.json")
    transport = _read(config.output_dir / "transport/summary.json")
    mediation = _read(config.output_dir / "gdn_mediation/state_and_shapley_summary.json")
    confirm = _read(config.output_dir / "confirm/causal_effects.json")
    assert calibration and localization and transport and mediation and confirm
    decisions = confirm["confirm_decisions"]
    causal = confirm.get("causal_transport")
    factorial = confirm.get("mediation_factorial")
    causal_point = {} if causal is None else causal["point"]
    factorial_point = {} if factorial is None else factorial["point"]
    lines = [
        "# InfiniteVL–Qwen FFN 量化差异强因果根因报告",
        "",
        "标签：`cpu_reference_backend`、`fake_quantized`、`ffn_only`、"
        "`diagnostic_counterfactual`、`historically_reused_evaluation`、"
        "`no_speed_or_memory_claim`。",
        "",
        "> GDN state/mediator repair 和误差运输均为诊断性反事实，不是部署量化方法。",
        "",
        "## 最终判定",
        "",
        f"- H_R（相对误差暴露）：`{'PASS' if decisions['H_R'] else 'NOT_CONFIRMED'}`",
        f"- H_G（GDN 递归传播）：`{'PASS' if decisions['H_G'] else 'NOT_CONFIRMED'}`",
        f"- H_RG（联合根因）：`{'PASS' if decisions['H_RG_joint'] else 'NOT_CONFIRMED'}`",
        f"- 交互类型：`{decisions['interaction']}`",
        "",
        "## 主效应",
        "",
        "| 指标 | 点估计 | 95% CI |",
        "|---|---:|---:|",
    ]
    if causal is not None:
        for key, label in (
            ("infinitevl_residual_transport_effect", "InfiniteVL 相对误差运输效应"),
            ("qwen_reverse_transport_effect", "Qwen 反向压力效应"),
            ("mean_bidirectional_did_closure", "双向 DiD 平均闭合率"),
        ):
            ci = causal["ci95"].get(key)
            lines.append(
                f"| {label} | {_percent(causal_point.get(key))} | "
                f"{('—' if ci is None else f'[{_percent(ci[0])}, {_percent(ci[1])}]')} |"
            )
    if factorial is not None:
        for key, label in (
            ("residual_explained_fraction", "残差相对误差解释比例"),
            ("gdn_explained_fraction", "GDN carry 解释比例"),
            ("joint_explained_fraction", "联合解释比例"),
            ("factorial_interaction", "R×G 交互"),
        ):
            ci = factorial["ci95"].get(key)
            lines.append(
                f"| {label} | {_percent(factorial_point.get(key))} | "
                f"{('—' if ci is None else f'[{_percent(ci[0])}, {_percent(ci[1])}]')} |"
            )
    gain = localization["primary_screen_exact_dose1"]
    lines.extend(
        [
            f"| GDN/SWA block gain 比 | {float(gain['point']):.4f} | "
            f"[{float(gain['ci95'][0]):.4f}, {float(gain['ci95'][1]):.4f}] |",
            "",
            "## GDN 内部归因",
            "",
            f"- diagnostic8 预注册 Shapley 选择：`{', '.join(confirm['selected_mediator_groups']) or 'none'}`。",
            f"- confirm192 中效应区间为正的中介组：`{', '.join(confirm['confirmed_mediator_groups']) or 'none'}`。",
            "- 未进入 confirm 或未通过 confirm 的中介组保持“未解析”，不从 diagnostic8 外推。",
            "",
            "## 序列长度异质性",
            "",
        ]
    )
    length = confirm.get("length_stratified_effects")
    if length is None:
        lines.append("- GDN 未通过 screen，因此未扩展长度异质性确认。")
    else:
        carry = length["carry_clamp"]
        lines.extend(
            [
                "| 长度桶 | 配对样本 | carry 解释比例 | 95% CI |",
                "|---|---:|---:|---:|",
            ]
        )
        for key, label in (("le_256", "≤256"), ("257_512", "257–512"), ("gt_512", ">512")):
            row = carry["bins"][key]
            if row["status"] == "not_estimable":
                lines.append(f"| {label} | 0 | 不可估计 | — |")
            else:
                ci = row["ci95"]["explained_fraction"]
                lines.append(
                    f"| {label} | {row['paired_samples']} | "
                    f"{_percent(row['point']['explained_fraction'])} | "
                    f"[{_percent(ci[0])}, {_percent(ci[1])}] |"
                )
        trend = carry["trend"]
        lines.append(
            f"- 样本级长度趋势斜率（每 log2 token）：{_number(trend['slope_per_log2_token'])}，"
            f"95% CI [{_number(trend['ci95'][0])}, {_number(trend['ci95'][1])}]。"
        )
    lines.extend(
        [
            "",
            "## confirm192 QA 与 logits 次指标",
            "",
            "| 模型/条件 | QA 宏平均 | target NLL | answer-token KL | argmax 改变率 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for kind, conditions in confirm["qa_core_conditions"].items():
        for condition, row in conditions.items():
            lines.append(
                f"| {kind}/{condition} | {_number(row.get('mean_score'), 2)} | "
                f"{_number(row.get('target_nll'))} | {_number(row.get('mean_logits_kl'), 6)} | "
                f"{_percent(row.get('target_argmax_change_rate'))} |"
            )
    mbq = confirm.get("mbq_directional_replication")
    mbq_line = (
        "- 未执行：至少一个 screen 因素未通过，按预注册规则不扩大。"
        if mbq is None
        else "- qig_acceptance32 双向结果："
        f"`{'PASS' if mbq['directional_checks']['both_directions_replicate'] else 'NOT_REPLICATED'}`。"
    )
    lines.extend(
        [
            "",
            "## MBQ 方向复核",
            "",
            mbq_line,
            "",
            "## 协议边界",
            "",
            "- 主识别范围是 RTN W4A16、非对称 g128、CPU FP32 fake quant。",
            "- MBQ 结果仅为 qig_acceptance32 方向复核，不能替代 confirm192。",
            "- confirm192 在本协议内部与 calibration/screen 零重叠，但在历史实验中已使用，不是全新外部测试集。",
            "- 本报告不提供速度、内存、packed kernel 或部署收益结论。",
        ]
    )
    progress = stage_progress(config, "report")
    analyses = {
        "format_version": 1,
        "audit_digest": audit["config_digest"],
        "calibration": calibration,
        "localization": localization,
        "transport": transport,
        "mediation": mediation,
        "confirm": confirm,
    }
    summary = {
        "format_version": 1,
        "decisions": decisions,
        "selected_mediator_groups": confirm["selected_mediator_groups"],
        "claims": config.to_dict()["claims"],
    }
    atomic_json(progress.root / "analyses.json", analyses)
    atomic_json(progress.root / "report_summary.json", summary)
    atomic_text(progress.root / "report_zh.md", "\n".join(lines) + "\n")
    progress.mark_complete(("analyses.json", "report_summary.json", "report_zh.md"))
    return summary
