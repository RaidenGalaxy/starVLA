from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.infinitevl_quant.progress import atomic_json, atomic_text
from experiments.infinitevl_quant_ffn_bf16_4x64.scoring import SCORER_ID

from .config import STAGES


REPORT_LABELS = (
    "cpu_reference_backend",
    "fake_quantized",
    "ffn_only",
    "matched",
    "within_protocol_holdout192",
    "historically_reused_evaluation",
    "no_speed_or_memory_claim",
)


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{numeric:.{digits}f}" if math.isfinite(numeric) else "—"


def _fmt_percent(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{100.0 * numeric:.{digits}f}%" if math.isfinite(numeric) else "—"


def _fmt_interval(
    interval: Sequence[float] | None, *, digits: int = 3, percent: bool = False,
) -> str:
    if interval is None or len(interval) != 2:
        return "—"
    formatter = _fmt_percent if percent else _fmt
    return f"[{formatter(interval[0], digits)}, {formatter(interval[1], digits)}]"


def _validate_analysis(analyses: Mapping[str, Any]) -> None:
    allowed = {"paired", "difference_in_differences", "factorial", "not_run"}
    unknown = set(analyses) - allowed
    if unknown:
        raise ValueError(f"unknown report analysis sections: {sorted(unknown)}")
    for section in ("paired", "difference_in_differences", "factorial"):
        payload = analyses.get(section, {})
        if not isinstance(payload, Mapping):
            raise ValueError(f"analysis section {section} must be a mapping")
        for name, result in payload.items():
            if not isinstance(name, str) or not isinstance(result, Mapping):
                raise ValueError(f"invalid entry in analysis section {section}")
            if result.get("scorer") != SCORER_ID:
                raise ValueError(
                    f"{section}/{name} is not explicitly tagged with {SCORER_ID}; "
                    "legacy scores are rejected rather than mixed"
                )
    not_run = analyses.get("not_run", [])
    if isinstance(not_run, (str, bytes)) or not isinstance(not_run, Sequence):
        raise ValueError("not_run must be a sequence of descriptions")


def _stage_status(value: Any) -> tuple[str, str]:
    if isinstance(value, str):
        return value, ""
    if isinstance(value, Mapping):
        status = str(value.get("status", "not_reported"))
        detail = str(value.get("detail", ""))
        return status, detail
    return "not_reported", ""


def _validate_evidence(evidence: Mapping[str, Any]) -> None:
    unknown = set(evidence) - set(STAGES)
    if unknown:
        raise ValueError(f"unknown evidence stages: {sorted(unknown)}")
    for stage, record in evidence.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"evidence/{stage} must be a mapping")
        artifacts = record.get("artifacts", {})
        if not isinstance(artifacts, Mapping):
            raise ValueError(f"evidence/{stage}/artifacts must be a mapping")
        for name, artifact in artifacts.items():
            if not isinstance(name, str) or not isinstance(artifact, Mapping):
                raise ValueError(f"invalid evidence artifact under {stage}")
            for field in ("path", "relative_path", "sha256", "bytes", "summary"):
                if field not in artifact:
                    raise ValueError(f"evidence/{stage}/{name} misses {field}")


def _evidence_summary(value: Any, *, limit: int = 360) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _evidence_lines(evidence: Mapping[str, Any]) -> list[str]:
    lines = [
        "| 阶段 | 证据产物 | SHA256 | 摘要 |",
        "|---|---|---|---|",
    ]
    emitted = False
    for stage in STAGES:
        record = evidence.get(stage, {})
        if not isinstance(record, Mapping):
            continue
        artifacts = record.get("artifacts", {})
        if not isinstance(artifacts, Mapping):
            continue
        if not artifacts and stage != "report":
            emitted = True
            status = str(record.get("status", "not_run"))
            lines.append(f"| {stage} | — | — | `{status}`：无可索引证据产物 |")
            continue
        for name, artifact in sorted(artifacts.items()):
            emitted = True
            digest = str(artifact.get("sha256", ""))
            short_digest = digest[:16] + "…" if len(digest) > 16 else digest
            path = str(artifact.get("relative_path", artifact.get("path", "—")))
            summary = _evidence_summary(artifact.get("summary", {})).replace("|", "\\|")
            lines.append(
                f"| {stage} | `{name}` (`{path}`) | `{short_digest or '—'}` | `{summary}` |"
            )
    if not emitted:
        lines.append("| — | — | — | 尚无通过阶段身份校验的证据产物 |")
    return lines


def _zero_note(interval: Sequence[float] | None) -> str:
    if interval is None or len(interval) != 2:
        return "QA 未执行"
    low, high = (float(interval[0]), float(interval[1]))
    return "区间包含 0" if low <= 0 <= high else "区间不含 0"


def _paired_table(entries: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = [
        "| 比较 | 配对样本 | QA 宏平均差（分） | QA 95% CI | token-weighted NLL 相对变化 | NLL 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in sorted(entries.items()):
        point = result.get("point", {})
        lines.append(
            f"| {name} | {result.get('paired_samples', '—')} | "
            f"{_fmt(point.get('qa_macro_delta_points'))} | "
            f"{_fmt_interval(result.get('qa_macro_delta_points_ci95'))} | "
            f"{_fmt_percent(point.get('target_nll_relative_change'))} | "
            f"{_fmt_interval(result.get('target_nll_relative_change_ci95'), percent=True)} |"
        )
    return lines


def _did_table(entries: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = [
        "| 比较 | 配对样本 | QA DiD（InfiniteVL−Qwen，分） | QA 95% CI | NLL相对变化 DiD | NLL 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in sorted(entries.items()):
        point = result.get("point", {})
        lines.append(
            f"| {name} | {result.get('paired_samples', '—')} | "
            f"{_fmt(point.get('qa_macro_did_points'))} | "
            f"{_fmt_interval(result.get('qa_macro_did_points_ci95'))} | "
            f"{_fmt_percent(point.get('target_nll_relative_change_did'))} | "
            f"{_fmt_interval(result.get('target_nll_relative_change_did_ci95'), percent=True)} |"
        )
    return lines


def _factorial_table(entries: Mapping[str, Mapping[str, Any]]) -> list[str]:
    lines = [
        "| 模型/比较 | 配对样本 | QA 加性交互（分） | QA 95% CI | NLL 乘性交互 | NLL 95% CI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in sorted(entries.items()):
        point = result.get("point", {})
        lines.append(
            f"| {name} | {result.get('paired_samples', '—')} | "
            f"{_fmt(point.get('qa_macro_additive_interaction_points'))} | "
            f"{_fmt_interval(result.get('qa_macro_additive_interaction_points_ci95'))} | "
            f"{_fmt_percent(point.get('target_nll_multiplicative_interaction'))} | "
            f"{_fmt_interval(result.get('target_nll_multiplicative_interaction_ci95'), percent=True)} |"
        )
    return lines


def _uncertainty_lines(analyses: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for section, label, interval_key in (
        ("paired", "配对比较", "qa_macro_delta_points_ci95"),
        ("difference_in_differences", "跨模型 DiD", "qa_macro_did_points_ci95"),
        ("factorial", "W4 2×2 交互", "qa_macro_additive_interaction_points_ci95"),
    ):
        for name, result in sorted(analyses.get(section, {}).items()):
            lines.append(
                f"- {label} `{name}`：{result.get('bootstrap_samples', '—')} 次，"
                f"seed={result.get('seed', '—')}；{_zero_note(result.get(interval_key))}。"
            )
    if not lines:
        lines.append("- 尚无 bootstrap 结果，不能判断抽样不确定性。")
    return lines


def render_report_zh(
    analyses: Mapping[str, Any],
    *,
    stage_status: Mapping[str, Any] | None = None,
    evidence: Mapping[str, Any] | None = None,
    title: str = "InfiniteVL 与 Qwen FFN 量化因果对照",
) -> str:
    """Render an evidence-bounded report without filling missing experiments.

    ``analyses`` may contain ``paired``, ``difference_in_differences``, and
    ``factorial`` mappings whose values are produced by :mod:`statistics`.
    Missing sections and non-completed stages are explicitly reported instead
    of being silently treated as completed.
    """
    _validate_analysis(analyses)
    supplied_evidence = dict(evidence or {})
    _validate_evidence(supplied_evidence)
    supplied_status = dict(stage_status or {})
    status_rows = {
        stage: _stage_status(supplied_status.get(stage, "not_reported"))
        for stage in STAGES
    }
    label_text = "、".join(f"`{label}`" for label in REPORT_LABELS)
    lines = [
        f"# {title}",
        "",
        f"结果标签：{label_text}。",
        "",
        "> 本报告只陈述已落盘的配对结果。低比特结果是在 FP32 CPU 上执行的 QDQ/fake-quant "
        "质量模拟，不代表 packed kernel 的内存、延迟或吞吐收益。",
        "",
        "## 协议边界",
        "",
        f"- QA 统计只接受 `{SCORER_ID}`；旧 legacy scorer 行会被拒绝，绝不混算。",
        "- QA 主统计是逐数据集均值再宏平均；NLL 主统计按 target token 总数加权。",
        "- bootstrap 在每个数据集内有放回抽取样本 ID，同一次抽样用于全部比较单元。",
        "- `ffn_only` 标识 matched 主比较范围；W4 2×2 诊断另含 `attn_main`，不应把该诊断误读为纯 FFN 结果。",
        "- `within_protocol_holdout192` 只表示它与本协议的 calibration、acceptance、screen 在 ID/图像/问题哈希上不重叠。",
        "- 该 192 条来自此前已运行过的评测清单，必须标记 `historically_reused_evaluation`；它不是历史上未见的新测试集，不能表述为全新外部验证。",
        "",
        "## 已观察结果",
        "",
    ]

    paired = analyses.get("paired", {})
    did = analyses.get("difference_in_differences", {})
    factorial = analyses.get("factorial", {})
    if paired:
        lines.extend(["### 配对 BF16 对照", ""])
        lines.extend(_paired_table(paired))
        lines.append("")
    if did:
        lines.extend(["### InfiniteVL−Qwen difference-in-differences", ""])
        lines.extend(_did_table(did))
        lines.extend([
            "",
            "正的 NLL DiD 表示 InfiniteVL 的相对 NLL 增幅高于 Qwen；这是配对差异描述，"
            "本身不证明某个内部机制是原因。",
            "",
        ])
    if factorial:
        lines.extend(["### W4 FFN × attn_main 2×2 交互", ""])
        lines.extend(_factorial_table(factorial))
        lines.extend([
            "",
            "交互定义：QA 为 `decoder_safe − FFN − attn_main + BF16`；NLL 乘性交互为 "
            "`decoder_safe × BF16 / (FFN × attn_main) − 1`。",
            "",
        ])
    if not (paired or did or factorial):
        lines.extend(["尚无可核验的统计结果。", ""])

    lines.extend(["## 统计不确定性", ""])
    lines.extend(_uncertainty_lines(analyses))
    lines.extend([
        "",
        "95% percentile interval 包含 0 表示当前样本不能排除零差异；不包含 0 也不等同于"
        "证明模型结构因果，仍需 trace、反事实与协议内留出复核共同支持。",
        "",
        "## 证据索引",
        "",
        "下表只索引身份标记、`identity.json` 和必需产物均验证通过的阶段；SHA256 指向实际落盘 JSON。",
        "",
    ])
    lines.extend(_evidence_lines(supplied_evidence))
    lines.extend([
        "",
        "## 尚未执行或尚未提供",
        "",
        "| 阶段 | 状态 | 说明 |",
        "|---|---|---|",
    ])
    for stage in STAGES:
        status, detail = status_rows[stage]
        if status != "completed":
            lines.append(f"| {stage} | {status} | {detail or '未提供完成证据'} |")
    experimental_stages = STAGES[:-1]
    experiment_complete = all(
        status_rows[stage][0] == "completed" for stage in experimental_stages
    )
    if all(status == "completed" for status, _ in status_rows.values()):
        lines.append("| — | — | 所有阶段的身份和必需产物均已验证 |")
    for item in analyses.get("not_run", []):
        lines.append(f"| 分析项 | not_run | {item} |")

    missing_sections = [
        name for name, value in (
            ("paired bootstrap", paired),
            ("cross-model DiD", did),
            ("W4 2×2 factorial", factorial),
        ) if not value
    ]
    if missing_sections:
        lines.extend([
            "",
            "未提供的统计部分：" + "、".join(missing_sections) + "。本报告不会为其补写结果。",
        ])
    lines.extend([
        "",
        "## 可解释范围",
        "",
        "- 已观察的分数/NLL 差异只适用于列出的样本、位宽、g128 fake-quant 与 CPU 参考计算路径。",
        "- W4 factorial 可识别 FFN 与 attention 量化在该设计下的统计交互；不能单独定位到某一层或某一异常值机制。",
        "- 跨模型 DiD 抵消了部分共同任务难度，但不能自动消除 tokenizer、激活分布或状态传播路径差异。",
        "- `report` 阶段完成只表示报告文件已生成；只有 audit、RTN、trace、counterfactual、MBQ/QIG 和协议内留出复核都验证通过，才表示实验链完整。",
        "- 在 `within_protocol_holdout192` 未完成前，不给出最终 GO；在 MBQ/QIG 未完成前，不把 RTN 结果表述为方法上限。",
        "- 即使协议内留出复核完成，也因 `historically_reused_evaluation` 只能给出探索性复核结论；正式外部确认需要新的未使用评测清单。",
        f"- 当前实验链完整性：{'完整' if experiment_complete else '不完整'}。",
        "",
    ])
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    analyses: Mapping[str, Any],
    *,
    stage_status: Mapping[str, Any] | None = None,
    evidence: Mapping[str, Any] | None = None,
    title: str = "InfiniteVL 与 Qwen FFN 量化因果对照",
) -> dict[str, Any]:
    """Write deterministic machine-readable and Chinese report artifacts."""
    report_dir = Path(output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    status = dict(stage_status or {})
    indexed_evidence = dict(evidence or {})
    _validate_evidence(indexed_evidence)
    experimental_stages = STAGES[:-1]
    completed = [
        stage for stage in experimental_stages
        if _stage_status(status.get(stage))[0] == "completed"
    ]
    summary = {
        "format_version": 1,
        "scorer": SCORER_ID,
        "labels": list(REPORT_LABELS),
        "stage_status": status,
        "protocol_completion": {
            "all_experimental_stages_validated": len(completed) == len(experimental_stages),
            "validated_experimental_stages": completed,
            "required_experimental_stages": list(experimental_stages),
            "report_completion_does_not_imply_protocol_completion": True,
        },
        "holdout_semantics": {
            "report_label": "within_protocol_holdout192",
            "evaluation_history": "historically_reused_evaluation",
            "historically_unseen": False,
        },
        "evidence": indexed_evidence,
        "analyses": dict(analyses),
    }
    atomic_json(report_dir / "report_summary.json", summary)
    atomic_text(
        report_dir / "report_zh.md",
        render_report_zh(
            analyses,
            stage_status=status,
            evidence=indexed_evidence,
            title=title,
        ),
    )
    return summary
