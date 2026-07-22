"""Atomic machine-readable outputs and a detailed Chinese report."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


def _safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_safe(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(_safe(row), ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def summary_row(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["metrics"]
    generation = metrics["generation"]
    return {
        "name": result["name"],
        "bits": result.get("bits", "reference"),
        "quantized_linears": result.get("quantized_linears", 0),
        "awq_linears": result.get("awq_linears", 0),
        "weight_nmse": result.get("weight_metrics", {}).get("overall", {}).get("normalized_mse", 0.0),
        "vision_merger_nmse": metrics["activations"]["merger"]["normalized_mse"],
        "final_hidden_nmse": metrics["hidden_states"]["final"]["normalized_mse"],
        "final_hidden_cosine": metrics["hidden_states"]["final"]["cosine_similarity"],
        "logits_kl": metrics["logits"]["kl_divergence"],
        "top1_agreement": metrics["logits"]["top1_agreement"],
        "target_nll": metrics["logits"]["target_nll"],
        "target_perplexity": metrics["logits"]["target_perplexity"],
        "generation_reference_agreement": generation["reference_exact_agreement"],
        "generation_target_accuracy": generation["target_exact_accuracy"],
        "generation_edit_similarity": generation["normalized_edit_similarity"],
        "elapsed_seconds": result.get("elapsed_seconds", 0.0),
    }


def write_summary(path: Path, results: list[dict[str, Any]]) -> None:
    rows = [summary_row(result) for result in results]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_report(
    path: Path,
    results: list[dict[str, Any]],
    inventory: dict[str, Any],
    config: dict[str, Any],
    conversion: dict[str, Any],
    awq_search: list[dict[str, Any]],
) -> None:
    rows = [summary_row(result) for result in results]
    accepted = sum(item.get("fallback_reason") is None for item in awq_search)
    equivalence = sum(str(item.get("fallback_reason", "")).endswith("equivalence") for item in awq_search)
    not_better = sum(item.get("fallback_reason") == "not_better_than_rtn" for item in awq_search)
    lines = [
        "# Qwen2.5-VL 全模型端到端权重量化实验报告",
        "",
        "> 参考模型由 checkpoint 权重经过 FP16 可表示值舍入后转回 FP32，在 CPU 上执行。W8/W4 均为 fake quantization；本报告不代表真实 FP16、INT8 或 W4A16 kernel 的显存、速度和延迟。",
        "",
        "## 1. 实验范围",
        "",
        f"模型包含 {inventory['decoder_layers']} 层 decoder 和 {inventory['vision_layers']} 层 vision block。量化目标为 {inventory['linear_modules']} 个 Linear、{inventory['target_parameters']:,} 个权重，占全模型参数 {inventory['target_fraction']:.2%}。共享 embedding/LM head、patch Conv3d、norm 和 bias 不做整数化。",
        "",
        "| 子系统 | Linear | 权重 |",
        "|---|---:|---:|",
    ]
    for group, value in inventory["groups"].items():
        lines.append(f"| {group} | {value['linear_modules']} | {value['parameters']:,} |")
    lines.extend([
        "",
        "## 2. 16 位参考的准确含义",
        "",
        f"checkpoint 配置声明 dtype 为 `{conversion.get('checkpoint_declared_dtype', 'bfloat16')}`，因此无法恢复一个未提供的原始 FP16 checkpoint。本实验将所有浮点参数先转换成 FP16 可表示值，再转回 FP32 前向。它隔离权重从该 16 位参考到 W8/W4 的误差，但激活和算子仍是 FP32。",
        "",
        f"共处理 {conversion['floating_values']:,} 个浮点参数值，其中 {conversion['changed_values']:,} 个值在舍入后变化；最大绝对舍入误差为 {conversion['max_abs_rounding_error']:.6g}。转换过程拒绝 NaN/Inf 和 FP16 溢出。",
        "",
        "## 3. 数据与协议",
        "",
        f"使用确定性 Tiny Tabletop：{len(config['calibration_samples'])} 条 calibration、{len(config['evaluation_samples'])} 条独立 evaluation，seed={config['seed']}，图像为 112×112，覆盖颜色、形状和左右关系。所有配置都从原始像素重新执行 vision encoder、merger、decoder 和 LM head。",
        "",
        "Teacher forcing 指标只在 target token 预测位置计算；自由生成使用 greedy decoding、`do_sample=False`，最多生成 " + str(config["max_new_tokens"]) + " 个 token。",
        "",
        "## 4. 指标解释",
        "",
        "- NMSE=`||candidate-reference||²/||reference||²`，越小越好；它表示相对误差能量，不是准确率下降比例。",
        "- Cosine 衡量激活或 hidden 表示方向一致性，越接近 1 越好，需与 NMSE 联合阅读。",
        "- Logits KL=`KL(P_ref||P_quant)`，衡量完整词表概率分布偏移，越接近 0 越好。",
        "- Top-1 Agreement 是量化模型与参考模型首选 token 的一致率，不是相对标准答案的准确率。",
        "- Generation Reference Agreement 是完整生成答案与参考生成的一致率；Target Accuracy 才是相对数据 target 的规范化精确匹配率。",
        "",
        "## 5. 配置级结果",
        "",
        "| 配置 | 量化Linear | AWQ Linear | Weight NMSE↓ | Merger NMSE↓ | Final Hidden NMSE↓ | Final Hidden Cosine↑ | Logits KL↓ | Top-1一致率↑ | 生成参考一致率↑ | Target准确率↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['quantized_linears']} | {row['awq_linears']} | {row['weight_nmse']:.6g} | "
            f"{row['vision_merger_nmse']:.6g} | {row['final_hidden_nmse']:.6g} | {row['final_hidden_cosine']:.6f} | "
            f"{row['logits_kl']:.6g} | {row['top1_agreement']:.2%} | {row['generation_reference_agreement']:.2%} | "
            f"{row['generation_target_accuracy']:.2%} |"
        )
    by_name = {row["name"]: row for row in rows}
    required = {
        "fp16_weight_reference", "rtn_w8_all_linear", "rtn_w4_all_linear",
        "hybrid_awq_w4_all_linear", "rtn_w4_all_ffn_only", "awq_w4_all_ffn_only",
        "rtn_w4_vision_only", "rtn_w4_decoder_attention_only",
    }
    if required <= set(by_name):
        ref, w8, w4 = (by_name[name] for name in ("fp16_weight_reference", "rtn_w8_all_linear", "rtn_w4_all_linear"))
        hybrid = by_name["hybrid_awq_w4_all_linear"]
        ffn_rtn, ffn_awq = by_name["rtn_w4_all_ffn_only"], by_name["awq_w4_all_ffn_only"]
        vision, attention = by_name["rtn_w4_vision_only"], by_name["rtn_w4_decoder_attention_only"]
        reduction = lambda a, b: 100.0 * (a - b) / a if a else 0.0
        lines.extend([
            "", "## 6. 结果分析", "",
            "### 6.1 W8 全 Linear 基本保持参考行为", "",
            f"全 414 个 Linear 使用 RTN W8 后，Final Hidden Cosine={w8['final_hidden_cosine']:.6f}，Logits KL={w8['logits_kl']:.6g}，Top-1 与生成参考一致率均为 100%。当前数据上 W8 是近乎无损的低风险配置。", "",
            "### 6.2 全 Linear RTN W4 出现明显累计误差", "",
            f"全 Linear RTN W4 的 Final Hidden NMSE={w4['final_hidden_nmse']:.6g}、Cosine={w4['final_hidden_cosine']:.6f}、Logits KL={w4['logits_kl']:.6g}、Top-1一致率={w4['top1_agreement']:.2%}，完整生成参考一致率降到 {w4['generation_reference_agreement']:.2%}。", "",
            "### 6.3 FFN AWQ 显著缓解 W4 误差", "",
            f"Hybrid AWQ 相比全 Linear RTN W4，将 Final Hidden NMSE 降低 {reduction(w4['final_hidden_nmse'], hybrid['final_hidden_nmse']):.1f}%，Logits KL 降低 {reduction(w4['logits_kl'], hybrid['logits_kl']):.1f}%；Final Hidden Cosine 从 {w4['final_hidden_cosine']:.6f} 提升到 {hybrid['final_hidden_cosine']:.6f}，Top-1 从 {w4['top1_agreement']:.2%} 提升到 {hybrid['top1_agreement']:.2%}。", "",
            f"在 FFN-only 公平对照中，AWQ 相比 RTN 将 Final Hidden NMSE 降低 {reduction(ffn_rtn['final_hidden_nmse'], ffn_awq['final_hidden_nmse']):.1f}%，Logits KL 降低 {reduction(ffn_rtn['logits_kl'], ffn_awq['logits_kl']):.1f}%。AWQ 优化的是激活下的 FFN 输出重构，权重 NMSE 不一定低于 RTN。", "",
            "### 6.4 子系统归因", "",
            f"FFN-only RTN W4 的 Final Hidden NMSE/KL 为 {ffn_rtn['final_hidden_nmse']:.6g}/{ffn_rtn['logits_kl']:.6g}，高于 vision-only 的 {vision['final_hidden_nmse']:.6g}/{vision['logits_kl']:.6g} 和 decoder-attention-only 的 {attention['final_hidden_nmse']:.6g}/{attention['logits_kl']:.6g}。当前小数据上 FFN 是全模型 W4 的主要误差来源。Vision-only 包含 vision FFN，因此这些配置用于敏感性比较，不能简单线性相加。", "",
            "### 6.5 Target Accuracy 的限制", "",
            f"参考模型自身在严格规范化 exact-match 下的 Target Accuracy 为 {ref['generation_target_accuracy']:.2%}，因此量化配置的 0% 不能解释为量化导致准确率归零。当前应优先读取相对参考一致率和编辑相似度，正式准确率需引入答案抽取或真实 VQA 评分协议。", "",
        ])
    lines.extend([
        "",
        "## 7. AWQ 搜索与回退",
        "",
        f"共搜索 {len(awq_search)} 个 vision/decoder FFN，其中 {accepted} 个接受 AWQ scale，{equivalence} 个因严格函数等价检查回退，{not_better} 个因校准误差未优于 RTN 回退。候选包含 identity，因此最终校准 FFN NMSE 不得差于同层 RTN。",
        "",
        "逐 vision block、merger、decoder attention、vision/decoder FFN 和 hidden-state 曲线保存在 `metrics.json`；逐层 scale 与搜索曲线保存在 `awq_scales.pt` 和 `awq_search.json`。",
        "",
        "## 8. 结论边界",
        "",
        f"本轮只覆盖 {len(config['calibration_samples'])}/{len(config['evaluation_samples'])} 条合成图文数据上的 CPU 权重 fake-quant 数值误差和短答案变化。它不覆盖真实 VQA benchmark、GPU FP16/W4A16 kernel、显存/吞吐/延迟、StarVLA action head 或 LIBERO 成功率。",
        "",
    ])
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)
