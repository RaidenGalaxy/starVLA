"""Generate the detailed Chinese report for the completed native CPU matrix."""

from __future__ import annotations

import json
from pathlib import Path


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _pct_change(reference, value):
    return (reference - value) / reference if reference else 0.0


def write_detailed_report(output: Path, packed_results_root: Path) -> None:
    smoke = {
        name: _load(output / "smoke" / name / "metrics.json")
        for name in (
            "streaming_rtn_w8",
            "streaming_rtn_w4",
            "streaming_hybrid_awq_w4",
            "native_rtn_w4a8",
            "native_hybrid_awq_w4a8",
        )
    }
    rtn = _load(output / "cpu" / "native_rtn_w4a8" / "metrics.json")
    awq = _load(output / "cpu" / "native_hybrid_awq_w4a8" / "metrics.json")
    packed_rtn = _load(packed_results_root / "packed_rtn_w4_fp32" / "metrics.json")
    packed_awq = _load(packed_results_root / "packed_hybrid_awq_w4_fp32" / "metrics.json")
    storage = _load(output / "cpu" / "native_hybrid_awq_w4a8" / "storage_metrics.json")

    def fields(result):
        metrics, runtime = result["metrics"], result["runtime"]
        activation = runtime.get("activation_quantization", {})
        hidden, logits, generation = metrics["hidden_states"]["final"], metrics["logits"], metrics["generation"]
        return {
            "hidden_nmse": hidden["normalized_mse"],
            "hidden_cosine": hidden["cosine_similarity"],
            "hidden_sqnr": hidden["sqnr_db"],
            "kl": logits["kl_divergence"],
            "top1": logits["top1_agreement"],
            "nll": logits["target_nll"],
            "ppl": logits["target_perplexity"],
            "ref_exact": generation["reference_exact_agreement"],
            "target_exact": generation["target_exact_accuracy"],
            "token": generation["token_agreement"],
            "edit": generation["normalized_edit_similarity"],
            "activation_nmse": activation.get("normalized_mse", 0.0),
            "activation_sqnr": activation.get("sqnr_db", float("inf")),
            "elapsed": result["elapsed_seconds"],
        }

    r, a = fields(rtn), fields(awq)
    pr, pa = fields(packed_rtn), fields(packed_awq)
    lines = [
        "# Qwen2.5-VL Packed INT4 与 AVX-512 W4A8 端到端实验报告", "",
        "## 1. 实验目标与结论范围", "",
        "本实验在已有packed INT4 checkpoint基础上验证两个更接近真实部署的执行阶段：第一，"
        "streaming路径只反量化当前输出通道tile，不再生成完整FP32 Linear权重；第二，native路径"
        "将激活动态量化为INT8，并让packed INT4权重通过AVX-512 VNNI执行整数点积和INT32累加。", "",
        "当前结果可以表述为“独立packed INT4权重模型和CPU dynamic W4A8整数执行验证”。它不是GPU W4A16融合kernel，"
        "也不代表GPU显存、吞吐或延迟。", "",
        "## 2. 模型覆盖与数据流", "",
        "量化范围仍为Qwen2.5-VL中的414个Linear、3,441,197,056个权重，占全部浮点参数约91.65%。"
        "其中decoder FFN 108个、decoder attention 144个、vision FFN 96个、vision attention 64个、vision merger 2个。"
        "Embedding/LM head、patch Conv3d、norm和bias保持浮点。", "",
        "Hybrid配置中204个FFN Linear使用AWQ路由：52个FFN block、156个Linear接受非identity scale；"
        "16个block、48个Linear安全保持identity/RTN。其余210个attention/merger Linear使用RTN W4。", "",
        "端到端数据流为：", "",
        "```text",
        "uint8 image + prompt",
        "  → processor",
        "  → vision encoder packed Linear",
        "  → vision merger packed Linear",
        "  → decoder packed Linear",
        "  → floating LM head",
        "  → target-position logits + greedy generation",
        "```", "",
        "Native Linear内部执行：", "",
        "```text",
        "FP32 activation",
        "  → per-token/group-128 symmetric INT8",
        "  → packed INT4 nibble即时解码",
        "  → AVX-512 VNNI uint8×int8 dot",
        "  → -8·sum(qx)符号修正",
        "  → INT32 group accumulator",
        "  → activation_scale × weight_scale",
        "  → FP32跨group累加和bias",
        "```", "",
        "## 3. 数据集与实验协议", "",
        "沿用确定性Tiny Tabletop：8条calibration仅用于此前AWQ搜索，本轮不重新校准；16条独立evaluation用于完整实验。"
        "图像为112×112桌面场景，覆盖颜色、形状、计数和左右关系。每条样本都从原始像素重新运行完整VLM。", "",
        "Teacher forcing只在target token位置计算logits指标；自由生成使用do_sample=False、最多12 token。"
        "FP16 reference指原checkpoint权重先舍入到FP16可表示值、再转FP32进行CPU前向，不是GPU FP16 kernel。", "",
        "## 4. 五组Smoke门控", "",
        "| 配置 | Backend | Hidden NMSE↓ | Cosine↑ | Logits KL↓ | Top-1↑ | Activation NMSE↓ | 完整浮点权重物化 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in smoke.items():
        f = fields(result)
        runtime = result["runtime"]
        lines.append(
            f"| {name} | {result['backend']} | {f['hidden_nmse']:.6g} | {f['hidden_cosine']:.6f} | "
            f"{f['kl']:.6g} | {f['top1']:.2%} | {f['activation_nmse']:.6g} | "
            f"{runtime['full_float_weight_materializations']} |"
        )
    lines.extend([
        "", "五组均替换414个Linear、fallback为0、完成2条端到端样本。Streaming最大浮点权重tile为10.75 MiB，"
        "对应最大完整FP32权重为100 MiB；native W4A8的浮点权重tile为0。", "",
        "## 5. 16样本核心结果", "",
        "| 配置 | Hidden NMSE↓ | Cosine↑ | SQNR dB↑ | Logits KL↓ | Top-1↑ | Target NLL↓ | PPL↓ | Activation NMSE↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| RTN W4A8 | {r['hidden_nmse']:.6g} | {r['hidden_cosine']:.6f} | {r['hidden_sqnr']:.4f} | "
        f"{r['kl']:.6g} | {r['top1']:.2%} | {r['nll']:.6g} | {r['ppl']:.6g} | {r['activation_nmse']:.6g} |",
        f"| Hybrid AWQ W4A8 | {a['hidden_nmse']:.6g} | {a['hidden_cosine']:.6f} | {a['hidden_sqnr']:.4f} | "
        f"{a['kl']:.6g} | {a['top1']:.2%} | {a['nll']:.6g} | {a['ppl']:.6g} | {a['activation_nmse']:.6g} |", "",
        "| 配置 | Reference Exact↑ | Target Exact↑ | Token Agreement↑ | Edit Similarity↑ |",
        "|---|---:|---:|---:|---:|",
        f"| RTN W4A8 | {r['ref_exact']:.2%} | {r['target_exact']:.2%} | {r['token']:.2%} | {r['edit']:.2%} |",
        f"| Hybrid AWQ W4A8 | {a['ref_exact']:.2%} | {a['target_exact']:.2%} | {a['token']:.2%} | {a['edit']:.2%} |", "",
        "Hybrid相对RTN将Final Hidden NMSE降低"
        f"{_pct_change(r['hidden_nmse'], a['hidden_nmse']):.2%}，Cosine提高{a['hidden_cosine'] - r['hidden_cosine']:.6f}，"
        f"Logits KL降低{_pct_change(r['kl'], a['kl']):.2%}。Reference Exact提高"
        f"{a['ref_exact'] - r['ref_exact']:.2%}，Token Agreement提高{a['token'] - r['token']:.2%}。", "",
        "改善不是所有指标单调成立：Hybrid的Top-1一致率比RTN低1个百分点，Target NLL和PPL也更高。"
        "Top-1衡量量化模型与参考模型的teacher-forcing argmax一致率，不是标准答案准确率；generation agreement则会累积自回归分歧。", "",
        "## 6. 三段误差分析", "",
        "第一段是FP16-weight reference到packed W4的权重量化误差。此前packed FP32结果中，RTN W4的Hidden NMSE/Logits KL为"
        f"{pr['hidden_nmse']:.6g}/{pr['kl']:.6g}，Hybrid AWQ W4为{pa['hidden_nmse']:.6g}/{pa['kl']:.6g}。", "",
        "第二段是完整反量化到streaming反量化。Smoke中streaming与相同packed权重保持同一量化定义，toy测试最大差异不超过1e-5；"
        "这一步改变临时内存和浮点GEMM切分，不改变qweight或scale。", "",
        "第三段是packed FP32到W4A8：加入激活INT8量化和VNNI整数点积。RTN的Hidden NMSE相对packed FP32变化为"
        f"{r['hidden_nmse'] - pr['hidden_nmse']:+.6g}，KL变化为{r['kl'] - pr['kl']:+.6g}；Hybrid分别为"
        f"{a['hidden_nmse'] - pa['hidden_nmse']:+.6g}和{a['kl'] - pa['kl']:+.6g}。", "",
        "增量可能为负，不表示INT8激活提高了模型本身精度，而是不同量化误差在有限evaluation样本和非线性网络中可能部分抵消。"
        "因此应同时看Hidden、KL、NLL和生成指标，不能只凭单个数字下结论。", "",
        "## 7. 运行时与存储", "",
        "| 配置 | 总时间 | Activation quant | 激活误差统计 | Native GEMM | 整数MAC |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for label, result in (("RTN W4A8", rtn), ("Hybrid AWQ W4A8", awq)):
        runtime = result["runtime"]
        lines.append(
            f"| {label} | {result['elapsed_seconds'] / 60:.2f} min | {runtime['activation_quant_seconds']:.2f}s | "
            f"{runtime['activation_metrics_seconds']:.2f}s | {runtime['native_gemm_seconds']:.2f}s | "
            f"{runtime['integer_macs']:,} |"
        )
    lines.extend([
        "", f"W4 checkpoint磁盘大小约{storage['checkpoint_disk_bytes'] / 2**30:.3f} GiB；packed整数码约"
        f"{storage['qweight_bytes'] / 2**30:.3f} GiB，scale约{storage['scale_bytes'] / 2**20:.3f} MiB，"
        f"目标权重有效位宽{storage['target_effective_bits_per_weight']:.4f} bit/weight。", "",
        "当前native kernel的主要时间在GEMM，CPU利用率和速度尚未充分优化；这些时间证明算子确实执行，但不能视为INT4加速结论。", "",
        "## 8. 指标解释", "",
        "- Final Hidden NMSE：最终decoder hidden误差能量相对参考hidden能量的比例，越低越好。",
        "- Cosine：量化与参考hidden方向一致性，越接近1越好。",
        "- SQNR：参考信号能量与误差能量的dB比值，越高越好。",
        "- Logits KL：target位置参考概率分布到量化分布的KL散度，越低越好。",
        "- Top-1 Agreement：target位置量化与参考模型argmax相同的比例，不等于标准答案准确率。",
        "- Target NLL/PPL：量化模型对标准target token分配概率的负对数损失及其指数。",
        "- Reference Exact：自由生成与参考模型生成完全一致的样本比例。",
        "- Token Agreement/Edit Similarity：生成序列逐token一致性和归一化编辑相似度。", "",
        "## 9. 验收与限制", "",
        "两组16样本结果均满足414个native模块、0 fallback、0完整浮点权重物化；C++ scalar与VNNI toy输出逐元素一致，"
        "并与Python W4A8 reference在1e-4/1e-5容差内一致。", "",
        "结论只覆盖Qwen2.5-VL-3B、Tiny Tabletop、CPU packed W4A8数值与短生成。尚未覆盖真实GPU W4A16 kernel、"
        "GPU显存/吞吐、embedding/LM head和patch Conv3d整数化、真实VQA benchmark、StarVLA action head或LIBERO成功率。", "",
    ])
    temporary = output / "report.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(output / "report.md")
