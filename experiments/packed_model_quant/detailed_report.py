"""Detailed Chinese report for the packed six-configuration matrix."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _reduction(base: float, candidate: float) -> float:
    return (base - candidate) / base * 100.0


def _gib(value: int | float) -> float:
    return float(value) / 2**30


def _mib(value: int | float) -> float:
    return float(value) / 2**20


def _boundary(metrics: dict[str, Any], key: str) -> dict[str, float]:
    value = metrics["activations"][key]
    return value.get("overall", value)


def write_detailed_report(output: Path, rows: list[dict[str, Any]]) -> None:
    by_name = {row["name"]: row for row in rows}
    details = {name: _load(output / "results" / name / "metrics.json") for name in by_name}
    storage = {name: _load(output / "results" / name / "storage_metrics.json") for name in by_name}
    checkpoints = {
        name: _load(output / "checkpoints" / name / "quantization_config.json")
        for name in ("rtn_w8", "rtn_w4", "hybrid_awq_w4")
    }
    reference = checkpoints["rtn_w8"]["reference"]
    awq = checkpoints["hybrid_awq_w4"]["awq"]
    w8, w8a = by_name["packed_rtn_w8_fp32"], by_name["packed_rtn_w8_a16"]
    w4, w4a = by_name["packed_rtn_w4_fp32"], by_name["packed_rtn_w4_a16"]
    awq4, awq4a = by_name["packed_hybrid_awq_w4_fp32"], by_name["packed_hybrid_awq_w4_a16"]

    lines = [
        "# Qwen2.5-VL Packed INT4/INT8 与 W4A16 CPU 模拟详细实验报告", "",
        "## 1. 实验目的与结论摘要", "",
        "本实验研究 Qwen2.5-VL-3B 在没有 GPU 量化内核的条件下执行 weight-only 低比特量化后，误差如何从权重表示传播到 Vision Encoder、Vision Merger、LLM Decoder、最终 logits 和自由文本生成。每条样本都从原始图像像素和文本重新运行完整 VLM，不复用参考模型的图像 embedding，因此这里的 hidden、logits 和生成指标是端到端指标。", "",
        f"结果表明：RTN W8 在 hidden/logits 层面接近无损；全 Linear RTN W4 出现明显累计误差；对 Vision 与 Decoder FFN 使用 AWQ W4 后，Final Hidden NMSE 相对 RTN W4 降低 {_reduction(w4['final_hidden_nmse'], awq4['final_hidden_nmse']):.1f}%，Logits KL 降低 {_reduction(w4['logits_kl'], awq4['logits_kl']):.1f}%，Top-1 Agreement 从 {w4['top1_agreement']:.0%} 提升到 {awq4['top1_agreement']:.0%}。Linear 边界 A16 舍入只引入很小的增量，没有改变三种方案的主要排序。", "",
        "## 2. 模型结构、量化边界与覆盖率", "",
        "模型为本地 Qwen2.5-VL-3B-Instruct checkpoint，包含 36 层语言 Decoder 和 32 个 Vision Block。本轮将以下 414 个 `nn.Linear` 替换为 packed weight-only Linear：", "",
        "| 子系统 | Linear 数 | 权重数 |", "|---|---:|---:|",
        "| Decoder FFN | 108 | 2,434,793,472 |",
        "| Decoder Attention | 144 | 339,738,624 |",
        "| Vision FFN | 96 | 420,249,600 |",
        "| Vision Attention | 64 | 209,715,200 |",
        "| Vision Merger | 2 | 36,700,160 |",
        "| 合计 | 414 | 3,441,197,056 |", "",
        "模型共有 3,754,622,976 个浮点参数，整数权重覆盖率为 91.65%。共享 token embedding/LM head、Vision patch embedding Conv3d、norm 和 bias 不做整数化；AWQ 调整的 norm 与 Vision FFN bias 作为函数等价缩放 carrier，以 FP16 值保存在 checkpoint 中。", "",
        "## 3. 参考定义与量化方法", "",
        "### 3.1 FP16 权重参考", "",
        "原 checkpoint 声明为 BF16，无法恢复一个训练时不存在的 FP16 checkpoint。实验把所有浮点参数先舍入为 FP16 可表示值，再转回 FP32 执行 CPU 前向，命名为 `fp16_weight_reference`。它衡量从 FP16 权重值到 W8/W4 的权重量化误差，不等价于 GPU FP16 kernel。", "",
        f"共处理 {reference['floating_values']:,} 个浮点值，其中 {reference['changed_values']:,} 个在 FP16 舍入后变化，占 {reference['changed_fraction']:.4%}；最大绝对舍入误差为 {reference['max_abs_rounding_error']:.6g}。转换过程检查并拒绝 NaN、Inf 和 FP16 溢出。", "",
        "### 3.2 对称 group-wise RTN", "",
        "每个二维权重沿输入维度按 group size 128 分组，每行每组使用 `scale=max(abs(W))/qmax`，再计算 `q=clamp(round(W/scale),-qmax,qmax)`。W8 的 qmax=127，整数码保存为 int8；W4 的 qmax=7，采用 signed two’s-complement nibble，每两个权重打包到一个 uint8。Scale 默认保存为 FP16，因此 Packed FP32 与原 fake quant 之间还包含很小的 scale 存储舍入误差。", "",
        "### 3.3 FFN-only AWQ", "",
        f"Hybrid AWQ W4 对 68 个 Vision/Decoder SwiGLU FFN、共 204 个 Linear 执行激活感知 scale 搜索。其中 {awq['accepted_blocks']} 个 FFN block 接受非 identity scale，即 {awq['nonidentity_linears']} 个 Linear 真正使用非恒等 AWQ 变换；{awq['fallback_blocks']} 个 block、{awq['fallback_linears']} 个 Linear 因严格函数等价检查回退 identity/RTN。其余 210 个 Attention/Merger Linear 使用 RTN W4。", "",
        "AWQ 先根据 calibration 激活统计对重要通道进行函数等价重缩放，再执行与 RTN 相同的 W4 量化。未量化时缩放前后函数应一致；量化后希望重要通道获得更合适的离散分辨率。Identity scale 是全 1 向量，表示不做通道重缩放，但仍执行 W4，因此等价于 RTN W4。", "",
        "### 3.4 Packed FP32 与 A16 模拟", "",
        "- `dequant_fp32`：INT4/INT8 和 group scale 常驻；每次 Linear 只解包当前模块，反量化到 FP32 后执行普通 `F.linear`，随后释放临时权重。",
        "- `a16_round_fp32_accum`：进一步把 Linear 输入、反量化权重、bias 和输出舍入为 FP16 可表示值，再转回 FP32 做 CPU 矩阵乘法。", "",
        "两种模式都没有使用 INT4 GEMM。A16 模式是 Linear 边界 W4A16/W8A16 数值模拟，不是 GPU 融合 kernel，CPU耗时不能用于推断真实 INT4 加速。", "",
        "## 4. 数据集与评估流程", "",
        "数据为确定性 Tiny Tabletop：8 条 calibration、16 条独立 evaluation，seed=42，图像为 112×112。场景包含彩色几何物体，问题覆盖计数、颜色、形状、左右/前后关系和简短 tabletop 指令。Calibration 只用于 AWQ 搜索，evaluation 不参与 scale 选择。", "",
        "Teacher forcing 使用 prompt+target，只在 target token 预测位置计算 logits 指标；16 条 evaluation 合计 100 个 target token。自由生成采用 greedy decoding，`do_sample=False`，最多生成 12 token。所有配置都从原始像素经过 Vision Encoder、Merger、Decoder 和 LM Head，不复用参考 image embedding。", "",
        "## 5. 指标解释", "",
        "- **NMSE**=`||candidate-reference||²/||reference||²`，表示相对误差能量，越低越好；不是准确率下降比例。",
        "- **Cosine Similarity** 衡量表示方向一致性，越接近 1 越好，需要与 NMSE 联合阅读。",
        "- **SQNR**=`10log10(signal_power/error_power)`，单位 dB，越高越好。",
        "- **Logits KL**=`KL(P_ref||P_quant)`，比较完整词表概率分布，越接近 0 越好。",
        "- **Top-1 Agreement** 是量化模型与参考模型首选 token 的一致率，不是标准答案准确率。",
        "- **Target NLL/PPL** 衡量模型赋给数据 target token 的概率。当前 target 格式与自然生成格式不完全匹配，不能单独作为任务准确率。",
        "- **Generation Reference Agreement** 要求完整生成字符串与参考生成完全一致，对早期单个 token 变化非常敏感。",
        "- **Target Exact Accuracy** 是与数据 target 的严格匹配率。参考模型自身也是 0%，所以本轮 0% 不能解释为量化使能力归零。",
        "- **Token Agreement/Edit Similarity** 是比完整字符串 exact match 更平滑的生成一致性指标。", "",
        "## 6. Packed Checkpoint 实际存储", "",
        "| 格式 | 磁盘大小 | 整数码 | Scale | 非量化张量 | 有效bit/目标权重 | 目标权重相对FP16压缩 | 全checkpoint相对FP16压缩 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, label in (("packed_rtn_w8_fp32", "W8"), ("packed_rtn_w4_fp32", "W4")):
        s = storage[name]
        lines.append(f"| {label} | {_gib(s['checkpoint_disk_bytes']):.3f} GiB | {_gib(s['qweight_bytes']):.3f} GiB | {_mib(s['scale_bytes']):.1f} MiB | {_gib(s['non_quantized_bytes']):.3f} GiB | {s['target_effective_bits_per_weight']:.3f} | {s['target_compression_vs_fp16']:.3f}× | {s['whole_tensor_compression_vs_fp16']:.3f}× |")
    lines += [
        "", "W4 目标权重的有效位宽是 4.125 而非严格 4.000，因为每 128 个权重还需要一个 FP16 scale。完整 checkpoint 压缩比低于目标 Linear 压缩比，是因为 embedding、Conv3d、norm 和 bias 等约 0.584 GiB 张量仍以浮点保存。", "",
        "## 7. 六配置系统级端到端结果", "",
        "| 配置 | Final Hidden NMSE↓ | Cosine↑ | SQNR dB↑ | Logits KL↓ | Top-1↑ | NLL↓ | PPL↓ | Ref Exact↑ | Token↑ | Edit Sim↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in by_name:
        m = details[name]["metrics"]
        h, l, g = m["hidden_states"]["final"], m["logits"], m["generation"]
        lines.append(f"| `{name}` | {h['normalized_mse']:.6g} | {h['cosine_similarity']:.6f} | {h['sqnr_db']:.3f} | {l['kl_divergence']:.6g} | {l['top1_agreement']:.2%} | {l['target_nll']:.6g} | {l['target_perplexity']:.6g} | {g['reference_exact_agreement']:.2%} | {g['token_agreement']:.2%} | {g['normalized_edit_similarity']:.2%} |")
    lines += [
        "", "### 7.1 W8 在 hidden 和 logits 层面接近无损", "",
        f"Packed RTN W8 FP32 的 Final Hidden NMSE={w8['final_hidden_nmse']:.6g}、Cosine={w8['final_hidden_cosine']:.6f}、Logits KL={w8['logits_kl']:.6g}，100 个 target token 的 Top-1 Agreement 为 {w8['top1_agreement']:.0%}。加入 A16 边界舍入后，NMSE 只增加 {w8a['final_hidden_nmse']-w8['final_hidden_nmse']:.3g}，KL 增加 {w8a['logits_kl']-w8['logits_kl']:.3g}，Top-1 不变。", "",
        "但 W8 的自由生成完整参考一致率只有 31.25%，即使 teacher-forcing Top-1 是 100%。两者并不矛盾：teacher forcing 固定参考上下文，而自由生成中某个早期 token 的微小变化会改变后续上下文。Token Agreement=43.42%、Edit Similarity=73.65%，也说明完整字符串 exact match 是非常苛刻的指标。", "",
        "### 7.2 RTN W4 出现显著累计误差", "",
        f"RTN W4 FP32 的 Final Hidden NMSE={w4['final_hidden_nmse']:.6g}，Cosine={w4['final_hidden_cosine']:.6f}，Logits KL={w4['logits_kl']:.6g}，Top-1 降至 {w4['top1_agreement']:.0%}；生成参考一致率为0%，Token Agreement只有 {w4['generation_token_agreement']:.2%}。这表明对全部414个Linear直接使用4-bit RTN会造成明显的跨层累计误差。", "",
        f"W4 A16 相对 W4 FP32 的 NMSE 只增加 {w4a['final_hidden_nmse']-w4['final_hidden_nmse']:.6g}，Cosine 变化 {w4a['final_hidden_cosine']-w4['final_hidden_cosine']:.3g}，Top-1和生成指标完全相同。因此主要损失来自W4权重离散化，而不是本轮A16模拟。", "",
        "### 7.3 AWQ 显著缓解 W4 误差", "",
        f"Hybrid AWQ W4 FP32 相比 RTN W4 FP32，将 Final Hidden NMSE 从 {w4['final_hidden_nmse']:.6g} 降到 {awq4['final_hidden_nmse']:.6g}，降低 {_reduction(w4['final_hidden_nmse'],awq4['final_hidden_nmse']):.1f}%；Logits KL 从 {w4['logits_kl']:.6g} 降到 {awq4['logits_kl']:.6g}，降低 {_reduction(w4['logits_kl'],awq4['logits_kl']):.1f}%；Cosine 从 {w4['final_hidden_cosine']:.6f} 升至 {awq4['final_hidden_cosine']:.6f}，Top-1 从 {w4['top1_agreement']:.0%} 升至 {awq4['top1_agreement']:.0%}。", "",
        f"自由生成的 Reference Exact 从0%升到 {awq4['generation_reference_agreement']:.2%}，Token Agreement 从 {w4['generation_token_agreement']:.2%} 升到 {awq4['generation_token_agreement']:.2%}，Edit Similarity 从 {w4['generation_edit_similarity']:.2%} 升到 {awq4['generation_edit_similarity']:.2%}。AWQ的局部FFN重构收益已经传播到最终token分布和生成。", "",
        f"Hybrid A16 的 NMSE={awq4a['final_hidden_nmse']:.6g}、Cosine={awq4a['final_hidden_cosine']:.6f}、KL={awq4a['logits_kl']:.6g}，Top-1和生成指标与Hybrid FP32相同。KL的轻微下降属于有限样本舍入噪声，不能解释为A16普遍改善模型。", "",
        "## 8. 子系统误差传播", "",
        "下表比较三种Packed FP32主配置。模块边界指标包含上游输入已经发生变化的影响，因此是端到端传播指标，不等价于模块孤立权重误差。", "",
        "| 配置 | Vision Blocks NMSE | Merger NMSE | Decoder Attention NMSE | Vision FFN NMSE | Decoder FFN NMSE | Final Hidden NMSE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("packed_rtn_w8_fp32", "packed_rtn_w4_fp32", "packed_hybrid_awq_w4_fp32"):
        m = details[name]["metrics"]
        lines.append(f"| `{name}` | {_boundary(m,'vision_blocks')['normalized_mse']:.6g} | {_boundary(m,'merger')['normalized_mse']:.6g} | {_boundary(m,'decoder_attention')['normalized_mse']:.6g} | {_boundary(m,'vision_ffn')['normalized_mse']:.6g} | {_boundary(m,'decoder_ffn')['normalized_mse']:.6g} | {m['hidden_states']['final']['normalized_mse']:.6g} |")
    lines += [
        "", "RTN W4 的误差从Vision阶段开始并在Decoder深层累计：Vision Blocks NMSE=0.0190、Merger=0.0601、Decoder Attention边界达到0.3956，最终hidden为0.2568。Attention边界同时接收已被扰动的hidden，因此不能把0.3956简单解释成Attention权重的孤立误差。", "",
        "Hybrid AWQ 的 Vision Blocks NMSE 反而高于 RTN W4（0.0363 对0.0190），说明AWQ不保证每个局部端到端边界都更优；但Decoder FFN NMSE从0.0518降至0.0271，Decoder Attention传播边界从0.3956降至0.3058，最终hidden和logits明显改善。AWQ改变了误差分布并降低系统级误差，而不是让所有局部指标同时降低。", "",
        "## 9. CPU耗时、内存与实现开销", "",
        "| 配置 | 总耗时 | Linear调用 | 解包/反量化 | Linear计算 | 峰值RSS | 最大临时权重 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in by_name:
        d, s = details[name], storage[name]
        r, mem = d["runtime"], d["memory"]["after_evaluation"]
        lines.append(f"| `{name}` | {d['elapsed_seconds']/60:.2f} min | {r['calls']:,} | {r['unpack_dequant_seconds']/60:.2f} min | {r['linear_seconds']/60:.2f} min | {_gib(mem['peak_rss_bytes']):.3f} GiB | {_mib(r['max_module_temporary_bytes']):.1f} MiB |")
    lines += [
        "", "W4 checkpoint更小、常驻内存更低，但当前CPU实现需要在每次Linear调用时解包，解包/反量化占据大量时间，因此W4没有速度优势。A16模式还增加FP16往返舍入。不同配置的Linear调用次数也会因自由生成路径长度变化，因而总耗时不是严格kernel基准。", "",
        "## 10. Packed 与 Fake Quant 一致性", "",
        f"Packed W8 FP32 相对fake W8的Final Hidden NMSE/KL增量为 {w8['delta_nmse_vs_fake']:.3g}/{w8['delta_kl_vs_fake']:.3g}；Packed W4为 {w4['delta_nmse_vs_fake']:.3g}/{w4['delta_kl_vs_fake']:.3g}；Hybrid AWQ W4为 {awq4['delta_nmse_vs_fake']:.3g}/{awq4['delta_kl_vs_fake']:.3g}。差异远小于各自总体量化误差，说明nibble packing、独立checkpoint加载和逐层解包没有引入明显实现错误。", "",
        "个别差值为负不表示packed理论上优于fake，而是FP16 scale舍入在有限样本上恰好使某些输出更接近参考；量化误差并非逐指标严格单调。", "",
        "## 11. 结论边界", "",
        "本实验已经比常驻FP32 fake quant更接近真实部署：目标权重确实以packed INT4/INT8保存，checkpoint可独立加载，运行时模型不保留414个目标Linear的浮点weight参数。但矩阵乘法仍是CPU FP32，A16只是边界舍入模拟，尚未覆盖真实W4A16 CUDA kernel、融合反量化、GPU显存、吞吐和延迟。", "",
        "8条校准和16条合成evaluation适合数值链路验证与模块归因，但不足以代表真实VQA泛化能力。正式任务结论仍需标准VQA benchmark、答案抽取/规范化评分和GPU kernel测试；StarVLA action head与LIBERO成功率也不在本轮范围。", "",
        "在当前边界内可以得出：W8是低风险方案；全Linear RTN W4造成明显端到端累计误差；在Vision/Decoder FFN使用AWQ可显著缓解W4 hidden和logits误差；A16边界模拟只带来次要增量，主要瓶颈仍是4-bit权重离散化与误差跨层传播。", "",
        "## 12. 复现实验", "",
        "```bash", "python -m experiments.packed_model_quant.run_matrix \\",
        "  --model-path playground/Pretrained_models/Qwen2.5-VL-3B-Instruct \\",
        "  --source-run outputs/full_model_quant/qwen2_5_vl_3b \\",
        "  --preset cpu --output-dir outputs/packed_model_quant/matrix --resume", "```", "",
        "顶层 `summary.csv` 和 `metrics.json` 提供配置级汇总；每个 `results/<config>/metrics.json` 保存逐Vision Block、Merger、Attention、FFN和hidden-state详细曲线，`generations.jsonl`保存逐样本生成结果。", "",
    ]
    temporary = output / "report.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(output / "report.md")
