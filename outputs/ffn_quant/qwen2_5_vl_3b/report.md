# Qwen2.5-VL Decoder FFN 权重量化小规模实验报告

> 本实验把原始 checkpoint 权重加载为 FP32，在 CPU 上执行 fake quantization。它衡量的是仅量化 decoder FFN 权重后产生的数值和语言输出偏差，不等价于 GPU 上真实 FP16 与 INT4/W4A16 kernel 的精度、显存、延迟或吞吐对比。

## 1. 实验目的

Qwen2.5-VL-3B 的语言 decoder 中，FFN 权重占模型参数的大部分。本实验保持 vision encoder、projector/跨模态部分、attention、embedding 和 LM head 不变，只改动每层 MLP 的 `gate_proj`、`up_proj`、`down_proj`，回答三个问题：全层 FFN W8 是否接近原始参考；W4 从最后 1 层扩展到全部 36 层时误差如何累积；FFN-only AWQ 能否降低相同 W4 quantizer 的误差。

这里研究的是受控数值误差。实验没有让模型自由生成后做答案精确匹配，也没有运行 StarVLA action head 或 LIBERO，因此不能把 token 指标直接解释为机器人成功率。

## 2. 模型、量化范围与实现

- 模型 checkpoint：`playground/Pretrained_models/Qwen2.5-VL-3B-Instruct`。
- Decoder 层数：36。
- 目标 Linear：108 个，即 36 层 × 3 个 FFN projection。
- 目标权重：2,434,793,472 个，占全模型参数 64.85%。
- `last1` 为第 35 层，`last4` 为第 32～35 层，`all` 为第 0～35 层。
- Group size=128，沿权重输入维度分组；RTN W4 范围 `[-7,7]`，W8 范围 `[-127,127]`。
- 每组 scale=`max(abs(W_group))/qmax`，执行 quantize→dequantize 后仍用 FP32 Linear 前向。
- AWQ W4 根据 calibration 激活搜索函数等价 scale，再使用同一个 W4 group-128 quantizer；候选包含 identity，严格等价检查失败时回退到 identity/RTN。

每个配置开始前都恢复原始 FFN 和相关 norm 权重，结束后恢复并校验，避免不同量化配置相互污染。

## 3. 数据集与评估协议

### 3.1 Tiny Tabletop 数据集

数据由 Pillow 确定性生成，不依赖外部图片。每张图为 112×112 RGB 浅色桌面场景，包含 3 个彩色几何物体。颜色来自红、蓝、绿、黄，形状包括圆形、正方形、三角形，物体位于左上、中央偏下、右上三个位置。

问题按四类循环：识别最左物体、识别最右物体、根据颜色回答形状、根据形状回答颜色。例如 `Which object is furthest to the left on the tabletop?` → `The yellow square.`，以及 `What shape is the green object?` → `It is a triangle.`。

- 随机种子：42。
- Calibration：8 条，`calib_0000`～`calib_0007`，仅用于 AWQ 激活统计和 scale 搜索。
- Evaluation：16 条，`eval_0000`～`eval_0015`，不参与 scale 搜索。
- 两个 split 的 ID、图片和随机场景不重叠。
- 16 条 evaluation 共 100 个可预测 target token，Logits 指标在这些 token 上求平均。

该数据覆盖基础颜色、形状和左右关系，适合检查量化数据流和数值误差；但样本少、图片简单，不代表真实 VLM benchmark、长文本生成或机器人操作分布。

### 3.2 Teacher forcing 与参考缓存

输入由图像、prompt 和 target 组成，采用 teacher forcing：预测每个 target token 时，模型能看到此前的真实 target token。首先用原始 checkpoint 权重执行 FP32 CPU 前向，缓存每层 FFN 输出、每层 decoder hidden state，以及 target token 位置的 logits。随后每个量化配置与同一参考逐项比较。

为降低磁盘占用，参考 FFN、hidden state 和 logits 缓存以 FP16 保存；候选前向仍为 FP32。因此未量化层也可能出现约 `1e-8` 的缓存舍入底噪。原始参考行按定义记为零误差和完全一致。

## 4. 评价指标的含义

### 4.1 FFN NMSE、Cosine 与 SQNR

FFN normalized MSE 定义为 `||Y_quant-Y_ref||²/||Y_ref||²`，越接近 0 越好。它表示相对误差能量，例如 NMSE=0.04 是误差平方能量约占参考信号平方能量 4%，不表示准确率下降 4%。overall 值合并了 36 层、全部 evaluation 样本和 token。

FFN Cosine 衡量参考与量化 FFN 输出方向是否一致，越接近 1 越好。SQNR=`10log10(||Y_ref||²/||error||²)`，单位 dB，越大越好；它与 NMSE 是同一误差的不同表达。

### 4.2 Final Hidden NMSE 与 Final Hidden Cosine

Final Hidden 是最后一层 decoder 的输出表示。Final Hidden NMSE 同时反映方向和幅值偏差；Final Hidden Cosine 比较量化与参考 hidden vector 的方向，1 表示完全同向。Cosine 对整体幅值缩放不敏感，因此必须与 NMSE 联合阅读。

它比单层权重 MSE更接近 LM head 实际接收到的表示，因为前面各层产生的误差可能沿 decoder 深度累积、放大或抵消。

### 4.3 Logits KL

在每个 target token 预测位置，把参考和量化 logits 转成完整词表概率分布，计算 `KL(P_ref||P_quant)`，再对 100 个 target token 求平均。0 表示分布相同，越大表示量化后对词表概率的分配偏移越明显。KL 关注整个分布，即使最高概率 token 没变，它也可能上升。

### 4.4 Top-1 Agreement

比较参考模型和量化模型在每个 target token 位置的 argmax 是否相同。100% 表示量化模型的首选 token 始终与参考一致。

它不是答案正确率：如果两者同时预测错误也会记为一致。本实验有 100 个 target token，因此 96% 约表示 4 个位置的首选 token 改变，99% 约表示 1 个位置改变。

### 4.5 Target NLL 与 Perplexity

Target NLL 是模型赋给真实 target token 的平均负对数概率，越低表示 teacher forcing 下给标准 token 的概率越高；Perplexity=`exp(NLL)`。但量化噪声可能偶然提高这 100 个特定 token 的概率，所以 NLL 下降不能单独证明量化模型更好，必须结合 KL、Top-1 和 hidden-state 指标判断。

## 5. 完整实验结果

| 方法 | 范围 | 位宽 | FFN NMSE ↓ | FFN Cosine ↑ | SQNR/dB ↑ | Final Hidden NMSE ↓ | Final Hidden Cosine ↑ | Logits KL ↓ | Top-1 Agreement ↑ | Target NLL ↓ | PPL ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 原始权重参考 | none | 原始 | 0 | 1.000000 | Infinity | 0 | 1.000000 | 0 | 100% | 1.54491 | 4.6875 |
| RTN | all | W8 | 0.000125679 | 0.999938 | 39.01 | 0.00210374 | 0.998949 | 0.000128406 | 100% | 1.54072 | 4.6680 |
| RTN | last1 | W4 | 0.0027533 | 0.998657 | 25.60 | 0.0193414 | 0.990296 | 0.00778239 | 100% | 1.55033 | 4.7130 |
| RTN | last4 | W4 | 0.004771 | 0.997833 | 23.21 | 0.037859 | 0.980941 | 0.00887394 | 100% | 1.54012 | 4.6651 |
| RTN | all | W4 | 0.0399064 | 0.983941 | 13.99 | 0.203587 | 0.897787 | 0.0608058 | 96% | 1.33406 | 3.7964 |
| AWQ | last1 | W4 | 0.0027533 | 0.998657 | 25.60 | 0.0193414 | 0.990296 | 0.00778239 | 100% | 1.55033 | 4.7130 |
| AWQ | last4 | W4 | 0.004771 | 0.997833 | 23.21 | 0.037859 | 0.980941 | 0.00887394 | 100% | 1.54012 | 4.6651 |
| AWQ | all | W4 | 0.0257347 | 0.987608 | 15.89 | 0.165395 | 0.916852 | 0.0319901 | 99% | 1.46704 | 4.3364 |

## 6. 结果分析

### 6.1 W8 全层 FFN 基本保持参考行为

RTN W8 量化全部 36 层后，FFN NMSE=0.000125679，Final Hidden Cosine=0.998949，Logits KL=0.000128406，Top-1 Agreement=100%。说明当前数据上 W8 误差很小，经过 36 层传播后仍基本保持原始输出分布，是本轮最稳定的量化设置。

### 6.2 W4 误差随量化层数增加而累积

仅量化最后 1 层时，RTN W4 的 Final Hidden Cosine=0.990296，KL=0.00778239，Top-1 仍为 100%。扩展到最后 4 层后，Final Hidden Cosine 降至 0.980941，FFN NMSE 从 0.0027533 增至 0.004771，说明多层扰动开始叠加，但尚未改变首选 token。

全部 36 层使用 RTN W4 后，FFN NMSE=0.0399064，Final Hidden NMSE=0.203587，Final Hidden Cosine=0.897787，KL=0.0608058，Top-1 降至 96%。这说明单层看来可接受的 W4 误差会在深层 decoder 中累计并改变少量 token 的首选预测。

RTN W4 all 中 FFN NMSE 最大的 5 层为：L1=1.24752、L6=0.370378、L31=0.356027、L5=0.344111、L33=0.331307。逐层 hidden cosine 最低的 5 个深度位置为：L35=0.897787、L1=0.969001、L34=0.98706、L33=0.988806、L32=0.989436。这些值用于定位敏感层，不能直接解释为任务准确率。

### 6.3 AWQ 对全层 W4 有明显改善

AWQ W4 all 将 FFN NMSE 从 0.0399064 降到 0.0257347，相对降低 35.5%；KL 从 0.0608058 降至 0.0319901，相对降低 47.4%。Final Hidden Cosine 从 0.897787 回升到 0.916852，Top-1 从 96% 回升到 99%。

全层搜索中 26/36 层采用了通过严格等价检查的 AWQ scale，10 层回退到 identity/RTN。AWQ W4 all 中 FFN NMSE 最大的 5 层为：L31=0.286464、L33=0.255715、L34=0.224769、L32=0.20205、L22=0.121163。

这些结果支持 activation-aware scaling 能缓解全层 W4 误差，但 AWQ W4 all 仍明显差于 W8 all，不能说 W4 已经近乎无损。

### 6.4 AWQ last1/last4 为什么与 RTN 相同

AWQ last1 有 0/1 层采用非 identity scale，AWQ last4 有 0/4 层采用非 identity scale。所选末端层均因 `strict_fp32_equivalence_tolerance` 触发安全回退，之后使用与 RTN 相同的 W4 quantizer，因此指标完全相同。

这不是 AWQ 对末层必然无效，而是当前缩放融合实现、严格容差、校准样本和这些层激活共同产生的结果。后续应检查等价缩放位置与数值容差，并扩大 calibration 数据后复测。

### 6.5 NLL 更低不代表 RTN W4 all 更好

原始参考 NLL=1.54491，RTN W4 all=1.33406，表面上量化后更低。但后者的 KL 最大、Final Hidden Cosine 最低且 Top-1 降至 96%。更低 NLL 很可能只是噪声偶然提高了这批标准 token 的概率，不能证明整体概率分布或任务能力更优。

## 7. 当前结论

1. 在 8 条校准、16 条评估的合成 tabletop 小实验中，全层 FFN RTN W8 基本保持原始权重参考行为，可作为低风险基线。
2. W4 仅量化末端 1 层或 4 层时影响较小，但全部 36 层后误差明显累积，说明全层 W4 需要 activation-aware 方法或混合精度策略。
3. AWQ W4 all 相比 RTN W4 all 明显改善 FFN 重构、最终 hidden state 和 logits 分布，证明将 AWQ 迁移到 VLM decoder FFN 值得继续推进。
4. AWQ W4 all 尚未达到 W8 的稳定程度，末端层还存在严格等价检查回退问题。
5. 当前数据仅适合代码验收和早期方案筛选，不足以形成通用 VLM 精度结论。

## 8. 局限性与下一步

- 这是 fake quantization，没有 INT4 packing/CUDA kernel，不能报告真实显存、吞吐或延迟。
- 参考是 CPU FP32，不是 GPU FP16；GPU 阶段需重新建立 FP16 baseline。
- 只有 16 张合成图、100 个 target token且使用 teacher forcing；需要真实 VLM benchmark和自由生成答案评分。
- 尚未评估 StarVLA action head，不能外推为 LIBERO 成功率。
- 下一步先诊断末端层 `strict_fp32_equivalence_tolerance` 回退，确认缩放融合未量化时严格等价。
- 然后扩大 calibration/evaluation，增加逐层敏感性混合精度对照，并在 GPU 上评估真实 W4A16 kernel。

## 9. 结果文件

- `summary.csv`：配置级汇总。
- `metrics.json`：全部配置总体及逐层指标。
- `<configuration>/metrics.json`：单配置量化/AWQ 搜索记录。
- `awq_scales.pt`：逐层 AWQ scale。
- `module_inventory.json`：目标模块、shape、参数量和占比。
- `config.json`：模型路径、样本 ID、seed、环境版本和配置哈希。
