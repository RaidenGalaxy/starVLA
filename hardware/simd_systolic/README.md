# 固定两级 SIMD 脉动阵列（ASAP7）

Chisel / Verilator / DRAMSim3 工程，支持运行时切换的全部 11 种整数和浮点格式。当前唯一支持的算术配置是 **F2 两级**，最终阵列为 8×8；单 PE 和 2×2 用于分级验证。

| 配置 | 固定值 |
|---|---|
| 工艺 | ASAP7 7.5T RVT / NLDM |
| 核心时钟 | 1000ps，1GHz 验收目标 |
| DMA 与 SRAM 时钟 | 2000ps，500MHz 验收目标 |
| DRAMSim3 tCK | 1.0ns，其余周期参数不变 |
| SRAM | 128KiB，16 个 1024×64bit 单端口宏 |

配置源是 `configs/architecture.properties`。Python 构建拒绝 HF8/HF12、旧架构快照及改变后的依赖身份。Chisel 生成器也只接受两级。SRAM 访问和 CDC 等待不属于算术的两周期延迟。

**配置频率下的功能通过不代表物理时序通过。真实 SRAM 宏和完整系统的物理验收仍须查看证据。** 当前公开候选宏缺少与所选单元匹配的 GDS、晶体管网表及完整表征/签核证据。硬宏模型能用于联仿，但不能据此宣称真实 SRAM 已交付。详见 [迁移说明](MIGRATION_ASAP7.md)。

工具不足时使用 OpenFinRAM 构建 1024×64 的实际 6T SRAM 研究原型。最新包为 `artifacts/migration_asap7_f2/sram_openfinram_eco_access`：控制器修复已同步到 GDS／晶体管网表，通过 463 个等价点证明、2ns TT 提取后 setup／hold 检查和 16 宏供电连通探针。Liberty 仍为估算模型，完整 SRAM 的 DRC/LVS/PEX、晶体管行为及 PVT 表征未完成，500MHz 整宏保持未验收。构建与接入命令见迁移说明。

## 构建与统一验收

需要 Java 17、sbt、Verilator、Python 3、CMake、C++17；物理流程另需锁定的 Yosys、OpenROAD 工具。依赖版本与 SHA256 在 `dependencies.lock.json` 中。

```bash
cd /home/ubuntu/starVLA/hardware/simd_systolic
python3 scripts/bootstrap.py --physical
make migration-check JOBS=8
```

`migration-check` 冻结当前源码并构建 ALU、PE、8×8 阵列、2×2 系统及 2×2/8×8 GEMM；执行全格式、周期、全地址 SRAM、部分写、预取、输出转换、错误注入、单独/共同复位和相位偏移测试。结果保存在 `artifacts/migration_asap7_f2/functional/acceptance.json`。这份报告只验收功能，物理频率字段保持独立。

单独构建和检查：

```bash
python3 scripts/build.py --kind gemm --stages 2 --rows 8 --hierarchical --sim-threads 4
python3 scripts/check_descriptor.py --stages 2 --rows 8 --cols 8 --dram \
  --build-dir .build/gemm_2_8x8_mt4 --out artifacts/descriptor_f2_asap7
python3 scripts/check_sram_bank.py
python3 physical/sram_signoff.py --require-certified
```

## 存储与接口

A/B/C 容量保持 16/32/16KiB，分别由 2/4/2 个 8KiB 宏组成；分组汇总区为 64KiB、8 个宏。A/B/C 使用两个 64bit 宏并行保持 128bit 端口。单端口的部分字节写由适配器执行原子的读改写，计入真实访存字节和等待周期。SRAM 内容不被控制复位整体清零。

`GemmDescriptor`、外部 Scratch 请求/返回和 DMA 数据格式保持兼容。内部窥读与汇总接口现在使用请求/返回握手；写完成在存储提交后返回。任一外部复位都会中止当前事务，并让两个 FIFO 端点共同置位复位、分别同步释放，防止单域复位遗留旧响应。

浮点 F2 将原输入寄存边界移入 FMA 算术内部，保持两周期和精确单次 RNE 舍入；宽加减与舍入使用显式进位树。整数及运行时格式语义保持原有定义，算术结果与独立整数/有理数参考模型逐位比较。

## 物理流程

```bash
python3 physical/run.py --target fp64 --stages 2 --corner TT --period 1 --cts-only --label probe
python3 physical/run.py --target pe --stages 2 --corner TT --period 1
python3 physical/run.py --target gemm --stages 2 --rows 2 --corner TT --period 1 --synth-only
```

支持 TT/SS/FF 库选择。所有 SDC 显式使用 ns/fF，ASAP7 的原始 ps/fF Liberty 数值在合并时保持不变。系统流程提供双时钟 SDC、按综合网表生成的 Gray 总线约束、16 宏布局、供电和容量检查。真实宏未认证时默认拒绝系统布局布线；`--allow-unverified-sram` 只允许研究性流程调试，不能产生整体验收证书。

1GHz 未闭合时保留失败路径，不增加流水级、不降低验收频率。CTS 报告、零延迟门级仿真和提取后时序报告分别记录。旧 Nangate45/HF 结果保留为历史证据，不能认证新架构。
