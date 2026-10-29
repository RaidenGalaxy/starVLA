# ASAP7 / F2 / 500MHz SRAM 迁移

本轮实现将活动架构固定为两级，核心 1GHz、DMA/SRAM 500MHz、DRAMSim3 1ns。`configs/architecture.properties` 为共同配置源；最终频率仍由物理证据决定。

## 基线与身份

改造前源码归档于 `artifacts/migration_asap7_f2/baseline/source.tar.gz`，`manifest.json` 保存原源码、依赖锁和历史报告的 SHA256。新统一回归通过 `scripts/freeze_verification.py` 冻结 Chisel、RTL、仿真桥、脚本、测试、物理流程及全部工艺资源，构建和每个检查前后都核对同一源码身份。

2211 份原历史报告已逐一与基线哈希核对，全部原始字节独立保存至 `baseline/historical_reports.tar.gz`；`baseline/historical_reports_index.json` 记录归档身份，没有缺失或被修改后冒充原结果的条目。

## SRAM 真实硬宏依赖

候选源为 OpenROAD 官方 `asap7_sram_0p0`，锁定提交 `522eeccbccefcd66e61893fa1059df24d95e9f86`。所选接口为 `srambank_256x4x64_6t122`，对应 1024×64bit。公开 GDS 内包含位单元及外围电路结构，但其顶层为 `srambank_64b`，并非所选 LEF/Liberty 单元；不能只改名字就认为容量、针脚和版图已经匹配。

`python3 scripts/audit_sram.py` 输出实际 GDS 单元清单、层号、资源哈希、工具发现结果和缺失项。真实验收仍需匹配 GDS、晶体管网表、DRC/LVS/PEX，以及与 TT/SS/FF 一致的时序表征。工具或资源缺失时，这些项保持失败；不会生成虚构的物理证明。

同时核对了上述固定提交的完整上游文件树：151 个文件，包括三份 GDS、生成的 LEF/Liberty 和行为 Verilog；没有 SPICE/CDL 晶体管网表或 DRC/LVS/PEX/表征报告。原始路径清单见 `artifacts/migration_asap7_f2/upstream_sram_tree.json`，来源为 [官方固定版本文件树](https://github.com/The-OpenROAD-Project/asap7_sram_0p0/tree/522eeccbccefcd66e61893fa1059df24d95e9f86)。因此该缺口并非仅由本地下载范围造成。

提供完整宏包时，在宏资源目录放置 `physical_evidence.json`，指定 `macro`、所有设计视图的 `views_sha256`，以及 `drc/lvs/pex/characterization_TT/characterization_SS/characterization_FF` 的报告文件、SHA256 和通过状态。各表征记录还需提供 `liberty_file` 与 `verified_period_ps=2000`。生产审查入口为 `physical/sram_signoff.py`：逐角读取锁定标准单元库的实际默认工作点，核对工艺、电压和温度，缺失默认工作点时才读取名义值；本机 TT 为 0.70V／25℃。同名 TT 标签不能代替实际 PVT 比对。物理入口同时核对实际文件，不接受只有通过标志的清单。冻结回归中的旧 `scripts/audit_sram.py` 保留历史固定温度规则，不用于新的物理资格判定。

## 工具不足时的 SRAM 替代实现

用户确认没有完整工具后，已增加 [OpenFinRAM](https://github.com/shao-chien-lu/OpenFinRAM) 开源生成路径，固定提交 `751afff49d7f6ea6071c366732fbbe80c5f4c2e6`。编译器、五个子模块、补丁及 KLayout wheel 的身份记录在 `physical/sram_compiler.lock.json`。补丁固定控制器周期为 2ns，适配本机 OpenROAD 命令，增加显式的未验收版图导出模式；默认仍拒绝使用失败的控制器结果。

实际生成配置为 128 条物理字线、上下折叠、4 路列复用、64bit 数据宽度，逻辑容量为 **1024×64bit**。编译器的原生单元名为 `sram_x256x64x1`，名称中的 256 表示折叠后的物理字线数，不能据此把逻辑深度误判为 256。GDS 和晶体管网表各自数得 **65536 个活动 6T 位单元**，148 个引脚名匹配；16 份该宏对应系统的 128KiB。

宏包在 `artifacts/migration_asap7_f2/sram_openfinram/package.json`，包含 GDS、LEF、独立晶体管网表、估算 Liberty、三个 PVT 模型卡、逻辑功能封装、控制器失败报告和逐文件哈希。GDS/LEF 已统一原点，并把 Fin/Gate 伸出的几何包含在 30.888×77.372µm 的抽象边界内，电路拓扑不变。配套接口模型通过 2099 次写、2101 次读和 30 次读改写验证，控制复位不清空已提交数据。

```bash
python3 physical/build_sram.py --prepare --emit-unverified-layout \
  --out artifacts/my_generated_sram
python3 physical/run.py --target system --stages 2 --rows 2 --period 1 \
  --sram-package artifacts/my_generated_sram --synth-only --mapping-profile fast
```

输出目录采用不可覆盖策略，重复构建应指定新的 `--out`。综合时替换真实使用的宏封装并记录新 RTL/宏包身份，禁止给既有冻结网表重新贴标签。系统研究性布局布线仍需显式传入 `--allow-unverified-sram`。

**此替代方案交付的是已构建的 6T SRAM 研究原型。** 控制器在 2ns 下的 setup WNS 为 −0.102299727ns，Liberty 为估算模型；尚无完整 DRC/LVS/PEX 和三角表征，500MHz 未验收。编译器上游的开源表征[说明](https://github.com/shao-chien-lu/OpenFinRAM/blob/751afff49d7f6ea6071c366732fbbe80c5f4c2e6/docs/characterization_plan.md)也明确区分了估算库、器件级测量和完整表征。没有降低 500MHz 目标或用原型的估算值宣布时序通过。

商业 `v2lvs` 的替代转换器为 `physical/verilog_to_spice.py`：使用 Yosys 展开实际门级连接，严格按真实 CDL 库的引脚顺序生成实例；缺失输入直接失败，已确认的悬空输出及无信号的填充/衬底单元单独记录。该转换结果提供原理图参考，不等同于版图提取或 LVS。

替代宏已接入 2×2 TileSystem 综合：A/B/C/汇总区为 2/4/2/8 个原生硬宏，全部 128KiB 保留；标准单元为 232988.516636µm²，宏占地为 38237.861376µm²，尚无完整系统布局面积。实际综合网表的 8 个 FIFO、16 组／76bit Gray 指针生成与约束检查通过，见 `artifacts/migration_asap7_f2/generated_sram_system_diagnostic.json`。当前生产解析器为 `physical/mapped_system_constraints.py`，可穿过映射器插入的反相器和缓冲器定位真实寄存器引脚；直接寄存器测试夹具保留原文件，避免改变冻结的功能源码。实际系统物理构建会保存约束生成器及 Tcl 辅助脚本副本和哈希。

`physical/build_sram_sim.py` 从上述已综合网表对应的原始 RTL 快照构建匹配的宏模型联仿，同时核对宏封装、宏包和全部输入 RTL 的身份。该 2×2 系统已通过辅助计算、多时钟 DMA、汇总区最高地址 8191、跨 bank 隔离和结果背压检查，原始数据在 `artifacts/migration_asap7_f2/generated_sram_auxiliary`。这属于数字逻辑模型验证；晶体管行为和实际频率保持独立未验收状态。

同一替代宏的 2×2 GEMM 综合也已完成，标准单元为 235684.125355µm²，宏占地为 38237.861376µm²，全部 128KiB 保留。`physical/check_system_mapping.py` 可重新检查实际综合网表的容量和双时钟 CDC 约束，不产生物理通过证书。

```bash
python3 physical/build_sram_sim.py \
  --snapshot .build/physical/asap7_TT_gemm_2_2x2_1ns_openfinram \
  --sram-package artifacts/migration_asap7_f2/sram_openfinram \
  --cohort artifacts/migration_asap7_f2/functional/cohort.json
python3 physical/verify_sram_system.py \
  --build-dir .build/qualified_gemm_2_2x2_3bd2490c8625_openfinram_model \
  --cohort artifacts/migration_asap7_f2/functional/cohort.json \
  --out artifacts/my_generated_sram_verification
python3 physical/report_migration.py
```

以上构建目录名对应本次实际 RTL 身份；后续源码改变时使用构建命令打印的新目录。物理入口强制保存 PDN 失败通孔及 VDD/VSS 连接检查；仅设置全局电源网名不能算作物理供电完成。

已用真实 16 个宏运行供电布局探针，参考策略因所有宏网格均无可连接的形状／通孔而报 `PDN-0233`，见 `artifacts/migration_asap7_f2/sram_power_probe2/verification.json`。可通过 `physical/probe_sram_power.py` 在昂贵的整阵列布线前复现；该问题必须连同宏引脚接入和内部验证解决，不能以综合通过代替。

供电接入现已有可复核的修复版本：`sram_openfinram_power_access2` 在实际 GDS 中为原 M3 电源针脚添加 M4 电源金属和真实 VIA34 通孔，并按实际 GDS 保守生成 M4/M5 障碍，保留信号金属与下层阵列的保护。65536 个 6T 位单元、晶体管原理图及逻辑模型均保持原身份；新增金属的区域差分、间距和引脚覆盖单独记录，不冒充完整 DRC/LVS。

该版本通过了 16 宏 M5/M6 网格的 VDD/VSS 连通检查，见 `sram_power_probe5/verification.json`。保留参考行电源轨的较早抽象版本也通过了 `sram_power_probe4`；该探针没有标准单元，不认证完整系统供电、电迁移或压降。构建步骤为：

```bash
.deps/sram-tools/bin/python physical/extend_sram_power.py \
  --source artifacts/migration_asap7_f2/sram_openfinram \
  --out artifacts/my_sram_power
.deps/sram-tools/bin/python physical/refine_sram_obstructions.py \
  --source artifacts/my_sram_power --out artifacts/my_sram_power_access
python3 physical/probe_sram_power.py \
  --sram-package artifacts/my_sram_power_access --out artifacts/my_sram_power_probe
```

原始宏包与 14 项原生宏联仿结果保留原身份。供电修订版复用了逐字节一致的逻辑封装和理想晶体管原理图；这些数字模型证据不包含新增金属的寄生、模拟行为或频率认证。

后续控制器 hold 修复已形成更新的宏包 **`sram_openfinram_eco_access`**。该版本把实际重新布线的控制器同步装配到 GDS 和理想晶体管网表，核对全部控制器端口坐标和共用标准单元几何，并证明修复前后 463 个等价点全部通过。随后重新添加、核对供电金属及障碍；`sram_power_probe6` 确认 16 个最新宏的 VDD/VSS 连接通过。原始宏及所有失败尝试均保留原身份。

控制器在编译器 TT 库下的 2ns setup／hold WNS 为 +0.304815739ns／+0.047623947ns，布线 DRC 和电气检查无违规；改用核心平台锁定的 TT 库重新提取检查，裕量为 **+0.272263557ns／+0.052300528ns**。两组库均为 0.70V／25℃，分别保留独立库与提取哈希。固定 `sdel=0` 后仅四个硬连配置寄存器的数据端点不参加动态时序，覆盖审查逐个核对这些端点；其他缺约束会保持失败。该证据只覆盖控制器，最新宏 Liberty 尚未重新表征，整块 SRAM 的 500MHz 仍未验收。

对同一实际控制器 ODB 补充了锁定标准单元库的 SS／FF 角提取检查，未改变 2ns 目标、网表或配置模式。实际工作点从 Liberty 的默认 operating condition 读取，结果如下。

| 标准单元角 | 电压／温度 | setup WNS (ns) | hold WNS (ns) |
|---|---|---:|---:|
| TT | 0.70V／25℃ | +0.272263557 | +0.052300528 |
| SS | 0.63V／100℃ | −0.023455460 | +0.097663239 |
| FF | 0.77V／0℃ | +0.437795699 | +0.028125864 |

SS 的约 23ps setup 缺口保留为失败证据，最差路径为 `clk → saprechn[0]`。三角检查均确认动态端点覆盖完整；它们仅覆盖控制器和平台 RC 提取，不构成 SRAM 阵列的三角表征，也不代表完整宏具备匹配的 Liberty。复现使用 `physical/check_sram_mode_timing.py --library-profile architecture --corner SS`（或 `FF`）；历史 `architecture_tt25` 参数作为兼容别名保留，具体电压和温度以报告字段为准。

可复现的控制器修复与宏装配入口为 `physical/check_sram_mode_timing.py --repair-hold`、`physical/verify_sram_controller_eco.py`、`physical/integrate_sram_controller.py`，之后依次执行供电扩展和障碍抽取。宏装配必须以未经供电扩展的原始宏为父版本，防止旧供电通道的间距证明被误用于新的内部布线。

开源晶体管诊断入口为 `physical/prepare_analog.py` 和 `physical/check_sram_bitcell.py`。工具锁定在 `physical/analog_tools.lock.json`：本地 ngspice 42、OpenVAF 23.5.0、BSIM-CMG107 的确切源文件和编译器哈希均记录。BSIM-CMG107 由加州大学伯克利分校开发，保留上游版权及许可声明；来源为 Xyce 仓库的固定提交。移植只适配 Xyce 元数据、等值的参数边界常量写法，以及模拟器的模型／器件声明，未修改 Berkeley 模型方程或 ASAP7 的物理参数。FinFET 的 L／NFIN 保持原值，原理图 W 注释保存在报告中。

最新宏中实际使用的 `sram_cell_6t_122` 已在 TT／SS／FF 三角下，完成写 1、保持、读 1、读取后保留，以及相应的 0 状态检查，共 24 个检查点全部通过。证据见 `sram_bitcell_transistor/verification.json`。该试验使用明确的 10fF 外部位线负载和理想外围驱动，操作间隔为 2ns；它没有提取阵列布线寄生，不认证完整宏的读延迟、噪声裕量或 500MHz 运行能力。

`physical/check_sram_macro_transistor.py` 另保留完整 1024×64 宏的 407016 个晶体管，执行地址 0 的双极性写回／读取试验。首轮 UIC 启动在约 9.3ps 时出现奇异矩阵并中止，`sram_macro_transistor_tt/verification.json` 明确保持失败；仿真器退出码为 0 不能使中止的波形通过检查。工作点初始化重试具有独立目录和运行时上限，不覆盖原失败证据。完整宏功能、DRC/LVS、阵列 PEX 和 Liberty 表征均仍须各自取得。

完整宏的工作点初始化重试达到 600 秒上限，未得到有效读写波形。为隔离启动问题，又保留实际控制器、一个完整 1024 位列及其感放／写驱动／预充电电路作缩减诊断；两次工作点试验均达到 120 秒上限。缩减列明确移除了其他列的负载，不能认证完整宏的频率。上述失败和未完成状态见各自的 `verification.json`，未改写为成功。

随后加入显式标记为非物理的 1TΩ 数值收敛辅助，列级工作点诊断仍在 120 秒后超时，未取得波形，见 `sram_column_transistor_tt_shunt/verification.json`。该选项仅用于定位求解问题，不代表实际漏电或寄生模型。

## 验证和结果

完整功能入口：`make migration-check`。机器报告为 `artifacts/migration_asap7_f2/functional/acceptance.json`，日志和逐项数据均留在同一目录。运行中断或失败会留下已完成步骤和明确的失败日志；`functional_passed` 仅在所有必选场景通过后设为真。

本次冻结主回归已完成 **57/57** 项，原生生成宏的 2×2 联仿完成 **14/14** 项，均通过。完整 8×8 已覆盖全格式 GEMM、分组量化、313 个输出转换描述符、预取对照、37896 次 Scratch 事务、三种复位域、错误注入和四种相位；冻结源码身份仍为 `e4545dac1946…`。这些是配置时钟下的数字功能结果，整体验收仍因物理缺项保持未完成。

另完成原生宏封装对应的 8×8 综合与模型联仿：29 个描述符覆盖全部 11 种格式，128KiB 全地址／掩码／隔离验证共 37896 次事务，均通过。源码身份为 `a2a817ccaeb3…`，实际宏包为供电修订版 `728faa7c76ff…`，原始证据见 `generated_sram_gemm8/verification.json`。该结果不重新贴成最新控制器 ECO 的物理结果；最新 ECO 使用相同逻辑封装，并另有控制器等价和几何匹配证明。

SRAM 适配器独立验证包括深度跨 bank 地址、全字及部分字节掩码、读改写锁、同步读与复位后的数据保留。系统验证包括 128KiB 全地址、两级算术延迟、全部格式、K 边界、量化汇总、预取、输出转换、DRAM 排空、两路外部复位和固定周期下的相位扫查。测试桥打印实际时钟参数/相位以及实际拉起的复位输入，验收脚本核对现场记录。

物理诊断位于 `.build/physical/asap7_*`。原 F2 的完整 FMA 曾集中在输入寄存器后的单个周期，首次 ASAP7 CTS 探针未达到 1GHz；现已在固定两周期内重新安排寄存边界。最终结论须使用更新后相同网表的提取报告，不得拼接不同版本的 setup、hold 或数值结果。

性能是配置频率下的 RTL/DRAMSim3 预测，包含部分写的读改写开销、单端口仲裁和最终写排空。DMA 单方向理论接口上限为 8GB/s，实际吞吐由运行统计给出；不承诺全双工或等比例加速。

`artifacts/migration_asap7_f2/acceptance_matrix.json` 汇总各里程碑的证据和缺项；`performance_projection.json` 汇总已通过场景的实际计数、最终排空时间及 RTL/仿真器/DRAM 哈希。运行中的检查保持待完成状态。冻结逻辑接口的完整 57 项、原生宏 2×2 的 14 项，以及原生宏 8×8 的补充验证分别保留准确范围与实现身份。

## 已取得的物理诊断

FP64 独立两级算术组件在 1ns、ASAP7 RVT TT 下完成了布线与寄生提取诊断：setup WNS **−3.627387047ns**、hold WNS **+0.035016976ns**、TNS **−705.801818848ns**。布线 DRC、覆盖与约束策略检查通过，电气检查失败，频率未通过。可核对 `artifacts/migration_asap7_f2/physical_diagnostic.json` 中的原始网表、库、SDC、SPEF 和报告哈希。该诊断不覆盖完整 PE、阵列或 SRAM 系统。

同一布线后 FP64 网表通过 1200 个独立参考向量和随机时钟使能测试。这是零延迟门级数值检查，不能替代上述失败的提取后时序结果。

另保留了隔离的两级 FP64 优化副本：将操作选择移到乘法器之后，1200 个向量通过，CTS setup WNS 为 −2.953575611ns、hold WNS 为 −0.016942088ns。`physical/experiment_fp64.py` 在该输入快照上探索无附加寄存器的乘法压缩树，先验证独立数值参考，再综合和检查 CTS。此类副本没有替换冻结的主设计，也不能混用其 RTL/时序结果作为主设计验收证据。

压缩树快映射副本通过 4000 个 RTL／门级向量，CTS setup WNS 改善至 **−1.506628036ns**。零乘积独立处理、前缀保留属性精简两个后续副本各通过 44000 个 RTL 向量，setup WNS 分别为 −1.761987448ns、−1.610175729ns，没有接入主设计。逐版本身份和证据见 `fp64_optimization_history.json`；这些 CTS 结果均未达到 1ns，也不能替代提取后验收。

后续 one-hot 规格化副本通过 53788 个独立 FP64 向量，其中包含 5000 个接近相消的 FMA；CTS setup／hold WNS 为 **−1.271216035ns／−0.043812551ns**。保持乘积为进位保存形式并融合加数的副本通过相同规模数值检查，setup WNS 为 −1.359835625ns。one-hot 副本的完整布线和提取在独立目录进行，不能把 CTS 估计报告当成布线后结果。

one-hot 副本现已完成实际布线和提取：setup WNS **−1.418866158ns**、hold WNS **+0.072032273ns**、TNS **−287.944885254ns**。布线 DRC、约束与覆盖检查通过，电气检查仍失败；同一布线后网表的 4000 个零延迟参考向量全部通过。证据见 `fp64_onehot_routed_diagnostic.json`，不能与较早主设计的报告拼接为验收结果。radix-4 乘法副本通过 53788 个 FP64 向量及小位宽全部输入的形式检查，但 CTS setup WNS 为 −1.661776662ns，未改善最佳副本，因此保留为未采用的实验。

加／减与乘／乘加分开计算、共享既有第二级结果寄存器的副本也通过 53788 个 FP64 向量，算术仍为两级；CTS setup／hold WNS 为 −1.306100011ns／−0.113833889ns，未改善最佳副本。将已有中间寄存边界前移到宽位加法之前的副本通过 61788 个参考向量，但 CTS setup／hold WNS 为 −1.496372104ns／−0.038689207ns，仍未改善最佳副本。九个已完成 CTS 实验的身份、一个综合前端失败及后续 pending 方案均保存在 `fp64_optimization_history.json`。

完整冻结 PE 的 CTS setup／hold WNS 为 **−3.150645494ns／−0.324179143ns**，最差路径从 `io_in_bits_ctx[0]` 出发，主要延迟来自上下文读取的高扇出缓冲链。`physical/experiment_pe_context.py` 创建独立源码副本，以局部 32 位 one-hot 选择器实现完全组合的 16 上下文读取；组合等价证明通过，原算术寄存边界和读取延迟保持不变。该副本与经过认证的 FPU 输入快照结合后，通过全部 11 种格式的 74052 个 ALU 向量（背压及空闲恢复），以及 PE、状态读取、启动间隔检查；物理结果按副本单独记录。`--all-read-ports` 同时改造观察读端口，具有独立验证身份。

仅改造算术读取和同时改造观察读取两个副本的 CTS setup WNS 分别为 **−1.881908298ns／−1.921812057ns**，均未闭合 1ns。前者的新最差路径从格式输入出发，已据此启动局部组合格式译码探针；改造没有增加寄存器。各副本的实际输入、功能日志及 CTS 报告均独立保存。

INT64 独立物理入口为 `physical/run.py --target int64 --rtl-build <已验证的ALU构建目录> --period 1 --cts-only`。它从实际生成 ALU 固定格式为 I64，保留动态操作、144 位累加、掩码、转换、同步复位和握手；新 RTL 的身份与父 ALU 分开。`physical/check_int64.py <物理目录>` 已在连续发射和随机背压／空闲恢复两种模式下，各验证 10000 个独立参考向量，连续发射延迟严格为两周期。该组件结果不替代全格式 ALU 或完整 PE 验收。

同一 INT64 组件的映射后网表也通过上述两种模式各 10000 个向量；其 CTS setup／hold WNS 为 **−1.317819118ns／−0.090507828ns**。`int64_component_diagnostic.json` 同时绑定 RTL、门级数值检查和该 CTS 结果。针对整数转换的组合选择逻辑，另有保持两级的固定切片探针，结果按实际完成状态记录。

首个固定切片转换副本的 CTS setup／hold WNS 为 −1.361766458ns／−0.191488698ns，未改善基线，因此未合入。`physical/check_integer_conversion.py` 对全部五种整数格式，覆盖每个有效前导位置、正负数、RNE 奇偶半值边界、掩码，并在连续和背压／空闲模式下各验证 11100 个向量。后续注册前缀副本把平衡前缀结果与原有幅值寄存器并行采样，算术仍严格两周期；其测量单独记录。局部格式译码 PE 副本的 CTS setup／hold WNS 为 −2.020753384ns／−0.343869060ns，也未合入。

注册前缀转换副本已通过上述功能检查，INT64 CTS setup／hold WNS 为 −1.022802830ns／−0.185312837ns。`--integer-fast-sum` 仅将既有第二级的整数加、减及乘加求和替换为保留完整宽度的平衡前缀加法器，不增加算术级数；其 setup／hold WNS 为 −1.171057820ns／−0.199213400ns，未改善最佳副本。最初仅凭起点依赖累加器低位而推测为进位链并不充分；进一步逐单元检查发现长 AND5 链实际消费 `arithmetic._conversions_prefixes_30_leading`，来自转换结果的 144 路 Mux1H。`--balanced-integer-select` 据此保留显式平衡的结果选择树，并复用已注册的全幅值 OR 判断零，不增加算术级数。

SRAM 控制器另完成实际布线寄生提取：全可编程模式 setup/hold WNS 为 −0.039598770ns／−0.001399228ns；封装实际硬连的 `sdel=0` 工作模式为 **+0.322669357ns／−0.001399228ns**，电气检查无违规。模式约束核对了四个实际配置寄存器的连接与反相极性，适用于配置捕获后的固定模式，不能豁免动态地址、使能和数据时序。`sram_controller_mode/verification.json` 保留该结果；宏原型的 500MHz 资格仍未通过。

2×2 GEMM 系统的综合网表保留 A/B/C/汇总区的 2/4/2/8 个宏和全部 128KiB 容量，见 `artifacts/migration_asap7_f2/mapped_system_diagnostic.json`。此诊断绑定较早的 RTL 快照，不认证当前完整系统。标准单元面积为 235030.226935µm²；16 个候选宏的 1× LEF 占地为 37757.76768µm²，合计 272787.994615µm²。候选宏 Liberty 的 `area=0`，因此不能用该属性代表物理面积；实际整体布局面积尚未取得。

功能源码与参与功能测试的策略辅助模块在回归期间保持冻结。物理流程另存参考快照，实际物理结果绑定执行的 flow、SDC、网表、提取和库文件哈希；物理流程调试不会改变已经固定的功能 RTL。

`physical/check_system_floorplan.py` 将已认证综合网表与显式选择的物理宏修订版进行布局／供电检查，要求原生引脚及功能封装完全相同，并分别记录综合网表原宏身份和当前物理宏身份。该检查保留完整标准单元行供电网格，不能用仅含宏的供电探针替代；它处于标准单元布局之前，不认证完整系统物理时序。实测发现的 `remove_from_collection` 兼容性问题已在生产 SDC 生成器 `physical/mapped_system_constraints.py` 中修复；冻结功能策略夹具未被改写。修复后的端口选择语义由 Tcl 执行检查覆盖，实际系统仍须在 OpenROAD 中完成后续约束与物理验证。

当前 OpenSTA 不支持 `set_max_delay -datapath_only`；生产 Gray 约束现使用 `-ignore_clock_latency`，以实际发射寄存器实例为起点，保持库 setup 检查在预算内。仅使用 Q 引脚会增加内部引脚起点路径而遗漏真实的时钟发射路径，已由实际工具回归捕获并修复。回归刻意将源／目标时钟插入延迟设为 10ns／5ns，确认跨域最大延迟仍为 0.8 倍源周期、跨域 hold 被单独豁免，且第一／第二同步级之间的 setup、hold 都正常计时。物理工具测试当前共 25 项通过，完整系统时序验收仍须完成。

完整 2×2 系统的 `system_floorplan_eco4` 已生成约 1008MiB 的 ODB 和 409MiB 的 DEF，实际完整标准单元行供电网格的 VDD、VSS 连通性均通过；核心／存储时钟约束分别为 1ns／2ns。该结果仍处于标准单元布局之前。另实测发现 OpenDB 的全局供电连接会跳过 `dont_touch` 同步器，`physical/protected_power.tcl` 因此只在连接其 POWER／GROUND 引脚的数据库操作期间临时释放保护，并在返回前恢复；所有信号连接及正常时序检查保留。实际 OpenROAD 回归验证了连接、保护恢复、重复调用和信号网络不变。生产物理入口现在归档宏布局脚本、该辅助脚本和供电策略的执行副本及哈希。

`physical/repair_system_power.py` 已在上述真实系统数据库中接通 152 个受保护实例的 304 个电源脚，并核对其信号连接、保护实例集合均未改变。供电脚同时标记为 special pin；修复前后完整 DEF 字节一致，实际网格几何未改变，因此可沿用经过认证的父供电网连通性证据，见 `system_floorplan_protected_power2/verification.json`。首轮遗漏 special 标记而导致 DEF 多出普通供电 NETS 的失败也单独保留。生产流程还会在 CTS／ECO 新增单元后重新连接电源，并在详细布线后执行包含已放置标准单元的 VDD／VSS 检查；这些最终检查尚未完成，不能由布局前结果替代。

`python3 physical/index_arithmetic_experiments.py` 从独立实验目录更新两份算术实验索引，核对数值检查、映射网表、CTS 检查点及各步骤日志的身份；未完成时序的方案保留为 pending。索引中的功能通过和 CTS 测量不认证目标频率。

`python3 physical/snapshot_migration.py` 可更新源码快照、`change_inventory.json` 和 `changes.patch`。它先校验 M0 压缩包及所有原始文件哈希，再使用确定性的归档元数据保存当前源码；既有归档保留。`implementation/source_manifest.json` 只是当前源码索引，实验目录和历史功能结果仍保留各自的执行输入与身份。随后运行 `python3 physical/report_migration.py` 更新验收矩阵；该入口同时核对物理工具测试日志及其实际测试源码，拒绝静默复用不匹配的工具测试结果。
