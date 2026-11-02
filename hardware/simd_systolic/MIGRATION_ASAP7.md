# ASAP7 / F2 / 500MHz SRAM 迁移

本轮实现将活动架构固定为两级，核心 1GHz、DMA/SRAM 500MHz、DRAMSim3 1ns。`configs/architecture.properties` 为共同配置源；最终频率仍由物理证据决定。

当前异步映射已修正：旧系统自动映射的复位／置位脚存在功能错误，不能用于验收。新显式映射通过三角 Liberty 事件验证，系统副本全部 4123 个相关触发器完成引脚与连接核对；完整系统门级回归及修正后时序继续执行。当前工具回归 51 项通过，整体验收仍未完成；下文保留各次实验及失败历史。

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

列级电源渐升试验在 84.8ps 再次出现奇异矩阵及步长失败，尚未覆盖有效读写。另将源漏电阻改写为参数不变的等价电压支路形式，在独立 OSDI 中进行默认求解器／KLU 位单元试验，也未稳定启动，因此未采用。原始 OSDI 保持不变，并在 `bitcell_model_recheck.json` 中复核了全部 24 项 TT／SS／FF 检查，结果与原报告完全相同。新增失败不会被当作 SRAM 500MHz 的验收证据。

另一个独立诊断将名义为零的外部片电阻节点合并，移除 CMG107 的 1mΩ 数值下限；24 个单元检查通过，与原波形采样的最大差异约 87nV，但实际列仍在约 9.41ps 于控制器节点 `n301` 中止。该诊断明确记录方程修改，未采用为原始模型。为继续隔离 OSDI 移植与电路收敛问题，新增 `physical/prepare_xyce.py` 和 `physical/check_sram_xyce.py`，在本地构建并运行原生 CMG107；Xyce、Trilinos 和 SuiteSparse 的提交、构建命令及工具身份由 `physical/xyce_tools.lock.json` 锁定。原生工具构建或电路检查未完成时，报告保持待完成或失败。

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

加／减与乘／乘加分开计算、共享既有第二级结果寄存器的副本也通过 53788 个 FP64 向量，算术仍为两级；CTS setup／hold WNS 为 −1.306100011ns／−0.113833889ns，未改善最佳副本。将已有中间寄存边界前移到宽位加法之前的副本通过 61788 个参考向量，但 CTS setup／hold WNS 为 −1.496372104ns／−0.038689207ns；随后保留规格化平衡树的副本为 −1.596941590ns／−0.026972938ns。实际减法控制高扇出路径促成预取反副本，在原边界捕获已取反的加数及进位，61788 个参考向量通过，CTS 改善至 **−1.115403891ns／−0.046232268ns**，仍未闭合 1ns。已完成、失败及 pending 方案分别保存在 `fp64_optimization_history.json`，任何 CTS 改善都不能替代提取后验收。

完整冻结 PE 的 CTS setup／hold WNS 为 **−3.150645494ns／−0.324179143ns**，最差路径从 `io_in_bits_ctx[0]` 出发，主要延迟来自上下文读取的高扇出缓冲链。`physical/experiment_pe_context.py` 创建独立源码副本，以局部 32 位 one-hot 选择器实现完全组合的 16 上下文读取；组合等价证明通过，原算术寄存边界和读取延迟保持不变。该副本与经过认证的 FPU 输入快照结合后，通过全部 11 种格式的 74052 个 ALU 向量（背压及空闲恢复），以及 PE、状态读取、启动间隔检查；物理结果按副本单独记录。`--all-read-ports` 同时改造观察读端口，具有独立验证身份。

仅改造算术读取和同时改造观察读取两个副本的 CTS setup WNS 分别为 **−1.881908298ns／−1.921812057ns**，均未闭合 1ns。前者的新最差路径从格式输入出发，已据此启动局部组合格式译码探针；改造没有增加寄存器。各副本的实际输入、功能日志及 CTS 报告均独立保存。

INT64 独立物理入口为 `physical/run.py --target int64 --rtl-build <已验证的ALU构建目录> --period 1 --cts-only`。它从实际生成 ALU 固定格式为 I64，保留动态操作、144 位累加、掩码、转换、同步复位和握手；新 RTL 的身份与父 ALU 分开。`physical/check_int64.py <物理目录>` 已在连续发射和随机背压／空闲恢复两种模式下，各验证 10000 个独立参考向量，连续发射延迟严格为两周期。该组件结果不替代全格式 ALU 或完整 PE 验收。

同一 INT64 组件的映射后网表也通过上述两种模式各 10000 个向量；其 CTS setup／hold WNS 为 **−1.317819118ns／−0.090507828ns**。`int64_component_diagnostic.json` 同时绑定 RTL、门级数值检查和该 CTS 结果。针对整数转换的组合选择逻辑，另有保持两级的固定切片探针，结果按实际完成状态记录。

首个固定切片转换副本的 CTS setup／hold WNS 为 −1.361766458ns／−0.191488698ns，未改善基线，因此未合入。`physical/check_integer_conversion.py` 对全部五种整数格式，覆盖每个有效前导位置、正负数、RNE 奇偶半值边界、掩码，并在连续和背压／空闲模式下各验证 11100 个向量。后续注册前缀副本把平衡前缀结果与原有幅值寄存器并行采样，算术仍严格两周期；其测量单独记录。局部格式译码 PE 副本的 CTS setup／hold WNS 为 −2.020753384ns／−0.343869060ns，也未合入。

注册前缀转换副本已通过上述功能检查，INT64 CTS setup／hold WNS 为 −1.022802830ns／−0.185312837ns。`--integer-fast-sum` 仅将既有第二级的整数加、减及乘加求和替换为保留完整宽度的平衡前缀加法器，不增加算术级数；其 setup／hold WNS 为 −1.171057820ns／−0.199213400ns，未改善最佳副本。最初仅凭起点依赖累加器低位而推测为进位链并不充分；进一步逐单元检查发现长 AND5 链实际消费 `arithmetic._conversions_prefixes_30_leading`，来自转换结果的 144 路 Mux1H。`--balanced-integer-select` 据此保留显式平衡的结果选择树，并复用已注册的全幅值 OR 判断零，不增加算术级数。

平衡选择树副本的 CTS setup／hold WNS 为 −1.303078771ns／−0.238425285ns；最差路径转移到乘法器前的掩码缓冲链。`--late-integer-mask` 保留原结果及溢出掩码，移除对数值结果冗余的乘法输入掩码门控；功能与时序结果在独立副本中验证。

延后整数掩码的副本通过功能和舍入检查，CTS setup／hold WNS 为 **−0.809013546ns／−0.258647323ns**，最差路径转为 `a[63]` 至第一级乘积寄存器。后续 `--integer-product-pair` 在既有第一级边界保留两个带符号 Booth 压缩项，第二级再与 144 位累加值合并；没有增加流水级。小位宽证明、8 位全部输入枚举以及全格式功能检查分别保留证据，不能当作 64 位全输入形式证明。物理工具读取该网表时触发的解析器内存错误及各次修复尝试也保留在独立目录。

FP64 的 `choices` 映射保持原有 one-hot RTL 和两级寄存边界，CTS setup／hold WNS 为 **−1.016848445ns／−0.025288397ns**，同一映射网表通过 4000 个参考向量。8 位分块进位选择副本及双符号规格化副本均通过 61788 个参考向量，CTS setup WNS 分别为 −1.309365630ns、−1.169266701ns，未改善最佳结果。这些副本均未接入冻结主设计，也未闭合 1ns。

SRAM 控制器另完成实际布线寄生提取：全可编程模式 setup/hold WNS 为 −0.039598770ns／−0.001399228ns；封装实际硬连的 `sdel=0` 工作模式为 **+0.322669357ns／−0.001399228ns**，电气检查无违规。模式约束核对了四个实际配置寄存器的连接与反相极性，适用于配置捕获后的固定模式，不能豁免动态地址、使能和数据时序。`sram_controller_mode/verification.json` 保留该结果；宏原型的 500MHz 资格仍未通过。

2×2 GEMM 系统的综合网表保留 A/B/C/汇总区的 2/4/2/8 个宏和全部 128KiB 容量，见 `artifacts/migration_asap7_f2/mapped_system_diagnostic.json`。此诊断绑定较早的 RTL 快照，不认证当前完整系统。标准单元面积为 235030.226935µm²；16 个候选宏的 1× LEF 占地为 37757.76768µm²，合计 272787.994615µm²。候选宏 Liberty 的 `area=0`，因此不能用该属性代表物理面积；实际整体布局面积尚未取得。

功能源码与参与功能测试的策略辅助模块在回归期间保持冻结。物理流程另存参考快照，实际物理结果绑定执行的 flow、SDC、网表、提取和库文件哈希；物理流程调试不会改变已经固定的功能 RTL。

`physical/check_system_floorplan.py` 将已认证综合网表与显式选择的物理宏修订版进行布局／供电检查，要求原生引脚及功能封装完全相同，并分别记录综合网表原宏身份和当前物理宏身份。该检查保留完整标准单元行供电网格，不能用仅含宏的供电探针替代；它处于标准单元布局之前，不认证完整系统物理时序。实测发现的 `remove_from_collection` 兼容性问题已在生产 SDC 生成器 `physical/mapped_system_constraints.py` 中修复；冻结功能策略夹具未被改写。修复后的端口选择语义由 Tcl 执行检查覆盖，实际系统仍须在 OpenROAD 中完成后续约束与物理验证。

当前 OpenSTA 不支持 `set_max_delay -datapath_only`；生产 Gray 约束现使用 `-ignore_clock_latency`，以实际发射寄存器实例为起点，保持库 setup 检查在预算内。仅使用 Q 引脚会增加内部引脚起点路径而遗漏真实的时钟发射路径，已由实际工具回归捕获并修复。回归刻意将源／目标时钟插入延迟设为 10ns／5ns，确认跨域最大延迟仍为 0.8 倍源周期、跨域 hold 被单独豁免，且第一／第二同步级之间的 setup、hold 都正常计时。物理工具测试当前共 35 项通过，完整系统时序验收仍须完成。

完整 2×2 系统的 `system_floorplan_eco4` 已生成约 1008MiB 的 ODB 和 409MiB 的 DEF，实际完整标准单元行供电网格的 VDD、VSS 连通性均通过；核心／存储时钟约束分别为 1ns／2ns。该结果仍处于标准单元布局之前。另实测发现 OpenDB 的全局供电连接会跳过 `dont_touch` 同步器，`physical/protected_power.tcl` 因此只在连接其 POWER／GROUND 引脚的数据库操作期间临时释放保护，并在返回前恢复；所有信号连接及正常时序检查保留。实际 OpenROAD 回归验证了连接、保护恢复、重复调用和信号网络不变。生产物理入口现在归档宏布局脚本、该辅助脚本和供电策略的执行副本及哈希。

`physical/repair_system_power.py` 已在上述真实系统数据库中接通 152 个受保护实例的 304 个电源脚，并核对其信号连接、保护实例集合均未改变。供电脚同时标记为 special pin；修复前后完整 DEF 字节一致，实际网格几何未改变，因此可沿用经过认证的父供电网连通性证据，见 `system_floorplan_protected_power2/verification.json`。首轮遗漏 special 标记而导致 DEF 多出普通供电 NETS 的失败也单独保留。生产流程还会在 CTS／ECO 新增单元后重新连接电源，并在详细布线后执行包含已放置标准单元的 VDD／VSS 检查；这些最终检查尚未完成，不能由布局前结果替代。

`python3 physical/index_arithmetic_experiments.py` 从独立实验目录更新两份算术实验索引，核对数值检查、映射网表、CTS 检查点及各步骤日志的身份；失败步骤和待完成时序分别列出。索引中的功能通过和 CTS 测量不认证目标频率。

`physical/experiment_mapping.py --source <已验证的 FP64 物理目录> --label <新标签>` 保持原 RTL 及寄存边界，试验 `choices` 组合映射策略；搜索使用独立进程组并限制为 600 秒，不把寄存器交给 ABC 重定时。实际 ABC 策略及哈希记录在新构建中，CTS 后还须通过映射网表的参考向量检查。新生产物理构建另外保存启动时加载的 `physical/run.py` 源码副本，后续编辑不改变其执行身份。

`physical/continue_system.py` 从已认证的完整系统供电检查点继续布局及双时钟 CTS，保留综合网表原宏身份与实际物理宏身份，并设置运行上限。同步器的非激活复位／置位输入涉及常量驱动复制，`physical/protected_ties.tcl` 仅在此操作期间释放保护，随后验证所有正常信号连接和常量逻辑值，恢复保护；实际工具回归也覆盖异常退出时的保护恢复。物理工具回归现为 35 项通过。最终报告必须具有完整供电连通性证据才能宣告物理闭合；独立算术组件当前的时序诊断尚不包含该供电验收。

`python3 physical/snapshot_migration.py` 可更新源码快照、`change_inventory.json` 和 `changes.patch`。它先校验 M0 压缩包及所有原始文件哈希，再使用确定性的归档元数据保存当前源码；既有归档保留。`implementation/source_manifest.json` 只是当前源码索引，实验目录和历史功能结果仍保留各自的执行输入与身份。随后运行 `python3 physical/report_migration.py` 更新验收矩阵；该入口同时核对物理工具测试日志及其实际测试源码，拒绝静默复用不匹配的工具测试结果。

原生 Xyce 已构建为本地工具，CMG107 源码与 Berkeley 模型保持同一固定提交。首轮 `.temp` 指令被该工具忽略的报告已标为无效，并保留修正前原始 JSON。修正为 `.options device temp=` 后，从每个输出采样的 `{TEMP}` 确认实际 25℃／100℃／0℃；`sram_xyce_cell2/verification.json` 中的 24 个检查全部通过。原始 ASAP7 EOTACC=1e-10 在 Xyce 边界比较中产生约一 ULP 的范围警告，参数及方程均未修改。列级 UIC 首轮达到 180 秒上限，尚未完成读写，不能用于三角或宏频率验收。

`physical/check_sram_read_path.py` 进一步保留实际 6T 位单元、完整写驱动、八路列选择／预充电、感放、输出锁存与三态驱动，共 100 个实际晶体管。八对外部位线各使用 10fF 负载，输出为 2fF，理想控制信号按 2ns 操作间隔驱动。TT／SS／FF 的写 1、保持、读回 1、写 0、保持、读回 0 共 18 个检查全部通过，输出温度逐点验证；证据为 `sram_read_path/verification.json`。该结果没有行控制器、完整阵列负载或版图寄生，仍不认证整块宏的 500MHz。

整数进位保存副本使用展开内部总线后的实际工具输入完成 CTS，setup／hold WNS 为 **−0.543591499ns／−0.244812176ns**；同一工具输入通过连续、随机停顿两种模式各 10000 个参考向量，算术延迟严格保持两周期。最差路径为有效状态经全局算术使能的长缓冲链，后续 `--local-enable` 复制已有的两个有效位到各寄存器组旁，并用全格式回归检查使能一致性。Baugh-Wooley 稀疏校正副本与上述控制副本分别保留，未接入主设计。新的 FP64 `predecoded_chunk` 把每个分块的两种前导位及 sticky 前缀提前到原第一级计算；六组小位宽形式检查及 61788 个完整 FP64 向量通过，物理评估另行记录。

局部使能副本保留 86 个实际控制实例，全部格式、舍入、PE 检查及映射后两种模式各 10000 个 I64 向量通过；CTS setup／hold WNS 为 −0.482171208ns／−0.236749351ns，仍未闭合 1ns。Baugh-Wooley 副本为 −0.603919804ns／−0.250740647ns。预解码 FP64 为 −1.249851704ns／−0.098115973ns；四路恒定操作的 FP64 为 −1.286347032ns／−0.237375408ns，并通过 65788 个参考向量（含 4000 个默认操作码）。这些结果保持独立，未采用为主设计。

完整系统的 `system_cts_eco3` 在电气修复达到 3600 秒上限，保留全局布局检查点；续跑入口支持认证的 `--placement-from`。`system_cts_eco4` 实测触发受保护同步器时钟重连被拒绝，`physical/protected_clocks.tcl` 因此仅在 CTS 期间释放保护，之后验证每个受保护实例的单元、数据连接、时钟源及反相次数并恢复保护。实际双时钟夹具验证了正常操作及异常恢复，完整系统 `system_cts_eco5` 也核对了全部 152 个受保护实例。该轮生成了合法布局、双时钟 CTS 的 ODB 和网表，但整轮诊断在 1800 秒上限中止，布局后供电检查尚未完成，不能据此验收。

`--skip-electrical-repair` 仅用于获取分阶段 CTS 诊断；最终电气修复仍必选。未修复复位扇出导致库表外推，系统恢复／移除时序出现严重失败，该数值不作为可实现频率。`physical/reset_repair.tcl` 和 `physical/repair_system_resets.py` 使用锁定工具的逐网修复入口，保持复位根和极性、受保护实例及其数据／时钟连接；32 个实际受保护触发器的回归已覆盖插入缓冲、单元调整和异常恢复。物理工具回归共 35 项通过。该系统修复的 STA 仍属于布局 RC 估计，最终布线提取、电气与供电验收保持独立。


完整系统的复位逐网修复 `system_reset_repair2` 已完成：实际选择两个同时含复位脚和普通逻辑负载的网络（269／571 个输入负载），共插入 183 个缓冲器，并核对所有负载的逻辑根、极性及受保护实例的数据／时钟连接。目标时钟保持 1ns／2ns；单个存储域恢复路径从 −69.407539368ns 改善到约 −7.65ns，仍未通过。总体 setup／hold WNS 为 −71035.929687500ns／−1.924145103ns；最差总体路径来自其他大扇出控制网络，库表外推仍严重，不能作为可实现频率。`--controls` 从认证网表统计候选负载，再按 OpenDB 实际输入数逐网修复；普通数据负载、时钟排除、保护恢复均由实际工具夹具覆盖，物理回归现为 35 项通过。

原生 SRAM 单列诊断进一步保留 `colgrp_sram_6t122` 的完整 1024 个 6T 位单元、dummy 单元及 I/O 外围，共 6286 个晶体管。`sram_read_array_tt3/verification.json` 的 TT 六项写入／保持／读回检查全部通过，操作间隔为 2ns，最大积分步长为 20ps，记录实际采样时间；此前脚本节点名错误、1ps 步长达到上限的记录分别保留。该实验仅检查一个地址、使用理想控制源、没有行驱动器和版图寄生。后续三角诊断另增加其余 1023 个位单元的保持检查，结果按实际完成状态进入验收矩阵；不能据此宣告整块宏 500MHz 合格。

`physical/route_arithmetic.py` 从认证的两级组件 CTS 检查点，在独立目录继续 setup／hold 修复、布线和提取，保留原始数值证据、映射与检查点身份，并限制进程组运行时间。当前 INT64 局部使能副本进入该流程。`physical/check_int64.py --routed` 将布线后实际网表与同一份提取时序报告绑定，再做连续／随机停顿的零延迟数值检查；该检查不能替代物理时序验收。


`sram_read_array_pvt/verification.json` 现已完成 TT 0.70V／25℃、SS 0.63V／100℃、FF 0.77V／0℃，每个输出样本均核对实际温度。完整单列的 18 项选中单元读写检查、18 项覆盖其余 1023 个单元的保持检查全部通过，SPICE 设备方程与参数未改变。初值采用交替存储状态，只用于瞬态初始化，没有新增维持电压源；这也不代表 SRAM 复位清零。该证据不含其余 63 列、真实行控制器／驱动或版图 RC，整块宏的三角 Liberty／DRC／LVS／PEX 和 500MHz 资格仍未完成。

FP64 指数选择简化副本 `compact_exponent_fast` 用单个前缀加法器合成最终指数，保持原两级切分；65788 个独立参考向量通过，CTS setup／hold WNS 为 −1.305695057ns／−0.148118660ns，未改善最佳结果，未合入主设计。映射后的补充数值检查按实际完成状态记录。

首轮大控制网修复曾插入 1011 个缓冲器，但连接核对仍反复遍历同一共享源网络，未完成验证，已保留为失败诊断。逻辑根核对现对每次独立连接快照使用缓存，修复前后分别建立；实际工具回归刻意在修复桩中改接源信号，确认新快照会拒绝错误逻辑，并恢复保护。`system_control_repair2` 改为四个主要网络的有界诊断，每个网络通过核对后原子保存连接检查点；该检查点尚未合法化或完成供电、布线验收。


INT64 首轮完整续跑达到 1800 秒上限，未完成布线提取，失败记录保留；已有 14815 个 hold 缓冲器及合法化后的 `post_initial_hold.odb`，通过 `initial_hold_checkpoint.json` 绑定 RTL、映射、SDC、执行驱动及父数值证据。`physical/route_arithmetic.py --resume-initial-hold` 与生产入口同名选项可从这一实际阶段开始；它核对原始证据和复制后数据库、SDC 的哈希，拒绝变更时钟树或把该阶段当作 CTS 验收。新续跑只扩大工具运行上限，算术仍为两级、核心仍为 1ns，不能把布线前 hold 裕量当作最终通过。


控制器所用的 x1／x2 DFF 原生晶体管电路还单独完成三角检查：Liberty 明确定义上升沿采样、`QN=!D`。`sram_controller_flops/verification.json` 在 2ns 时钟、2fF 输出负载下通过 48 项边沿／保持／极性检查，实际温度逐点确认；48 个晶体管的参数与方程未改变。该检查排除了所测独立触发器的极性模型不匹配，仍不证明完整自定时控制器与阵列的组合启动或整块宏频率。


`--resume-global-route` 从已认证的 `post_global_route.odb` 和同一份布线引导继续详细布线及提取，保留原始两级、1ns 约束；工具运行上限与芯片时钟周期分别记录。实际 CLI 检查已确认：改动时钟约束、数据库或布线引导都会在生成新实现目录之前被拒绝，见 `arithmetic_resume_guard5/verification.json`；原布线前 hold 检查点的两项拒绝检查另见 `arithmetic_resume_guard4/verification.json`。生产工具回归现为 35 项通过，拒绝检查单独计数。

系统控制网第二轮追加 16 个网络，均完成逻辑根／极性核对并保存连接检查点；整轮在全局时序报告阶段达到上限，保留失败证据，不算整轮通过。`--analyze-only` 可对保存的实际合法化数据库单独计时，不重做网络修复，也不改变放置或连接；此入口不执行电气或供电验收。

完整 PE 局部控制副本的首轮导入触发 OpenROAD 分配器错误，失败目录保留；展开同一映射网表内部总线后的独立重试仍按 1ns 进行。另有整数二进制转换副本，保留进位保存乘法和局部使能，已通过全格式 ALU 及 PE／状态／启动间隔检查，物理结果与舍入专项按实际完成情况记录；未合入冻结主设计。


`physical/finish_migration_jobs.py` 把当前两项整数物理任务接入有上限的本地收尾流程：物理生产者结束后，对实际布线网表或映射网表执行两种运行模式的数值检查，再更新实验索引、源码快照和验收矩阵；失败和超时保留，不自动合入实验 RTL。该流程使用独立锁和进程组，状态位于 `artifacts/migration_asap7_f2/job_completion/state.json`，应用重连不需要重新启动正在运行的检查。

控制网修复后的不可变数据库已完成单独 STA，setup／hold WNS 为 **−4041.216064453ns／−1.924145103ns**，仍未达到目标，且存在严重库表外推；`system_control_timing3/verification.json` 仅证明该次时序分析完成，不包含电气、供电或布线验收。原始异步复位到复位同步器异步脚的检查边界还需专门审计；任何后续精确例外须有结构及实际工具依据，级间数据检查继续保留。


详细布线与寄生提取已拆成独立的可恢复阶段。上一轮 INT64 实际路由器已报告 0 个 DRC 违规，但 OpenRCX 仅推进到 58% 就触发 7200 秒整轮上限，因此该轮没有提取后 STA 或硬件频率结果。新版 `--detailed-route-only` 在提取前原子保存 `post_detailed_route.odb`、布线网表及来源清单；`--resume-detailed-route` 验证数据库、网表、SDC、原始 RTL／映射和父数值证据后，仅继续提取与 STA。真实双时钟小电路已跨两个工具进程完成这两个阶段，1ns／2ns 约束保留，且未重复详细布线；物理工具回归现为 35 项通过。变更时钟、ODB 或全局布线引导的五项实际 CLI 拒绝检查已用当前脚本重跑，旧版本证据保留。

`physical/finish_arithmetic_route.py` 对 INT64 的详细布线、提取和最终布线网表数值检查分别计时，状态位于 `int64_route_checkpointed/state.json`。布线和提取各有 7200 秒上限，阶段之间通过实际检查点传递，不改变两级流水或 1ns 时钟。每步成功只代表对应诊断完成；最终核心／系统频率仍需完整验收。

原二进制整数转换副本的映射后连续／随机停顿各 10000 个向量及舍入专项均已通过；其 CTS setup／hold 为 −1.105575919ns／−0.193932325ns，面积 17716µm²，未优于局部使能副本。新增 `int64_retimed_binary2` 在原第一寄存器边界准备移位、指数与 sticky 前缀，仍保留两级；已通过 11 格式 74052 个 ALU 向量、舍入、PE／状态／启动间隔检查，CTS 和映射后验证按实际结果记录。首轮 Chisel 位宽展开失败独立保留，未改变冻结主设计。

`system_control_repair4` 继续从前一轮已核对且合法化的系统数据库修复最多 32 个剩余大控制网，保持核心 1ns、存储 2ns。每个完成网络保存连接检查点；运行中结果不算电气、供电或布线验收通过。


`physical/audit_system_reset_boundary.py` 新增对真实系统数据库的只读复位审计：从门级驱动结构检查三组、六个异步复位脚的完整四项布尔输入组合，核对两级数据链、时钟来源并生成级间 setup／hold 报告。此入口不插入任何时序例外，也不改变源 SDC。实际工具夹具已确认错误的 AND 合并、断开的级间数据和错接的时钟均会被拒绝；工具回归现为 35 项通过。完整系统诊断记录为 `system_reset_boundary1`，其完成状态按实际证据进入报告。


收尾流程支持绑定具体生产者 PID 及启动身份，等待该生产者自己的映射数值检查结束，避免重复启动同一检查。若包装日志在取哈希后恰好追加了自身 JSON 记录，只允许校验“原哈希覆盖的前缀 + 完全相同的记录”后归档原记录并补全日志哈希；原结果正文、额外尾部或运行状态不得修改。三项专项检查均通过，物理工具回归共 35 项。新候选收尾状态位于 `job_completion_retimed/state.json`；其中历史路由超时与当前分阶段路由 `int64_route_checkpointed` 分开记录。


历史 `system_reset_boundary1/verification.json` 完成过三组同步器的组合复位锥、时钟来源、两级连接及级间时序检查。**该次审计没有模拟 Liberty 的触发器行为，其 RESETN／SETN 预期随后证实接反，因此不能证明复位功能正确。** 原报告及级间时序数据保留，旧映射系统全部保持功能未合格状态；以文末的新映射验证及全实例修正证据为准。


`int64_retimed_binary2` 已完成 CTS：setup／hold WNS 为 **−0.967234790ns／−0.218623802ns**，面积 23070µm²。它通过了上述全格式、舍入及 PE 检查，但仍未达到 1ns，也未优于既有局部使能候选，因此未合入主设计。实际映射网表的连续／停顿数值验证由生产者收尾，结果单独绑定该候选 RTL 和网表身份。


INT64 已解决布线网表命名与 SPEF 读取错误。`physical/recover_routed_export.py` 仅将内部网络改为唯一名称，保留顶层端口；完整数据库实测改名／恢复后的 DEF 字节一致，单元、连接、布线对象和保护状态快照一致，共改名 314250 个内部网络。原始错误文件仍保留，原先 +0.0167ns 的 setup 裕量不作为验收证据。

`physical/parasitics.py` 现在核对显式 SPEF 读取区间、解析诊断、全部网络记录和驱动注释覆盖。锁定工具会在 SPEF 语法错误后返回成功，实际工具回归已重现并拒绝这一情况。对新导出数据出现的 32301 个未注释驱动，`physical/audit_parasitic_drivers.py` 在同一认证数据库中逐项证明均为无接收负载、无布线的端点；所有有负载或未知端点仍会被拒绝。此分类按数据库、SPEF、逐项表格和日志哈希绑定，没有新增时序例外。

`safe_names1/check_routed/verification.json` 的实际布线网表已通过连续／随机停顿各 10000 个参考向量，算术延迟仍严格两周期。`physical/recheck_routed_sta.py` 复用同一布线和 SPEF 重新完成注释覆盖与 STA；`safe_names_qualified1/recheck_verification.json` 确认核心 1ns 下 setup **−0.193053693ns**、hold **+0.038750034ns**、TNS **−108.466888428ns**。这是有效的提取时序失败证据；没有认定 INT64 或整个系统达标。当前关键路径从 `io_in_bits_op[2]` 进入算术控制逻辑。独立 `local_enable_clustered64` 试验在同一映射、两级与 1ns／相同 I/O 约束下评估时钟树和几何布局。

系统控制网候选统计已修复漏计单元声明行首个引脚的问题，并覆盖同一行多个端口、总线位和时钟排除。实际旧导出网表中现能识别 16276 个时钟网络；剩余最大信号网络约有 38553 个端口引用，均仍由 OpenDB 核对实际输入负载后修复。`system_control_repair5` 从上一轮认证数据库继续修复最多 32 个网络，每个网络继续核对逻辑根、极性和受保护连接。原始计数及失败记录保留。

当前工具回归 **44 项通过**；改动时钟、数据库及全局引导／详细网表的实际 CLI 拒绝检查共 **8 项通过**，见 `arithmetic_resume_guard8`、`9`、`10`。主设计 174 文件的冻结身份不变；新增诊断均绑定独立实现。完整系统、8×8 物理实现和 SRAM 内部 DRC／LVS／PEX、匹配 PVT 表征仍未验收。

INT64 隔离组件发现原映射仍保留不影响任何输出的浮点逻辑，旧布线最差路径的终点位于该不可观察逻辑中。因此，不能用旧组件 WNS 排名判断整数乘法结构的优劣。新增 `physical/cleanup_mapped_component.py` 只执行映射网表清理；`physical/observable_cone.py` 从全部输出反向遍历，包括时钟、复位及寄存器反馈，逐一核对保留单元类型、参数、连接和初始化状态，拒绝对可观察逻辑的任何改接或重定时。另一次通用 Yosys 等价归纳检查在 600 秒超时，单独保留为未通过，不能当成通用 SAT 证明。

`observable_cleanup1` 的顶层单元数从 135158 降到 26391，映射网表已通过连续／随机停顿各 10000 个独立参考向量，延迟仍为两周期。保持 1ns 和原 I/O 预算的新 CTS 副本 `observable_cleanup1_cts` 面积约 4821µm²，setup／hold WNS 为 **−0.413584054ns／−0.077897638ns**；实际最差路径已落在整数乘法第一阶段。其详细布线、规范化导出、寄生覆盖核对和布线后数值检查由 `int64_observable_pipeline1/state.json` 分阶段记录。Baugh 乘法副本也单独清理并重新验证，两者均未合入冻结主设计。

`system_control_repair5` 的原监督进程退出状态未能恢复，故其 `passed` 保持 false；现存工具日志显示 32 个网络分别完成逻辑核对，合法化数据库和 STA 均已保存，setup／hold WNS 为 **−405.770172119ns／−1.924145103ns**，仍有严重电气问题。恢复记录明确标注退出状态未知，不能改写成正常退出。`system_control_repair6` 从这份认证检查点继续处理最多 32 个网络。当前工具回归为 **48 项通过**，新增四项清理证书检查覆盖死寄存器反馈删除及可观察数据、寄存器、初值的错误变更拒绝；实际 CLI 拒绝检查仍为上述 8 项。

Baugh 副本 `baugh_observable_cleanup1` 清理后，CTS setup／hold WNS 为 **−0.079412147ns／−0.073115505ns**，面积约 5378µm²；实际标量化导入网表已另行通过连续／随机停顿各 10000 个向量，先前未标量化的检查完整归档。该候选进入 `physical/finish_arithmetic_route.py --from-cts --canonical-export` 的分阶段流程，分别保存详细布线、规范化导出、寄生驱动分类、布线数值和重新 STA 的结果。流程只有在最终寄生注释完整时才可完成；完成诊断仍不代表时序、供电和完整系统达标。

`physical/experiment_product_cut.py` 另试验原第一寄存器边界保留四项部分积，第二阶段完成压缩和累加，仍是两级。`four_term_cut1` 已通过 W=2、3、4、7 的全部输入 SAT 检查，以及 W=8 的全部 65536 组输入仿真，每个位宽均检查两种结果宽度。完整 ALU／PE 功能、实际映射和 CTS 分别有运行上限并记录在 `int64_four_term_cut1/state.json`；小位宽证明不等于完整 64 位证明，未完成的功能或时序不计通过。

系统第六轮 32 网修复正常退出，布局估计 setup／hold WNS 为 **−405.554351807ns／−1.924145103ns**。实际报告中，核心瓶颈反相器 `_2454204_` 的输出在网表中有 1037 个端口引用，实际输入负载由 OpenDB 复核，约 387.7ns 的单元延迟属于严重电气外推。`system_control_critical7` 因此按真实核心／存储路径选择五个网络，而不继续只按最大扇出排序；其选择来自认证网表，并继续执行 OpenDB 负载检查、逻辑根／极性及保护状态核对。SDC、两域周期和正常数据检查均未放宽。

`baugh_observable_cleanup1_qualified` 已完成实际布线网表连续／随机停顿各 10000 个向量检查和严格寄生注释核对。提取后 setup／hold WNS 为 **−0.056738172ns／+0.042377118ns**，TNS 为 **−2.872718573ns**；仍有电气问题且尚未验收供电网，不能认证 1GHz。最差路径经过 `arithmetic.ce` 控制网络，新增 `baugh_local_enable1` 因此组合已验证的 Baugh 乘法与局部两位有效状态副本，保持原两级。各项完整功能、实际映射与 CTS 证据仍分别保存。

四项部分积首轮门级检查因 Yosys 导出的单行超过 Verilator 40000 个预处理 token 而未编译，尚未执行数值比较。`physical/check_int64.py` 现对仅综合结果生成私有的空白排版副本，并保存原始映射哈希、实际仿真文件哈希及格式化器身份。`physical/experiment_product_cut.py --resume-source` 只恢复具有完整功能／映射证据的这类已记录编译失败，原结果不覆盖。`int64_four_term_recovery1` 已通过恢复后连续／停顿各 10000 个门级向量、可观察逻辑清理证书和清理后数值检查；CTS setup／hold 为 **−0.343604714ns／−0.094837166ns**，仍未达标，实际标量化导入检查继续单独绑定。

系统扇出统计进一步覆盖 `\\tile._waves_rdata [71]` 这样的转义层次名称加独立位选择。首次八轮定点诊断被旧统计拒绝，未进入数据库修改；失败记录保留。`system_control_critical8_fixed` 使用修正后的统计选择六个实际路径网络，并继续由 OpenDB 复核输入负载及逻辑。CTS 祖先验证允许更长的有界 ECO 链，仍逐级校验身份并拒绝环。更新后的完整工具回归 **48 项通过**，实际日志位于 `physical_tool_tests_bus_select.log`；原日志保留。

`physical/parallel_exponents.py` 提供独立的乘积指数及指数差计算，避免指数差串接乘积指数的进位链。所有压缩和前缀相加均在原第一阶段内完成，没有寄存器。`parallel_exponents_fast/exponent_native_widths1` 已对内部宽度／偏置 **6／7、7／15、10／127、13／1023** 的全部指数位与校正位输入通过 SAT 等价检查。该实验只改变独立 FP64 组件副本；完整浮点数值、异常、停顿和物理时序按实际完成状态验收，不能由指数子电路证明替代。

并行指数副本已通过 **57788 个 RTL 参考向量**，含 4000 个默认操作码别名向量、异常和抵消情况，进入两级／1ns 综合与 CTS。四项部分积恢复流程的实际标量化导入网表另通过连续／停顿各 10000 个向量，整条恢复诊断已完成；其上述负 setup／hold 裕量仍保留，未推进为已达标实现。

`physical/repair_system_hotpaths.py` 从认证布局 STA 的单元输出延迟选取待修复信号，解析实际网表中的输出绑定，并排除识别出的时钟网络；最终仍由原有 OpenDB／逻辑根保护入口决定是否允许修复。它已用第七轮真实报告验证，能选中先前遗漏的层次总线位。`system_hotpath_sequence1` 最多续跑三轮、每轮六个网络，每轮工具有 3600 秒上限；2ns 的异常单元延迟选网门限不改变核心 1ns／存储 2ns 的 SDC。每轮单独保存来源、选网依据、日志及核对结果，未找到极端延迟也不代表 setup／hold、电气、供电或完整系统验收通过。

`baugh_local_enable1` 已完成全 11 格式 74052 个 RTL 向量、舍入／PE／状态／启动间隔检查，以及实际标量化导入网表的连续／停顿各 10000 个向量。保持相同 1ns／I/O 预算，CTS setup 为 **+0.020310421ns**，hold 为 **−0.067333043ns**，面积约 5461µm²。`int64_baugh_local_route1` 从认证 CTS 和实际映射数值证据继续 hold 修复、布线、规范化导出、寄生分类及布线后验证；正 setup 裕量不能代替最终时序、电气和供电验收。

第八轮六网修复正常退出，布局估计 setup／hold 为 **−212.839019775ns／−1.924145103ns**，仍未达标。后续自动选网轮次继续保存各自的合法化结果和时序诊断，不降低周期，也未修改异步边界策略。

首个并行指数副本通过上述 RTL 检查和 4000 个实际映射向量，但 CTS setup／hold 为 **−1.408797503ns／−0.046434566ns**，没有改善结果，未采用。`effective_exponents_fast` 进一步将 A／B 的零编码直接转为有效指数，减少乘积／指数差压缩树的项数；其 SAT 证明明确要求 A／B 零标志由指数值派生，C 校正位仍任意，四种原生内部宽度均通过。该副本也通过 57788 个 RTL 向量，后续映射和物理结果独立记录。指数转换入口会拒绝不符合所证明校正规则的父实现。

异步映射核对发现实际功能缺陷：锁定 Liberty 的 `DFFASRHQNx1_ASAP7_75t_R` 在时钟沿产生 `QN=!D`，而旧自动映射恢复逻辑 Q 后仍将 RESETN／SETN 反接。`reset_liberty_probe1` 用实际单元模型重现了外部复位置位后与原 RTL 不一致。旧系统布局／CTS／控制优化结果均不再作为功能合格实现；`system_hotpath_sequence1` 第二轮已定点停止，失败及原始网表保留。冻结主 RTL 的 57 项功能检查和独立同步算术组件不受此缺陷影响。

新增 `physical/Asap7AsyncFlops.v` 显式映射四种正沿单异步复位类型，在自动 `dfflibmap` 前展开使能并完成映射；不支持的异步类型直接拒绝。`physical/check_async_mapping.py` 从实际 Liberty 生成模型，对复位到 0／1、正／负复位极性、使能及两级复位释放进行 **5010 次事件比较，TT／SS／FF 均通过**。综合入口绑定映射源码、验证与实际库哈希；复用入口及后续系统物理诊断拒绝未经验证的旧异步映射。生产综合函数还单独通过相同事件比较，相关测试包含实际 OpenROAD 引脚修正、保护状态恢复及双异步控制拒绝。

`physical/correct_system_async.py` 在旧系统认证数据库的独立副本中修正全部 **4123 个**单异步复位触发器，先验证另一路异步脚恒为无效，再交换两个脚。修正必须逐项保持其余连接、全部单元类型、位置、朝向、放置与保护状态，核对三组复位同步器，并保留源 SDC 语义；原数据库不覆盖。该结果只用于继续系统功能／物理验证，不能替代完整系统门级功能和布线验收。

`baugh_local_enable1_qualified` 已完成详细布线、规范化导出、严格寄生覆盖分类及连续／停顿各 10000 个布线网表向量。提取 setup／hold 为 **−0.058678064ns／+0.044165701ns**，TNS **−2.081262827ns**；setup、电气和供电仍未全部通过，没有认证 1GHz。`effective_exponents_choices` 保留相同 RTL 和两级，4000 个实际映射向量通过，CTS setup／hold 为 **−1.027604103ns／−0.004125020ns**，仍未优于此前最佳 FP64 setup，未采用。

`physical/check_system_gates.py` 直接编译修正后实际系统网表、Liberty 生成的标准单元模型及匹配的 SRAM 数字模型，复用认证测试台；它不把门级网表重新标为原 RTL 构建。计划重放四种时钟相位的全 11 格式、29 描述符数据，并检查 core／memory／双域在读、计算、写回期间的九种复位恢复。每个结果绑定实际网表、测试台、参考输出、库与运行时节拍；编译／仿真均有独立上限，未完成不计通过。首次导入失败保留，修正导出后的后续状态在 `system_gates_corrected3/state.json`，监督状态在 `system_gate_followup1/state.json`。

修正后的三组复位同步器已完成实际级间 setup／hold 报告及组合复位锥核对，源 SDC 未改变、未增加时序例外，见 `system_reset_boundary_corrected1`。全局 STA 完成，布局估计 setup／hold 仍为 **−200.674789429ns／−1.924145103ns**；这是失败诊断，不能推导可实现频率。`system_hotpaths_corrected1` 只从已认证的修正数据库与这份实际新 STA 选网，继续有界控制网优化。

首次系统门级编译暴露两个输出总线位的 OpenDB／Verilog 导出名称冲突：插入输出缓冲后，内部网仍以另一顶层位名命名，导出产生 `MULTIDRIVEN`。原编译已停止并保留，不能作为数值通过证据。`physical/canonical_port_aliases.tcl` 只重命名实际遮蔽其他 BTerm 的网络，保持顶层端口名；`physical/export_system_ports.py` 核对全部连接／位置／保护状态及恢复原名后的完整 DEF 字节一致。新门级入口拒绝多驱动及非预期的缺失引脚；时钟负载单元本来悬空的输出单独记录。原始 ODB 与全部失败导出均保留。
