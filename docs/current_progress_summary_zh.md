# 当前进展汇报（2026-05-20）

## 当前目标

为 Jamba2 风格混合序列模型（Mamba + Attention + MoE 交替层）设计一套统一资源复用硬件原型，
研究"用一个 MAC 单元服务多种算子"的资源-延迟权衡，并通过 Chisel 生成 SystemVerilog 做结构代理分析。

---

## 已完成模块

### 计算核心

| 模块 | 功能 | 测试 |
|---|---|---|
| `MacLane` / `MacLaneMixed` | 原子 MAC 单元，支持 `zeroSkip` 零跳过 | ✅ |
| `SerialSharedLinear4` | 16 周期串行 4×4 矩阵乘 | ✅ |
| `ConfigurableSerialLinear4` | M8-O 可配置 1/2/4 MAC lane 投影引擎，含 reduction tree | ✅ |
| `SerialCausalConvMini` | 串行因果卷积，支持状态存取 | ✅ |
| `SerialSelectiveScanMini` | 串行选择性扫描（SSM），支持状态存取 | ✅ |
| `UnifiedProjectionScheduler4` | 单 slot-table FSM，调度全部 10 个投影 | ✅ |
| `UnifiedJamba2MiniLayer` | 统一层（Mamba/Attention/MoE 三路），含状态存取接口 | ✅ |

### 内存子系统

| 模块 | 功能 |
|---|---|
| `LayeredWeightStoreMini` | 分层权重寄存器堆，按 layer-stride 地址解码 |
| `WeightAddressGenMini` | 权重地址生成器（含 MoE expert 字段） |
| `SequentialWeightLoaderMini` | BRAM 风格顺序加载路径（序列化写入） |

### Tile 级调度

| 模块 | 功能 |
|---|---|
| `UnifiedJamba2MiniTileScheduler` | L 实例调度器（每层一个物理 layer） |
| `UnifiedJamba2MiniFullTile` | 带权重加载的完整 tile，作为参考基准 |
| `SinglePhysicalLayerTile` | **核心贡献**：单物理层服务 L 逻辑层，含状态虚拟化 |

---

## 核心创新点

### 1. 四层硬件映射框架

对同一 Jamba2 Mini 层实现并对比四种资源-延迟层级：

| 层级 | 策略 | 结构乘法代理 | 延迟 |
|---|---|---|---|
| Baseline | 每个投影独立并行 MAC | 96 | 1 周期/token |
| SharedFabric | 每个算子共享 1 个 MAC lane | 69 | ~16 周期/投影 |
| SemanticSerial | 每种 mixer 类型共享 1 个 MAC lane | 42 | ~556 周期/token |
| **UnifiedSerial** | **全部 10 个投影共享 1 个 MAC lane** | **50** | ~556 周期/token |

UnifiedSerial 相比 Baseline 减少 **48%** 结构乘法代理。

### 2. SinglePhysicalLayerTile（M7-A+B）

用 1 个物理 `UnifiedJamba2MiniLayer` 替代 L 实例调度器，tile 级 instance-weighted 资源代理从 O(L) 降为 O(1)：

| 配置 | L 实例（FullTile） | 单物理层（SinglePhysicalTile） |
|---|---|---|
| 2 层 Context8 | 184 | **92** |
| 4 层 Context8 | 368 | **92** |
| 8 层 Context16 | 1,248 | **156** |

**M7-B 状态虚拟化**：每个逻辑层独立保存 SSM hidden state、causal-conv history、KV cache，通过 `restoreState` FSM 相位在每次 `launchLayer` 前注入。功能正确性通过 2-token 和 3-layer（Mamba/Mamba/Attention）等价性测试验证。

### 3. 量化分析

INT4 / INT6 / INT8 精度扫描：结构乘法代理**不随精度变化**（乘法器数量不变，只有位宽变化）；寄存器总位数随精度近似线性缩放（INT4 约为 INT8 的一半：6,104 vs 12,168 位）。

### 4. 零跳过稀疏化

`zeroSkip` 参数为 `MacLane` 加入比较器 + MUX，当任一操作数为零时跳过乘法运算。结构乘法代理不变（乘法器仍在 RTL 中），收益是稀疏激活下的动态功耗降低。

### 5. M8-O 投影 MAC 并行度优化

在不改变 `SinglePhysicalLayerTile` 和统一 slot-table 调度的前提下，新增
`projectionMacLanes = 1 / 2 / 4`。Context8 下 instance-weighted 乘法代理从
92 → 94 → 98 小幅增加，但投影解析延迟从 16 → 8 → 4 周期下降，形成资源-延迟
Pareto 曲线。其中 `Mac2` 是较好的折中点。

---

## 关键实验数据

| 实验 | 核心结果 |
|---|---|
| 四层资源对比 | 乘法代理：96 → 69 → 42 → 50 |
| 量化扫描（INT4–INT8） | 乘法代理恒定；寄存器位数：6,104 → 12,168 |
| Context 长度扫描 | 乘法代理近似线性增长：50（ctx4）→ 82（ctx8）→ 146（ctx16） |
| 层数扫描（SinglePhysicalTile） | instance-weighted 代理恒定 ~92（Context8），与 L 无关 |
| M8-O 投影并行度扫描 | Context8 Mac1/Mac2/Mac4：instance-weighted 92/94/98；投影周期 16/8/4 |
| 延迟预算 | 第 4 层单层 ~143 周期；4 层 tile ~556 周期 |
| 测试规模 | **219 个 Chisel 测试 + 28 个 Python 测试，全部通过** |

---

## 当前限制

1. **无综合后结果**：所有资源数据均为结构代理（生成 SystemVerilog 中 ` * ` 行数），未经 Vivado/Quartus 综合。综合工具可能通过常量传播和资源共享消除部分乘法器。

2. **Mini 参数规模**：lanes=4，4×4 权重矩阵，context length 最大 16。资源趋势有代表性，但具体数字无法直接外推到生产规模（如 4096 维隐状态、32 层）。

3. **近似 Attention**：KV score 归一化使用右移近似代替 softmax，足以做结构和功能验证，但不适用于精度敏感的部署场景。

4. **无真实稀疏数据**：零跳过分析是结构性的，未使用真实模型激活统计来估计实际零跳过率。

---

## 下一步计划

### 近期（论文方向）

- [ ] 补充 Abstract 和 Related Work 引用细节（文献信息待确认）
- [ ] 将全部章节整理为连贯论文草稿，供导师审阅

### 中期（工程方向）

- [x] **M8-O: 投影 MAC 并行度优化** — 可配置 `projectionMacLanes=1/2/4`，形成资源-延迟 Pareto 数据
- [ ] **M8: BRAM 风格权重/状态存储** — 将 register-file 改为 `SyncReadMem`，支持 Vivado BRAM 推断
- [ ] **M9: Verilator lint + 综合就绪清理** — 解决 FIRRTL lowering 产生的综合警告
- [ ] **M10: FPGA 综合 + 板级 demo** — 在 Xilinx Ultrascale+ 上跑 Vivado，获取 LUT/FF/DSP 实测数据

---

## 需要导师确认的问题

1. **结构代理的可信度**：当前以"生成 SV 中 ` * ` 行数"作为 DSP 使用量的预估代理，是否足以支撑论文论点？还是需要提供综合后数据？

2. **参数规模**：lanes=4、4×4 矩阵的 mini 规模是否足以说明资源-延迟趋势？是否需要在更大尺寸（如 lanes=16、16×16）上补充一组数据点？

3. **M7-B 的论文定位**：SinglePhysicalLayerTile 将 instance-weighted 资源代理从 O(L) 降为 O(1)，并通过功能等价测试验证状态虚拟化——这个贡献的分量是否达到章节级别，还是作为系统实现的一个特性描述？

4. **Related Work 覆盖**：§2.8 中列出了 FTRANS、FA-BERT、Mamba、Jamba 等文献，但引用细节（作者列表、会议/期刊、DOI）尚未填写完整，需要确认具体引用格式要求。

5. **下一阶段优先级**：建议的路线是先完成 M8（BRAM 存储）再做 FPGA 综合，还是直接尝试 M9/M10？如果时间有限，是否可以只用结构代理结果发论文？
