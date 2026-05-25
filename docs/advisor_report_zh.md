# 项目进展汇报

> 汇报时间：2026 年 5 月
> 项目：Jamba 2.0 Mini 硬件加速器原型（Chisel 实现）

---

## 一、项目目标（导师指导方针）

> 用 Chisel 写一个 Jamba 2.0 Mini 的硬件加速器，算法模型了解和硬件设计都需要，还是很有挑战性的，从核心算子做，参考一下网上的会议或期刊论文，例如 Mamba 和 Samba、Transformer 等，目标以硬件带动软件理解，就以产出一篇顶会论文为目标，一比一实现，资源的复用是瓶颈，最好能上 FPGA，我们有 FPGA，不同类型算子之间的各种资源复用，先提炼出每层的共性算子和非共性算子，看看复用的硬件资源和数据通路，然后以统一的结构支持这些算子，从而完成整体模型的执行，当然可以从硬件优化角度思考一下算法优化，算法迎合硬件做修改。

**1:1 的含义**：完整支持 Jamba2 Mini 的算法流程和关键模块语义，但硬件实现可以用 mini 规模、量化、稀疏化和资源复用来做优化；不是一开始就完整复刻真实模型的参数规模和部署系统。

---

## 二、论文定位

**工作名称（暂定）**：A Unified Resource-Reuse FPGA Accelerator for Jamba2-Style Hybrid Sequence Models

**目标期刊/会议**：FPGA、FPL、DAC、DATE

**核心主张**：用 Chisel 实现一个 Jamba2 Mini-style 加速器，完整支持 Mamba / Attention / MoE-lite 的 mini 算法流程，提出从算子级共享 fabric 到层级统一调度器的资源复用架构，量化资源-延迟的 Pareto 权衡，以结构性代理分析为主要评估手段（综合前），FPGA 综合验证为下一阶段目标。

---

## 三、拟定论文结构

以下为当前拟定的章节结构，供导师评价和调整。Ch1–Ch7 草稿已完成，待 FPGA 数据补入后修订。

### 第一章 绪论
- 研究背景：LLM 推理的计算与带宽瓶颈；Mamba/SSM 线性复杂度的优势与硬件挑战
- 研究动机：Jamba2 混合架构中多类算子并存，为每类路径单独部署硬件导致资源浪费；Linear 投影是三类路径的公共分母，是统一 MAC 的设计基础
- 主要贡献：算子分类框架、统一执行引擎、多层共享、MAC 并行度 Pareto、量化/稀疏协同

### 第二章 背景知识
- Mamba / 选择性状态空间模型（SSM）
- Transformer / Multi-head Attention
- Jamba2 混合层结构（Mamba:Attention ≈ 7:1，共享 MLP/MoE）
- FPGA 硬件资源（DSP、BRAM）与定点量化基础

### 第三章 算子分析
- Jamba2 每层算子拆解（Mamba 路径 / Attention 路径 / MLP-MoE 路径）
- 共性算子 vs 非共性算子分类（见下表）
- 10 次 Linear 投影是资源复用的核心切入点

### 第四章 统一执行引擎架构
- 整体架构：共享 Projection MAC + 专用算子单元（Conv / Scan / Attention Score）
- `UnifiedProjectionScheduler4`：槽表 FSM 统一调度 10 次投影
- `SinglePhysicalLayerTile`：单物理层 + 状态虚拟化，O(L)→O(1) 实例资源
- MAC 并行度参数化（`projectionMacLanes=1/2/4`）与 Pareto 权衡

### 第五章 算法-硬件协同优化
- INT8 定点量化：权重 INT8 + 累加器 INT32，`narrowToData()` 截断
- ZeroSkip MAC：零激活旁路，动态功耗优化
- 投影顺序优化：B/C 投影提前，减少 Scan 等待

### 第六章 实现与验证
- Chisel 设计方法论与模块层次
- 测试策略：ChiselTest 单元测试 + Python 黄金模型交叉验证
- SystemVerilog 结构代理分析（Mul-proxy）方法说明

### 第七章 评估
- 四层资源对比（Baseline → UnifiedSerial，Mul-proxy 96→50）
- 多层共享收益（instance-weighted proxy，O(L)→O(1)）
- MAC 并行度 Pareto 曲线（资源 vs 延迟）
- 量化精度分析（INT4/6/8 Mul-proxy 不变，寄存器位宽变化）
- *待补充*：FPGA 综合结果（LUT/DSP/BRAM/Fmax）、CPU baseline 加速比

### 第八章 结论与展望
- 贡献总结
- 局限：当前为结构代理分析，无综合后实测数据；参数规模为 mini
- 未来工作：BRAM 权重存储、FPGA 上板、规模扩展

---

## 四、算子分析与设计基础

Jamba2 每层包含 Mamba 路径（SSM + 因果卷积）、Attention 路径和 MLP/MoE 路径。三条路径共有 **10 次 Linear 投影**（Mamba×3 + Attention×4 + MLP/MoE×3），结构完全相同，是统一 MAC 阵列的设计基础。

| 算子 | Mamba | Attention | MLP | 分类 |
|------|:---:|:---:|:---:|------|
| Linear 投影 | ✓×3 | ✓×4 | ✓×3 | **共性**（复用目标） |
| RmsNorm / Residual | ✓ | ✓ | ✓ | 共性（轻量组合逻辑） |
| CausalConv1D | ✓ | — | — | 专用 |
| SelectiveScan（SSM） | ✓ | — | — | 专用 |
| DotProduct + KV Cache | — | ✓ | — | 专用 |
| SiLU + 逐元素乘 | — | — | ✓ | 专用 |

---

## 四、当前进展

### 4.1 四层资源复用框架（资源代理：层级 Mul-proxy）

| 方案 | 描述 | 层级 Mul-proxy |
|------|------|:---:|
| Baseline | 各算子独立实例化，并行计算 | 96 |
| SharedFabric | 每类算子共享一个 MAC 单元 | 69 |
| SemanticSerial | 按语义分组，Mamba/Attention 各一个调度器 | 42 |
| **UnifiedSerial** | 一个槽表 FSM 调度全部 10 次投影 | **50** |

> Mul-proxy = 生成 SystemVerilog 中乘法操作行数，用于综合前结构性资源估算。UnifiedSerial 比 SemanticSerial 略高（50 > 42）是因为 Mamba 和 Attention 两条路径的投影槽位在结构上同时展开。

### 4.2 多层共享：SinglePhysicalLayerTile（M7-A/B）

**问题**：若为每个逻辑层各保留一份物理硬件，实例加权资源随层数 L 线性增长（4 层时为 368）。

**方案**：`SinglePhysicalLayerTile` — 1 个物理 `UnifiedJamba2MiniLayer` 服务 L 个逻辑层，层切换时通过 save/restore FSM 将各层的 SSM 隐层状态、因果卷积历史、KV Cache 保存到状态寄存器文件，再恢复给下一层。

**结果**（实例加权 Mul-proxy）：

| 配置 | 多实例方案 | 单物理层方案 |
|------|:---:|:---:|
| 2 层 Context=8 | 184 | 92 |
| 4 层 Context=8 | 368 | 92 |
| 8 层 Context=16 | 1248 | 156 |

资源代理从 O(L) 降为接近 O(1)，已通过 2-token 等价性验证和 3 层（Mamba/Mamba/Attention）状态隔离测试。

### 4.3 MAC 并行度 Pareto 分析（M8-O）

将投影单元 MAC lane 数设为 1/2/4，评估资源与延迟的权衡（延迟为 FSM 结构推导的解析估算）：

| 配置 | 实例加权 Mul-proxy | Token 估算周期 | 相对 Mac1 |
|------|:---:|:---:|---|
| Mac1（最大复用） | 92 | 564 | 基准 |
| **Mac2（最优权衡）** | **94** | **364** | **资源 +2.2%，延迟 −35%** |
| Mac4（最快） | 98 | 264 | 资源 +6.5%，延迟 −53% |

### 4.4 量化与稀疏

- **INT4/INT6/INT8**：Mul-proxy 不变，寄存器位宽随精度降低约 50%/2-bit
- **ZeroSkip MAC**：零激活时旁路乘法，减少动态功耗（结构已验证）

### 4.5 验证状态

```
Chisel 测试：219 / 219 通过（69 个测试套件）
Python 测试： 28 /  28 通过（黄金模型交叉验证）
SystemVerilog 生成：clean
FPGA 综合：未开始
```

代码已上传 GitHub：https://github.com/ddixx1314-hash/jamba-accelerator

---

## 六、下一步计划

| 里程碑 | 内容 |
|--------|------|
| **M8-BRAM** | 将权重和层状态从寄存器迁移到片上 BRAM（`SyncReadMem`），模拟真实部署，减少 IO 端口数量 |
| **M9** | Verilator lint 清理，统一 reset 约定，综合 ready 预处理 |
| **M10** | Vivado 综合，采集 LUT/FF/DSP/BRAM/Fmax；条件允许时上板验证 |
| **精度实验** | CPU FP32 baseline 延迟测量；INT8 量化误差（cosine similarity）评估 |
| **论文完善** | 当前草稿 Ch1–Ch7 已完成，待 FPGA 数据填入评估章节后修订 |

---

## 七、想请教的问题

1. **综合优先级**：M8-BRAM（片上存储工程化）和 M9/M10（FPGA 综合）哪个更应该先做？还是同步推进？

2. **论文方向建议**：目前的工作围绕"异构算子间 MAC 资源复用"展开，从算子分类到统一调度器到多层共享，整体方向是否符合您的预期？有没有需要调整侧重点、补充或去掉的内容？

3. **规模扩展**：当前原型参数规模很小（lanes=4，4×4 权重矩阵）。在做 FPGA 综合之前，是否需要先把参数规模扩大到更贴近真实 Jamba2 Mini 的水平？

---

## 八、参考文献

| 论文 | 关键点 |
|------|--------|
| *Attention Is All You Need* (2017) | Transformer / Multi-head Attention |
| *Mamba* (2023) | Selective State Space Model，硬件感知选择性扫描 |
| *Jamba* (2024) | Mamba + Transformer 混合层结构 |
| *Jamba 2* (2024) | MoE 融合，分组注意力 |
| *Samba* (ICLR 2025) | Mamba + Sliding Window Attention |
| *S4* (2021) | 结构化 SSM，Mamba 前驱 |
| *FlashAttention* (2022/2023) | IO 感知注意力，资源效率视角 |
