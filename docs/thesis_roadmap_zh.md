# 论文规划：面向 Jamba 2 Mini 的统一资源复用 FPGA 硬件加速器

> 本文档记录论文的整体思路、技术路线、实现策略和评估计划。
> 所有章节均以导师指导为依据，随项目进展持续更新。

---

## 导师原话（核心指导方针）

> 用 Chisel 写一个 Jamba 2.0 Mini 的硬件加速器，算法模型了解和硬件设计都需要，还是很有挑战性的，从核心算子做，参考一下网上的会议或期刊论文，例如 Mamba 和 Samba、Transformer 等，目标以硬件带动软件理解，就以产出一篇顶会论文为目标，一比一实现，资源的复用是瓶颈，最好能上 FPGA，我们有 FPGA，不同类型算子之间的各种资源复用，先提炼出每层的共性算子和非共性算子，看看复用的硬件资源和数据通路，然后以统一的结构支持这些算子，从而完成整体模型的执行，当然可以从硬件优化角度思考一下算法优化，算法迎合硬件做修改。
>
> **1:1 的含义**：完整支持 Jamba2 Mini 的算法流程和关键模块语义，但硬件实现可以用 mini 规模、量化、稀疏化和资源复用来做优化；不是一开始就完整复刻真实模型的参数规模和部署系统。先让自己有个目标和参照上手，然后再试着做优化。

---

## 论文定位

**工作名称（暂定）**：

> A Unified Resource-Reuse FPGA Accelerator for Jamba2-Style Hybrid Sequence Models

**目标期刊/会议**：FPGA、FPL、DAC、DATE（顶会方向）

**核心主张**：

本文用 Chisel 实现一个 Jamba2 Mini-style FPGA accelerator，完整支持 Mamba / Attention / MLP / MoE-lite 的 mini 算法流程，并提出从算子级 shared fabric 到层级 semantic-serial scheduler 的资源复用架构，最终以 `Jamba2MiniAcceleratorTile` 形式完成端到端 token 执行。

**最终交付物层次**：

```
Jamba2MiniAcceleratorTile          ← 论文最终交付物（顶层）
  ├── token input/output interface
  ├── weight load interface / WeightBRAM
  ├── tile-level scheduler (layer sequencing)
  └── UnifiedJamba2MiniLayer       ← 核心执行引擎
        ├── 共享 projection MAC (SerialSharedLinear4)
        ├── SerialCausalConvMini   (专用)
        ├── SerialSelectiveScanMini (专用)
        ├── Attention score unit   (专用)
        └── RmsNormApprox × 2     (组合)
```

`UnifiedJamba2MiniLayer` 是论文最核心的技术贡献（统一调度器 FSM），`AcceleratorTile` 是它的顶层封装，是论文"系统"层面的交付物。

---

## 参考文献（必读）

本地论文缓存位于 [docs/papers/README.md](papers/README.md)，便于离线阅读和后续写引用。

| 论文 | 关键点 |
|------|--------|
| *Attention Is All You Need* (2017) | Transformer 基础，Multi-head Attention |
| *Mamba* (2023) | Selective State Space Model (S6)，硬件感知选择性扫描 |
| *Jamba* (2024) | Mamba + Transformer 混合层结构 |
| *Jamba 2* (2024) | 改进版本，MoE 融合，分组注意力 |
| *Samba* (2024, ICLR 2025) | Mamba + Sliding Window Attention 混合 |
| *S4* (2021) | 结构化 SSM，理解 Mamba 前驱 |
| *FlashAttention* (2022/2023) | IO 感知注意力，资源效率视角 |
| FPGA 加速相关：*A100/TPU 论文*、*Xilinx DSP 优化* | 硬件实现参考 |

---

## 第一章 绪论

### 1.1 研究背景

- 大语言模型（LLM）推理面临两大瓶颈：**计算量**和**内存带宽**
- Transformer 架构注意力复杂度 O(n²)，长序列开销大
- Mamba/SSM 架构线性复杂度，但需要专用硬件支持才能发挥优势
- 混合架构（Jamba2）兼顾两者，但带来了**多算子类型共存**的硬件挑战

### 1.2 研究动机

- 现有加速器大多专一：Flash Attention 针对 Transformer，MambaQuant 针对 SSM
- Jamba2 的混合性意味着：若为每类路径分别部署独立硬件 → 资源浪费巨大
- 核心洞察：**线性投影**是 Mamba、Attention、MLP 三类路径的公共分母，预期在 mini 规模配置下是运算量最大的单类算子（第六章给出实测 MAC 次数分解）
- 问题：**能否用一套统一硬件调度所有算子，同时保持完整语义？**

### 1.3 研究目标

1. 系统分析 Jamba2 每层的**共性算子**与**非共性算子**
2. 设计以**资源复用**为核心的**统一执行引擎**
3. 实现完整 Jamba2 Mini 推理流程（1:1 算法语义，mini 参数规模）
4. 探索**算法-硬件协同优化**：量化、稀疏化配合硬件设计
5. 上板验证（FPGA），采集 LUT/DSP/BRAM/Fmax 数据

### 1.4 主要贡献

1. **算子分类框架**：系统性地对 Jamba2-style 混合层进行共性/非共性算子拆分，建立资源共享映射关系
2. **统一执行引擎**：共享 projection MAC + 专用算子单元，统一调度器 FSM 支持三类路径
3. **三级资源对比**：Baseline / SharedFabric / SemanticSerial，从生成 SV 到 FPGA 综合量化复用收益
4. **算法-硬件协同**：INT8 量化 + 跳零稀疏化，配合 MAC 单元设计
5. **端到端 AcceleratorTile**：Chisel 原型，含 token 接口 + weight 加载 + FPGA 综合结果

---

## 第二章 Jamba2 Mini 模型与算子分析

### 2.1 Jamba2 整体结构

```
Token Embedding
    ↓
┌─────────────────────────┐
│  Jamba2 Layer × N       │  ← 每层选择一种路径
│  ├── Mamba 路径 (多数)   │
│  ├── Attention 路径 (少) │
│  └── 共享 MLP 路径       │
└─────────────────────────┘
    ↓
LM Head (output projection)
```

**关键特点**：
- 每 N 层中约有 1 层是 Attention，其余是 Mamba（Jamba2 比例约 1:7）
- 每层都有 RmsNorm + 残差结构
- MLP 路径每层都执行

### 2.2 各路径算子详解

#### Mamba 路径

```
x → RmsNorm → input_proj(Linear) → CausalConv1D → [分叉]
                B_proj(Linear) ────────────────────→ SelectiveScan → y_mamba
                C_proj(Linear) ────────────────────↗
                A(参数) ─────────────────────────────↗
    x + y_mamba → firstResidual
```

涉及算子：
- `Linear` × 3 (input, B, C 投影)
- `CausalConv1D`（1D 卷积，taps=4）
- `SelectiveScan`（状态空间递推，lanes=4）

#### Attention 路径

```
x → RmsNorm → Q_proj(Linear) → Q
            → K_proj(Linear) → K → 写入 KV Cache
            → V_proj(Linear) → V → 写入 KV Cache
            → score = Q · K^T (DotProduct × contextLength)
            → weight = score >> normShift   (移位近似归一化，无 softmax；完整 softmax 是后续扩展)
            → rawY = Σ weight_i · V_i
            → out_proj(Linear) → y_attention
    x + y_attention → firstResidual
```

涉及算子：
- `Linear` × 4 (Q, K, V, out 投影)
- `DotProduct`（Q·K 打分）
- `WeightedSum`（weight·V 累加）
- `KV Cache`（寄存器阵列）

#### MLP 路径（每层都有）

```
firstResidual → RmsNorm → gate_proj(Linear) → SiLU ─→ ⊗ → down_proj(Linear) → y_mlp
                        → up_proj(Linear) ───────────↗
    firstResidual + y_mlp → y (最终输出)
```

涉及算子：
- `Linear` × 3 (gate, up, down 投影)
- `SiLU`（激活函数，逐元素）
- `Element-wise multiply`

### 2.3 共性算子 vs 非共性算子

| 算子 | Mamba | Attention | MLP | 分类 |
|------|-------|-----------|-----|------|
| Linear (矩阵向量乘) | ✓ (×3) | ✓ (×4) | ✓ (×3) | **共性** |
| RmsNorm | ✓ | ✓ | ✓ | **共性（轻量）** |
| Residual Add | ✓ | ✓ | ✓ | **共性（轻量）** |
| CausalConv1D | ✓ | ✗ | ✗ | **专用** |
| SelectiveScan | ✓ | ✗ | ✗ | **专用** |
| DotProduct (Q·K) | ✗ | ✓ | ✗ | **专用** |
| WeightedSum (w·V) | ✗ | ✓ | ✗ | **专用** |
| KV Cache | ✗ | ✓ | ✗ | **专用** |
| SiLU + ElemMul | ✗ | ✗ | ✓ | **专用** |

**关键结论**：全层共有 **10 次 Linear 投影**（Mamba 3 + Attention 4 + MLP 3），结构完全相同。**这就是统一 MAC 阵列的设计基础。**

---

## 第三章 统一执行引擎架构

### 3.1 整体架构

**注意（MAC 数量的准确表述）**：

当前实现中，`SerialSharedLinear4`（投影 MAC）、`SerialCausalConvMini`（卷积 MAC）、`SerialSelectiveScanMini`（扫描 MAC）各自有独立的物理 MAC 单元，共约 3 个。因此论文定位应为：

> **projection-dominant unified MAC reuse**：投影（最主要的计算量）走共享 MAC；卷积、扫描、注意力打分由各自专用小单元处理。

未来若要声称"全局 1 个 MAC"，需要将 conv/scan/score 的 MAC 也合并进投影 MAC，难度更高，作为进阶方向。

```
┌──────────────────────────────────────────────────────────────────┐
│                    统一执行引擎 (UnifiedJamba2MiniLayer)           │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │               统一调度器 (Unified Scheduler FSM)          │     │
│  │   输入: layer_type (Mamba/Attention), start              │     │
│  │   输出: 权重选择, MAC 使能, 专用单元使能, 数据路由控制     │     │
│  └──────────────────────────┬────────────────────────────┘      │
│                             │                                     │
│  ┌──────────────────────────▼────────────────────────────┐      │
│  │         共享 Projection MAC (SerialSharedLinear4)       │      │
│  │  分时复用，每次投影 16 周期；支持全部 10 次 Linear 投影   │      │
│  │  (Mamba×3 + Attention×4 + MLP×3)                       │      │
│  └──────────────────────────┬────────────────────────────┘      │
│                             │                                     │
│  ┌──────────────────────────▼────────────────────────────┐      │
│  │              专用算子单元 (各自独立 MAC)                 │      │
│  │                                                         │      │
│  │  ┌──────────────────┐  ┌─────────────────────────────┐ │      │
│  │  │SerialCausalConv  │  │ Attention Score + KV Cache  │ │      │
│  │  │(Mamba only, 1MAC)│  │ (Attention only, 组合逻辑)   │ │      │
│  │  └──────────────────┘  └─────────────────────────────┘ │      │
│  │  ┌──────────────────┐  ┌─────────────────────────────┐ │      │
│  │  │SerialSelectiveScan│ │ SiLU + Element-wise Mul      │ │      │
│  │  │(Mamba only, 1MAC)│  │ (MLP, combinational)         │ │      │
│  │  └──────────────────┘  └─────────────────────────────┘ │      │
│  │  ┌──────────────────────────────────────────────────┐   │      │
│  │  │ RmsNormApprox × 2 + Residual Add (combinational) │   │      │
│  │  └──────────────────────────────────────────────────┘   │      │
│  └────────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────────┘
```

### 3.2 统一调度器状态机

#### Mamba 路径序列（每 token 约 76 周期）

```
idle
 → norm1 (组合逻辑, 0额外周期)
 → proj_input (MAC, 16周期)  ← weight=mambaInputWeight
 → proj_B     (MAC, 16周期)  ← weight=mambaBWeight
 → proj_C     (MAC, 16周期)  ← weight=mambaCWeight
 → causalConv (专用, 16周期)  ← SerialCausalConvMini
 → scan       (专用, 12周期)  ← SerialSelectiveScanMini
 → [firstResidual = x + y_mamba, 组合]
 → norm2      (组合逻辑)
 → proj_gate  (MAC, 16周期)  ← weight=mlpGateWeight
 → proj_up    (MAC, 16周期)  ← weight=mlpUpWeight
 → [silu+mul, 组合]
 → proj_down  (MAC, 16周期)  ← weight=mlpDownWeight
 → [y = firstResidual + y_mlp, 组合]
 → done
```

**总计**：6 × MAC (16周期) + Conv(16) + Scan(12) = 96+28 = **约 124 周期**（精确值含控制开销）

#### Attention 路径序列（每 token 约 65 周期）

```
idle
 → norm1 (组合逻辑)
 → proj_Q    (MAC, 16周期)  ← weight=qWeight
 → proj_K    (MAC, 16周期)  ← weight=kWeight
 → proj_V    (MAC, 16周期)  ← weight=vWeight
 → writeKV + score_compute (专用, 组合+1周期)
 → proj_out  (MAC, 16周期)  ← weight=attentionOutWeight
 → [firstResidual = x + y_attention]
 → norm2 (组合逻辑)
 → proj_gate (MAC, 16周期)
 → proj_up   (MAC, 16周期)
 → [silu+mul]
 → proj_down (MAC, 16周期)
 → [y = firstResidual + y_mlp]
 → done
```

**总计**：7 × MAC (16周期) + Score(组合) = **约 112 周期**

### 3.3 共享 MAC 阵列的权重切换机制

调度器通过寄存器选择向 MAC 送入不同的 weight/bias：

```
MAC_weight = MuxCase(mambaInputWeight, Seq(
  (proj_idx === B_PROJ)    → mambaBWeight,
  (proj_idx === C_PROJ)    → mambaCWeight,
  (proj_idx === Q_PROJ)    → qWeight,
  (proj_idx === K_PROJ)    → kWeight,
  ...
))
```

权重本身在本阶段来自 IO 端口（后期可改为 BRAM）。

### 3.4 数据通路寄存器

| 寄存器 | 位宽 | 用途 |
|--------|------|------|
| `xReg` | 4×8b | 暂存输入 x |
| `norm1Reg` | 4×32b | norm1 输出（或直接组合） |
| `projInputReg` | 4×32b | Mamba input_proj 输出 |
| `projBReg` / `projCReg` | 4×32b | B/C 投影结果 |
| `projQReg` / `projKReg` / `projVReg` | 4×8b | Q/K/V（窄化后） |
| `firstResidualReg` | 4×8b | x + mixerY |
| `norm2Reg` | 4×32b | norm2 输出 |
| `gateReg` / `upReg` | 4×32b | MLP 中间值 |
| `yReg` | 4×32b | 最终输出 |
| `stateReg` | 4×32b | SSM 隐状态（跨 token 保持） |
| `keyCache` / `valueCache` | contextLen×4×8b | KV Cache（跨 token 保持） |

---

## 第四章 算法-硬件协同优化

### 4.1 量化（INT8）

**策略**：
- 权重：INT8（8-bit 有符号定点）
- 激活：INT8（投影输出截断）
- 累加器：INT32（防止溢出）
- MAC 单元：`MacLane(dataWidth=8, accWidth=32)`

**算法影响**：
- `narrowToData()` 函数在每次投影后截断到 INT8
- RmsNorm 近似（`RmsNormApprox`）：用移位代替除法
- Softmax 近似：用比较+移位代替指数

**硬件优势**：
- INT8 MAC 比 FP32 面积小约 8-16×
- FPGA DSP48 原生支持 18×27 乘法，INT8 可打包

### 4.2 稀疏化（跳零 MAC）

**策略**：Zero-skip — 当输入或权重为零时跳过 MAC 操作

**硬件实现**：
```
when(input =/= 0.S && weight =/= 0.S) {
  acc := acc + input * weight
}
// 否则直接跳到下一个 lane
```

**效果**：
- 对稀疏激活（ReLU 后）效果显著
- SiLU/SSM 激活不保证稀疏，需与量化配合

### 4.3 投影顺序优化

**洞察**：Mamba 的 B_proj 和 C_proj 与 input_proj 使用相同的输入 `x`（经 norm1 后），可以并行准备，但 MAC 只有一个，必须串行。

**优化**：将 B/C 投影提前到 CausalConv 之前，使 conv 和 scan 的输入提前就绪：

```
标准顺序: input_proj → conv → B_proj → C_proj → scan
优化顺序: input_proj → B_proj → C_proj → conv → scan
```

这样 scan 开始时 B/C 已就绪，减少等待。

### 4.4 RmsNorm 近似

原始 RmsNorm：`y = x / rms(x) * weight`，需要开方运算。

近似实现（已在 `RmsNormApprox` 中）：
- 计算 `sum_sq = Σ x_i²`
- 用移位近似 `1/sqrt(sum_sq)`（查表或迭代）
- 误差在整数域可接受

---

## 第五章 实现与验证

### 5.1 Chisel 模块层次（当前状态）

```
已完成的模块：
├── MacLane / MacLaneMixed              ← 物理 MAC 单元
├── SerialSharedLinear4                 ← 串行矩阵向量乘 (1 MAC, 16周期)
├── SerialProjectionScheduler4          ← N 次投影调度器
├── UnifiedProjectionScheduler4         ← Jamba 层级命名投影槽位调度器
├── UnifiedJamba2MiniLayer              ← 统一 projection scheduler 驱动的完整 dense 层
├── UnifiedMoEPathMini                  ← 统一调度版 router + top-1 expert MoE-lite
├── UnifiedJamba2MiniAcceleratorTile    ← 单层 unified accelerator 顶层（token/weight/status 接口）
├── UnifiedJamba2MiniTileScheduler      ← 多层 unified layer sequential scheduler
├── SerialMambaProjectionGroup          ← Mamba 3 投影语义封装
├── SerialAttentionProjectionGroup      ← Attention 4 投影语义封装
├── SerialCausalConvMini                ← 串行卷积 (1 MAC)
├── SerialSelectiveScanMini             ← 串行 SSM (1 MacLaneMixed, 12周期)
├── SerialMambaMixerMini                ← 完整 Mamba 路径
├── SerialAttentionMixerMini            ← 完整 Attention 路径
├── SerialJamba2MiniLayer               ← 完整层 (两路 Mixer + SharedMlpPath)
├── SharedMlpPathMini                   ← MLP 路径
├── RmsNormApprox                       ← RmsNorm 近似
├── Jamba2MiniTile (top/)               ← 已有 SharedFabric 版 Tile（含 token stream + weight load 接口）
└── WeightStoreMini (memory/)           ← 已有片上权重存储模块

待实现的模块：
└── UnifiedJamba2MiniFullTile           ← 多层 scheduler + accelerator weight/load shell 融合
```

### 5.2 三级资源对比实验（已完成）

通过 `scripts/resource_reuse_analysis.sh` 生成的代理数据：

| 模块 | Baseline | SharedFabric | SemanticSerial |
|------|----------|--------------|----------------|
| Linear4 | 16 mul | 4 mul | **1 mul** |
| CausalConvMini | — | — | **1 mul** |
| SelectiveScanMini | 12 mul | — | **1 mul** |
| Jamba2MambaMixerMini | 44 mul | 29 mul | **~2 mul** |
| AttentionMixerMini | TBD | TBD | **~1 mul** |
| Jamba2MiniLayer | TBD | TBD | **~2 mul** |

*mul = 生成 SV 中 ` * ` 操作符行数（代理，非综合结果）*

**这组数据是论文 "资源复用收益" 小节的核心图表。**

### 5.3 验证策略

每个模块均需通过：

1. **单元测试（ChiselTest）**：功能正确性，与 Python golden model 对比
2. **集成测试**：SerialJamba2MiniLayer 完整 token 处理
3. **SystemVerilog 生成**：`GenerateResourceReuseSweep.scala`，确保无综合错误
4. **Python golden 对齐**：`python/golden/mamba_ops.py` 逐函数覆盖
5. **FPGA 综合**：Vivado，采集 LUT/FF/DSP/BRAM/Fmax

### 5.4 当前测试状态

```
Chisel 测试: 142/142 通过 ✓
Python 测试: 19/19 通过 ✓
SV 生成:     clean ✓
FPGA 综合:   未开始 ○
```

---

## 第六章 评估计划

### 6.1 面积（资源）评估

**FPGA 目标板**：Xilinx（待定型号，实验室已有）

**采集指标**：

| 指标 | 工具 | 含义 |
|------|------|------|
| LUT | Vivado 综合报告 | 逻辑资源 |
| FF | Vivado | 触发器（状态寄存器） |
| DSP48 | Vivado | 硬件乘法器 |
| BRAM | Vivado | 片上存储（KV Cache、权重） |
| Fmax | Vivado 时序报告 | 最高频率 |

**对比维度**：
- Baseline vs SharedFabric vs SemanticSerial（三级对比）
- Mamba 层 vs Attention 层（路径对比）
- 不同 contextLength（KV Cache 压力）

### 6.2 延迟评估

**延迟定义**：每 token 处理周期数 × 时钟周期（1/Fmax）

| 路径 | 估算周期（lanes=4, taps=4, ctx=4） |
|------|-------------------------------------|
| Mamba 路径 | ~124 周期 |
| Attention 路径 | ~112 周期 |
| CPU Python baseline | 待测量（μs 级） |

**加速比**：需在同等精度下对比 FPGA 时延与 CPU 时延。

### 6.3 精度评估

- INT8 量化后与 FP32 golden 的数值误差（MSE / cosine similarity）
- 语言模型指标：perplexity（如果规模支持）

---

## 第七章 论文章节规划

### 预计章节结构

```
1. Introduction
   1.1 Hybrid sequence model challenges
   1.2 Resource reuse opportunity
   1.3 Contributions

2. Background
   2.1 Mamba / SSM
   2.2 Transformer / Attention
   2.3 Jamba2 hybrid architecture
   2.4 FPGA resources (DSP, BRAM)

3. Operator Analysis
   3.1 Per-layer operator taxonomy
   3.2 Common vs. non-common operators
   3.3 Resource sharing opportunities

4. Unified Execution Engine Architecture
   4.1 Overview
   4.2 Shared MAC array design
   4.3 Unified scheduler FSM
   4.4 Specialized units
   4.5 Data path and register file

5. Algorithm-Hardware Co-design
   5.1 INT8 quantization
   5.2 Zero-skip sparsification
   5.3 Projection ordering optimization

6. Implementation
   6.1 Chisel design methodology
   6.2 Module hierarchy
   6.3 Verification methodology

7. Evaluation
   7.1 Resource comparison (3-tier)
   7.2 Latency analysis
   7.3 FPGA synthesis results
   7.4 Accuracy under quantization

8. Conclusion and Future Work
```

---

## 工作进度追踪

### 已完成

- [x] 核心 MAC 单元（MacLane / MacLaneMixed）
- [x] SerialSharedLinear4（1 MAC 串行投影）
- [x] SerialProjectionScheduler4（N 次投影调度）
- [x] SerialMambaProjectionGroup / SerialAttentionProjectionGroup
- [x] SerialCausalConvMini（串行卷积）
- [x] SerialSelectiveScanMini（串行 SSM，替换并行版）
- [x] SerialMambaMixerMini（完整 Mamba 路径）
- [x] SerialAttentionMixerMini（完整 Attention 路径）
- [x] SerialJamba2MiniLayer（完整层）
- [x] 三级资源代理对比数据
- [x] 142 个 Chisel 测试，19 个 Python 测试

### 进行中

- [ ] **UnifiedJamba2MiniLayer**（论文核心贡献，下一步）
  - [x] 设计并实现统一 projection-slot 调度器 `UnifiedProjectionScheduler4`
  - [x] 将 projection-slot 调度器接入 layer FSM
  - [x] 集成 Mamba conv/scan、Attention KV/score、dense MLP
  - [x] 将 MoE-lite router/expert 纳入统一调度

### 待完成

- [x] UnifiedJamba2MiniAcceleratorTile（单层 unified accelerator shell）
- [x] UnifiedJamba2MiniTileScheduler（多层 unified layer sequencing）
- [ ] UnifiedJamba2MiniFullTile（多层 scheduler + weight/load shell）
- [ ] 片上权重 BRAM / WeightStore 集成（替换大 IO 端口权重）
- [ ] FPGA 综合（Vivado）
- [ ] CPU vs FPGA 延迟对比
- [ ] INT8 量化精度评估
- [ ] 论文写作

---

## 关键设计决策记录

| 决策 | 选择 | 原因 |
|------|------|------|
| 数据类型 | INT8 权重 + INT32 累加 | 配合 FPGA DSP，减少面积 |
| MAC 复用粒度 | 单 MAC 时间复用 | 最大化资源复用，代价是延迟增加 |
| norm 实现 | 组合逻辑近似 | 避免除法器面积，近似误差可接受 |
| MLP 实现 | SharedMlpPathMini（共享 fabric） | 当前阶段，后续可替换为统一调度 |
| 权重存储 | IO 端口（当前）→ BRAM（目标） | 先验证功能，再优化存储 |
| 参数规模 | Mini (lanes=4, taps=4, ctx=4) | 先跑通完整流程，再扩规模 |

---

---

## 详细实施计划

以下按优先级排序，每个阶段有明确的完成标准。

---

### 阶段 0：理解现状（已完成）

**目标**：厘清已有代码的结构与局限

**结论**（见上文"串行工作现在用在哪里"）：
- `SerialJamba2MiniLayer` 有 2 个独立 MAC（Mamba 路径一个，Attention 路径一个），不共享
- MLP 路径用 `SharedMlpPathMini`，内部是多个并行 MAC
- 现有串行模块是"可复用组件"，下一步把它们串进一个统一调度器

---

### 阶段 1：UnifiedJamba2MiniLayer（论文核心贡献）

**目标**：用 1 个物理 MAC 覆盖 Mamba + Attention + MLP 三条路径的全部 10 次投影

#### 步骤 1.1 — 设计调度器状态机（先在纸上/文档确认，再写代码）

状态枚举（Mamba 路径，约 18 个状态）：

```
idle
→ proj_input → wait_input      (MAC, mambaInputWeight)
→ proj_B     → wait_B          (MAC, mambaBWeight)
→ proj_C     → wait_C          (MAC, mambaCWeight)
→ launch_conv → wait_conv      (SerialCausalConvMini)
→ launch_scan → wait_scan      (SerialSelectiveScanMini)
→ proj_gate  → wait_gate       (MAC, mlpGateWeight)
→ proj_up    → wait_up         (MAC, mlpUpWeight)
→ proj_down  → wait_down       (MAC, mlpDownWeight)
→ done_state → idle
```

状态枚举（Attention 路径，约 20 个状态）：

```
idle
→ proj_Q    → wait_Q           (MAC, qWeight)
→ proj_K    → wait_K           (MAC, kWeight)
→ proj_V    → wait_V           (MAC, vWeight)
→ write_kv_score               (写 KV Cache + 组合打分，1周期)
→ proj_out  → wait_out         (MAC, attentionOutWeight)
→ proj_gate → wait_gate        (MAC, mlpGateWeight)
→ proj_up   → wait_up          (MAC, mlpUpWeight)
→ proj_down → wait_down        (MAC, mlpDownWeight)
→ done_state → idle
```

**关键设计要点**：
- `macX`（MAC 输入）根据当前状态用 `MuxCase` 选择：`xReg` / `convOutReg` / `norm2Reg` / `rawYReg`
- `macWeight`（权重）根据状态用 `MuxCase` 选择对应的权重矩阵
- `macBias` 同理
- RmsNorm 在 `idle→proj_input` 和 `wait_scan→proj_gate` 之间是组合逻辑，不需要额外状态

#### 步骤 1.2 — 新建文件

**新建**：`src/main/scala/jamba/fabric/UnifiedJamba2MiniLayer.scala`

模块结构：
```scala
class UnifiedJamba2MiniLayer(...) extends Module {
  // 1 个共享 MAC
  val mac = Module(new SerialSharedLinear4(dataWidth, accWidth))

  // 专用单元（复用已有模块）
  val conv = Module(new SerialCausalConvMini(lanes, taps, dataWidth, accWidth))
  val scan = Module(new SerialSelectiveScanMini(lanes, dataWidth, stateWidth, accWidth))

  // Attention 专用（组合逻辑，复用 SharedDotProduct 或内联）
  val keyCache   = Reg(Vec(contextLength, Vec(lanes, SInt(dataWidth.W))))
  val valueCache = Reg(Vec(contextLength, Vec(lanes, SInt(dataWidth.W))))

  // RmsNorm（组合逻辑，复用 RmsNormApprox）
  val norm1 = Module(new RmsNormApprox(lanes, dataWidth, accWidth))
  val norm2 = Module(new RmsNormApprox(lanes, dataWidth, accWidth))

  // 统一调度器 FSM（所有状态在这里）
  val state = RegInit(idle)
  // MuxCase 选择 MAC 的 weight/bias/x
  mac.io.weight := MuxCase(io.mambaInputWeight, Seq(...))
  ...
}
```

#### 步骤 1.3 — 新建测试

**新建**：`src/test/scala/jamba/fabric/UnifiedJamba2MiniLayerSpec.scala`

测试用例：
1. Mamba token，identity 权重 → 与 `SerialJamba2MiniLayer` 同参数输出一致
2. Attention token，单 token → KV cache 写入正确
3. 连续 Mamba + Attention token → 状态跨 token 保持
4. clear → 全部状态归零
5. 资源代理验证：生成 SV 后 ` * ` 数量应为 1（只有 1 个 MAC）

#### 步骤 1.4 — 加入资源对比扫描

修改 `src/main/scala/jamba/top/GenerateResourceReuseSweep.scala`，增加：

```scala
ChiselStage.emitSystemVerilogFile(
  new UnifiedJamba2MiniLayer() {
    override def desiredName = "Jamba2MiniLayer_Unified"
  }, firtoolOpts = firtoolOptions,
  args = Array("--target-dir", targetDir)
)
```

**完成标准**：
- [ ] `sbt test` 全部通过（含新 spec）
- [ ] 资源代理：`Jamba2MiniLayer_Unified` 的 ` * ` 数量 = 1
- [ ] 与 `SerialJamba2MiniLayer` 输出数值完全一致

---

### 阶段 2：UnifiedJamba2MiniAcceleratorTile（完整加速器顶层）

**背景**：项目已有 `src/main/scala/jamba/top/Jamba2MiniTile.scala`（SharedFabric 版 Tile，含 token stream + weight load 接口 + `WeightStoreMini`），可作为参考蓝本。

**目标**：以 `UnifiedJamba2MiniLayer` 为内核，实现对标已有 Tile 接口的 Unified 版顶层

#### 步骤 2.1 — 新建 Tile 模块

**新建**：`src/main/scala/jamba/top/UnifiedJamba2MiniAcceleratorTile.scala`

参考 `Jamba2MiniTile.scala` 的接口设计：
- `io.start / io.clear / io.inValid / io.inReady / io.in`（token 输入）
- `io.outValid / io.out`（token 输出）
- `io.useLoadedWeights`（使用片上权重）
- weight load 接口（与 `WeightStoreMini` 对接）

内核替换为 `UnifiedJamba2MiniLayer`（每层一个，层间 y→x 连接）。

**关键设计**：
- `attentionEvery` 参数控制 Mamba/Attention 层比例（Jamba2 约 1:7）
- 每层的 SSM 状态和 KV Cache 跨 token 保持
- Tile 级别 FSM：`idle → layer0 → layer1 → ... → done`

#### 步骤 2.2 — 加入规模扫描

修改 `src/main/scala/jamba/top/GenerateScaleSweep.scala`，增加 `UnifiedJamba2MiniAcceleratorTile` 配置，与现有 `Jamba2MiniTile` 并列对比。

**完成标准**：
- [ ] 4 层 Tile 功能测试通过，与 Python golden 对比一致
- [ ] 接口与已有 `Jamba2MiniTile` 兼容（可替换）
- [ ] 资源对比：Unified Tile vs SharedFabric Tile 的 MAC 代理数量

---

### 阶段 3：片上权重存储（BRAM）

**目标**：把权重从 IO 端口移到 BRAM，模拟真实部署场景

#### 步骤 3.1 — 设计 WeightBRAM 模块

**新建**：`src/main/scala/jamba/fabric/WeightBRAM.scala`

```scala
class WeightBRAM(
    numLayers: Int,
    lanes:     Int,
    dataWidth: Int
) extends Module {
  // 用 SyncReadMem 模拟 BRAM
  // 地址空间：layer_idx × proj_idx × row × col
  // 读接口：给 UnifiedJamba2MiniLayer 提供当前投影的权重行
}
```

**与调度器集成**：
- 调度器每个 MAC 状态开始时发出地址请求
- BRAM 返回权重行（1 周期延迟，或同步读）

**完成标准**：
- [ ] BRAM 读延迟不影响调度器状态机正确性
- [ ] 资源对比：BRAM 版本 IO 端口数量大幅减少
- [ ] FPGA 综合中 BRAM 原语被识别（不是分布式 RAM）

---

### 阶段 4：FPGA 综合与上板

**目标**：采集真实的 LUT/DSP/BRAM/Fmax 数据

#### 步骤 4.1 — Vivado 综合

```bash
# 生成 SV
sbt "runMain jamba.top.GenerateResourceReuseSweep generated/resource_reuse"

# 进入 Vivado
# 新建工程，添加 generated/resource_reuse/Jamba2MiniLayer_Unified.sv
# 设置目标器件（实验室 FPGA 型号）
# 运行综合 + 实现，采集报告
```

**采集数据**：

| 指标 | Baseline | SharedFabric | SemanticSerial | Unified |
|------|----------|--------------|----------------|---------|
| LUT | — | — | — | — |
| DSP48 | — | — | — | — |
| BRAM | — | — | — | — |
| Fmax (MHz) | — | — | — | — |

#### 步骤 4.2 — 时钟频率优化

常见问题及应对：
- 关键路径在 MuxCase（权重选择）→ 拆成流水级或减少选择深度
- RmsNorm 组合路径过长 → 加寄存器打拍
- KV Cache 读写时序 → 改为 SyncReadMem（BRAM 推断）

**完成标准**：
- [ ] 综合无错误，时序收敛（Fmax ≥ 100 MHz，目标）
- [ ] DSP48 数量 = 1（对应 1 个 MAC）
- [ ] 填写上表所有数据

---

### 阶段 5：算法-硬件协同优化实验

**目标**：量化、稀疏化带来的收益数据，作为论文优化章节的内容

#### 步骤 5.1 — INT8 量化精度实验

**Python 端**：在 `python/golden/mamba_ops.py` 中实现量化版本：

```python
def quantize_int8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x), -128, 127).astype(np.int8)

def jamba2_mini_layer_quantized(x, weights, ...):
    # 每次投影后 quantize_int8
    ...
```

**评估指标**：
- FP32 输出 vs INT8 输出的 MSE / cosine similarity
- 验证 Chisel 实现（INT8 MAC）与量化 Python golden 的数值对齐

#### 步骤 5.2 — 跳零稀疏化实验

在 `SerialSharedLinear4` 或 `MacLane` 中增加 zero-skip：

```scala
// 在 SerialSharedLinear4 的 run 状态中
val skip = (io.x(laneReg) === 0.S) || (weightRow(colReg) === 0.S)
when(!skip) {
  acc := acc + io.x(laneReg) * weightRow(colReg)
}
colReg := colReg + 1.U  // 无论是否 skip 都推进（保持时序一致）
```

**评估**：统计稀疏激活比例对实际周期数的影响（需要动态 zero-skip，跳过 colReg 推进）

#### 步骤 5.3 — CPU baseline 测量

```python
# python/benchmarks/cpu_baseline.py
import time
import numpy as np

def benchmark_jamba2_mini_token(x, weights, num_tokens=100):
    start = time.perf_counter()
    for _ in range(num_tokens):
        y = jamba2_mini_layer_fp32(x, weights)
    elapsed = (time.perf_counter() - start) / num_tokens
    print(f"CPU per-token latency: {elapsed*1e6:.1f} μs")
```

**FPGA 延迟**：`周期数 / Fmax`，例如 124 周期 / 100 MHz = 1.24 μs

**完成标准**：
- [ ] CPU baseline 数据（μs/token）
- [ ] FPGA 延迟估算（μs/token）
- [ ] 量化误差在可接受范围（cosine similarity > 0.99）

---

### 阶段 6：论文写作

**目标**：整理所有数据，撰写论文

#### 写作顺序（建议）

1. **第三章（算子分析）** — 先写，表格数据已有，最容易落笔
2. **第六章（评估）** — 数据驱动，填表格
3. **第四章（架构）** — 边实现边写，实现完成后整理
4. **第二章（背景）** — 参考论文读完后写
5. **第一章（绪论）** — 最后写，此时贡献点最清晰
6. **第七章（结论）** — 最后

#### 关键图表清单

| 图表 | 内容 | 数据来源 |
|------|------|---------|
| Fig 1 | Jamba2 层结构 + 算子标注 | 手绘 |
| Fig 2 | 统一执行引擎架构图 | 本文设计 |
| Fig 3 | 调度器状态机 | 本文设计 |
| Table 1 | 算子分类表（共性/非共性） | 本文分析 |
| Table 2 | 三级资源对比（代理数） | `resource_reuse_analysis.sh` |
| Table 3 | FPGA 综合结果 | Vivado 报告 |
| Table 4 | CPU vs FPGA 延迟对比 | benchmark 测量 |
| Table 5 | INT8 量化精度 | Python 实验 |

---

### 整体时间线（参考）

| 阶段 | 估计工作量 | 产出 |
|------|-----------|------|
| 阶段 1：UnifiedJamba2MiniLayer | 1~2 周 | 核心模块 + 测试 |
| 阶段 2：Jamba2MiniTile | 1 周 | 完整骨架 |
| 阶段 3：WeightBRAM | 1 周 | 片上存储 |
| 阶段 4：FPGA 综合 | 1~2 周 | 综合数据 |
| 阶段 5：协同优化实验 | 1 周 | 量化/稀疏数据 |
| 阶段 6：论文写作 | 持续进行 | 论文初稿 |

**当前位置：阶段 1 起点**

---

*最后更新：2026-05-18*
*维护者：论文作者*
