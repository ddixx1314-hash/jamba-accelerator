# 相关论文阅读笔记

> 整理与本项目最相关的加速器论文，重点关注架构设计、资源使用和性能数字，用于 baseline 对比和设计参考。
> 持续更新。最后更新：2026-05-25

---

## 一、模型论文

### Samba（arxiv 2406.07522，ICLR 2025）

**全名**：Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling

**核心思想**：逐层交替 Mamba（SSM）和 Sliding Window Attention（SWA），兼顾线性复杂度的长序列处理和注意力机制的精确短程召回。

**和本项目的关系**：
- Samba 的混合层结构（Mamba + Attention 交替）和 Jamba2 的设计思路相近，但比例不同（Samba 更激进地交替，Jamba2 约 7:1 偏向 Mamba）
- 两者都面临同一个硬件挑战：Mamba 和 Attention 路径的算子异构，需要在硬件上统一支持
- 本项目的统一调度器和多层共享方案同样适用于 Samba-style 架构

**值得关注的内容**：
- Sliding Window Attention 的 KV Cache 大小固定（窗口大小），比全局 Attention 更适合硬件实现
- SSM 状态的跨层传递机制

---

## 二、FPGA 加速器论文（Baseline 候选）

### LightMamba（arxiv 2502.15260，2025 年 2 月）

**全名**：LightMamba: Efficient Mamba Acceleration on FPGA with Quantization and Hardware Co-design

**目标平台**：Xilinx Versal VCK190（AI Engine + FPGA），Alveo U280

---

#### 核心架构

三大功能单元：**MMU**（矩阵乘法单元）+ **SSMU**（SSM 单元）+ **HTU**（Hadamard 变换单元）

- **MMU**：时间复用处理输入/输出投影，MAC tree 架构；先生成 Δ、B、C 并存入 on-chip buffer，X 和 Z 交替产生（Computation Reordering）
- **SSMU**：完全展开 SSM，各算子由专用单元 + FIFO 连接，形成流水线
- **HTU**：定制 Hadamard 变换，消除 activation outlier，支持 rotation-assisted 量化

**流水线优化**：
- **Computation Reordering**：重排 MMU 计算顺序，总计算时间减少 32%，硬件利用率从 58% 提升到 96%
- **Fine-grained Tiling**：按 nₚ × pₚ tile 尺寸切分 SSMU 中的 BX、Āhₜ₋₁、hₜ 中间数据；消除流水线气泡，同时将 URAM 使用量从 246 降至 **61（减少 4×）**

---

#### 量化方案

- **线性层**：rotation-assisted 量化（Hadamard 变换消除 outlier）+ W4A4 或 W8A8
- **SSM 块**：power-of-two（PoT）细粒度量化，大部分计算降至 4-bit
- **精度对比**（Mamba2-2.7B，越低越好 PPL）：
  - W8A8：LightMamba 4.07 PPL（接近 FP16 的 4.10），优于 RTN/SQ/OS+
  - W4A4：LightMamba 6.48 PPL（明显优于 RTN 17.46 / SQ 8.26 / OS+ >100）

---

#### FPGA 资源（Table IV）

| 平台 | 精度 | LUT | FF | DSP | BRAM | URAM | 吞吐 (tokens/s) | 能效 (tokens/s/W) |
|------|------|----:|---:|----:|-----:|-----:|----------------:|------------------:|
| VCK190 | W4A4 | 107k | 130k | 228 | 912 | 61 | **7.21** | 2.25 |
| VCK190 | W8A8 | 111k | 134k | 228 | 914 | 61 | 3.61 | 1.45 |
| U280   | W4A4 | 297k | 394k | 1164 | 912 | 61 | **93** | — |

**GPU Baseline（FP16，吞吐 tokens/s）**：RTX 2070 = 65，RTX 4090 = 138

**关键观察**：
- LightMamba (VCK190 W4A4) 与 RTX 2070 吞吐接近（7.21 vs 65 tokens/s），但能效高 **4.65–6.06×**
- U280（带宽 460GB/s）模拟达到 **93 tokens/s**，超越 RTX 2070（1.43×）
- W4A4 vs W8A8：吞吐提升 **2×**（7.21 vs 3.61），资源几乎不变（DSP=228 不变，LUT 增加 4k）
- DSP 占用极低（228）是因为大量计算用 LUT 实现的 MAC tree 而非 DSP

---

#### 对本项目的参考价值

| 维度 | LightMamba | 本项目对应 |
|------|-----------|-----------|
| 投影时间复用 | MMU 顺序处理输入/输出投影 | **UnifiedProjectionScheduler4** 10 路投影顺序调度 |
| SSM 专用单元 | SSMU 完全展开，流水线 | **SerialSelectiveScanMini** 逐步状态更新 |
| 非线性近似 | HTU + PoT 量化，4-bit | **RmsNormApprox**（梯度替代） |
| 资源分析粒度 | FPGA 综合（LUT/DSP/BRAM） | 结构代理（Mul-proxy，综合前） |

- ✅ MMU 时间复用投影 = 直接类比 UnifiedProjectionScheduler4，可在论文中明确对标
- ✅ SSMU 专用单元 + HTU 的资源分配方式值得参考
- ⚠️ 只针对纯 Mamba 模型，无 Attention 和 MoE 路径——正是本项目扩展之处
- **Baseline 适用性**：★★★★☆

---

### FastMamba（arxiv 2505.18975，IEEE FPGA 2025）

**全名**：FastMamba: A High-Speed and Efficient Mamba Accelerator on FPGA with Accurate Quantization

**目标平台**：Xilinx Virtex-7 VX690T（28nm，**250MHz**）

---

#### 核心架构

两大计算组：**定点计算组**（Hadamard Linear + 卷积 + SSM 模块）+ **浮点计算组**（RMS Norm + SiLU 模块）

**VPU（向量处理单元）**共 5 种类型（Table I）：

| 编号 | 类型 | 功能 | 输入→输出 |
|------|------|------|-----------|
| ① | PAU（Parallel Adder Unit） | A + B | A·n, B·n → P·n |
| ② | PMU（Parallel Multiplier Unit） | A × B | A·n, B·n → P·n |
| ③ | PMA（Parallel Multiplier Adder Unit） | A × B + C | A·n, B·n, C·n → P·n |
| ④ | HAT（Hadamard Adder Tree） | ∑Aᵢ（归约）| A·n → P·1 |
| ⑤ | MAT（Multiplier Adder Tree） | ∑(Aᵢ×Bᵢ) | A·n, B·n → P·1 |

**Hadamard-based Linear Module**：
- 6 个并行 computing group（i=0,…,5）
- 每组：4 个并行 HAT（Hadamard 变换）+ 64 个并行 MAT（8-bit 矩阵乘）
- 生成 8-bit 量化激活向量 X̂ₕ[i]（长度 4），暂存 buffer，再做矩阵积

**Convolution Module**：32 个 MAT 单元，对长度 4 的向量做 1-D 卷积

**SSM Module**（三步，对应 Fig. 7）：
- Step1：PAU + **24 并行非线性近似单元（NAU）** → 计算 Δ̄ ∈ ℝ¹×²⁴
- Step2：两路并行：PMU + NAU（指数模式，exp(ΔA)）+ 32 并行 PMU（Q = QB）
- Step3：32 并行 PMU/PMA + MAT → 更新隐状态 H，输出 Y

**Nonlinear Approximation Unit（NAU）**（Fig. 8）：
- 24 路并行，16-bit 定点输入输出
- 双模式：指数模式（EXP-INT，实现 eˣ ≈ 2^(u+v)，u 整数部分，v 由 8 段高精度一阶近似）和 SoftPlus 模式
- **一阶线性近似方案**：
  - eˣ（x≤0）：eˣ = 2^(u+v) = 2ᵘ × 2ᵛ ≈ 2ᵘ × 1.0111₂
  - SoftPlus（x≤0）：SoftPlus(x) ≈ eˣ（由 EXP-INT 直接计算）
  - SoftPlus（x>0）：SoftPlus(x) = e^(-x) + x（利用对称性转换为 x≤0 情形）
- 与浮点 NAU 对比：**节省 56% DSP 和 49% FF**

**浮点计算组**：
- RMS Norm 和 SiLU 在 Mamba2 计算负载中占比较小（Fig. 1 所示）
- 选择浮点单元以避免精度损失，带来的硬件开销较小

---

#### 量化方案

- **线性层**：Hadamard 变换消除 outlier + W8A8 INT8
- **SSM 块**：power-of-two（PoT）细粒度量化（卷积 + SSM 乘法）
- **非线性函数**：一阶线性近似（SoftPlus、exp），无需查表
- **精度对比**（Mamba2-130M，W8A8，越高越好 ACC）：
  - FastMamba W8A8 平均准确率 42.2%，接近 FP16 的 42.6%，优于 NormalQ（39.8）/SmoothQ（41.7）

---

#### FPGA 资源（Table III 系统配置 + Table IV 资源）

**Table III 系统配置**：

| 指标 | CPU | GPU | **FastMamba** |
|------|-----|-----|---------------|
| 平台 | Intel Xeon Silver 4210R (14nm) | NVIDIA RTX 3090 (8nm) | Xilinx VX690T (28nm) |
| 频率 | 2400MHz | 1395MHz | **250MHz** |
| 计算单元 | 10 Cores | 328 Tensor Cores | **3333 DSPs** |
| 吞吐（decode） | 111 tokens/s | — | **5.68 tokens/s** |
| 能效（decode） | 0.37 token/(s·W) | — | **0.61 token/(s·W)** |

**Table IV 资源分布（VC709）**：

| 模块 | LUT（%） | FF（%） | DSP（%） | BRAM（%） |
|------|--------:|-------:|--------:|----------:|
| Linear | 132,030 (30.5%) | 84,514 (9.8%) | 48 (1.3%) | 0 |
| Convolution | 14,125 (3.3%) | 13,201 (1.5%) | 256 (7.1%) | 0 |
| **SSM** | **73,597 (17.0%)** | **58,196 (6.7%)** | **2,376 (66.0%)** | 0 |
| RMS Norm & SiLU | 57,315 (13.2%) | 87,633 (10.1%) | 192 (5.3%) | 0 |
| Buffer | 13,597 (3.1%) | 64,898 (7.5%) | 0 | 956 (65.0%) |
| Others | 44,120 (10.2%) | 46,022 (5.3%) | 192 (5.3%) | 0 |
| **Total** | **334,784 (77.3%)** | **354,464 (40.9%)** | **3,333 (92.5%)** | **956 (65.0%)** |

**关键观察**：
- SSM 模块消耗 **2,376 DSP（占总 DSP 的 71.3%）**，是主要资源瓶颈
- Linear 模块只用 48 DSP（1.3%），因为 Hadamard-based 的 MAT 单元用 LUT 实现
- RMS Norm & SiLU 用浮点单元（57k LUT + 88k FF），无 DSP
- BRAM 全部在 Buffer（956，65%），计算模块不用 BRAM

---

#### 性能

- **Prefill 加速比（Mamba2-130M）**：最高 68.80× CPU，8.90× GPU；平均 55.70× CPU，6.06× GPU
- **Decode 吞吐**：5.68 tokens/s（Mamba2-2.7B）
- **Decode 能效**：0.61 token/(s·W)，优于 RTX 3090（**1.65×**）
- 加速主要来源：定点量化 + 并行 VPU + 流水线设计

---

#### 对本项目的参考价值

| 维度 | FastMamba | 本项目对应 |
|------|-----------|-----------|
| 非线性近似 | 一阶线性近似 SoftPlus/exp（PoT + 8 段分段线性） | **RmsNormApprox**（梯度替代近似），SiLU 组合逻辑 |
| SSM 计算 | 32 并行 PMU/PMA/MAT，3333 DSP 总量 | **SerialSelectiveScanMini** 单 MAC lane 串行 |
| 资源权衡 | 高并行（多 DSP → 低延迟） | 高共享（1 MAC lane → 低面积）|
| 线性层 | Hadamard 变换 + 64 并行 MAT × 6 组 | 时间复用 `UnifiedProjectionScheduler4` |

- ✅ 非线性函数一阶近似做法可直接参考（本项目 RMSNorm 近似）
- ✅ 资源分布表（Table IV）数字完整，是论文中"与 FPGA baseline 对比"的最佳参照
- ✅ VPU 类型定义（PAU/PMU/PMA/HAT/MAT）与本项目 MacLane 的原子 op 对应方式类似
- ⚠️ 同样只针对 Mamba2，无混合路径
- ⚠️ 3333 DSP vs 本项目 1 MAC lane：两者处于 资源-延迟 Pareto 的两个极端，可作为论文的"对比定位"
- **Baseline 适用性**：★★★★★，资源数字完整，最适合作论文 baseline

---

## 三、技术参考

### CODA（导师分享）

**核心思想**：通过数学等价变换，将 Transformer 中的访存密集型小算子（LayerNorm、激活函数、残差加法等）融合进相邻 GEMM 的 epilogue，让这些操作在数据还在寄存器/SRAM 中时完成，避免独立的 memory-bound kernel。

**结论**：整个 Transformer 经过变换后可以表示为一系列 GEMM + epilogue 的组合，用几个优化原语就能写出接近理论上限的 kernel。

**对本项目的启示**：

| 当前做法 | CODA 启发的改进方向 |
|----------|---------------------|
| RMSNorm 作为独立组合逻辑模块，输出需额外寄存器 | 将 RMSNorm 的缩放操作融合进后续投影 MAC 的输入端 |
| 每次投影后 `narrowToData()` 截断独立执行 | 将截断融合进 MAC 输出端，不需要额外周期 |
| 激活函数（SiLU）独立组合逻辑 | 融合进 down_proj 的 MAC 输入端 |

这是减少 memory bound 的一个具体方向，与导师强调的"动态控制和数据复用"直接对应。

---

## 四、与本项目的差异定位

| 维度 | LightMamba | FastMamba | 本项目 |
|------|-----------|-----------|--------|
| 模型覆盖 | 纯 Mamba | 纯 Mamba2 | **Mamba + Attention + MoE 混合** |
| 核心贡献 | 量化 + 流水线重排 | 量化 + 高并行 VPU | **异构算子间 MAC 复用 + 多层状态虚拟化** |
| 资源规模 | 228–1164 DSP | **3,333 DSP** | **1 MAC lane（最小化面积）** |
| 资源分析 | FPGA 综合（LUT/DSP/BRAM） | FPGA 综合（LUT/DSP/BRAM）| 结构代理（Mul-proxy，综合前） |
| 目标模型规模 | 130M–2.7B | 130M–2.7B | Mini 原型（lanes=4，4×4 矩阵） |
| 动态控制 | 静态流水线 + 重排 | 固定 VPU 数据流 | 静态槽表 FSM（**待加强动态控制**） |
| 平台 | VCK190 / U280 | VC709 | Chisel → SystemVerilog（无板卡） |

**论文定位**：
- LightMamba / FastMamba 的 **MMU 时间复用** 和本项目的 **UnifiedProjectionScheduler4** 是独立提出的类似设计——说明单 MAC lane 时间复用是 FPGA 上 Mamba 投影加速的一个合理方向
- 本项目的独特贡献在于把这个思路扩展到了 **Mamba + Attention + MoE 三路异构统一**，而已有工作只做了 Mamba 单路
- FastMamba（3333 DSP）和本项目（1 MAC）处于资源-延迟 Pareto 的两端，可作为论文"design space exploration"的两个端点引用
