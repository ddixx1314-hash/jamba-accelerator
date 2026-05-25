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

**核心架构**：
- 空间架构，部分展开一个 Mamba block
- 三大单元：矩阵乘法单元（MMU）+ SSM 单元（SSMU）+ Hadamard 变换单元（HTU）
- MMU：时间复用处理输入/输出投影，MAC tree 架构
- SSMU：完全展开 SSM，流水线执行，各算子由专用单元 + FIFO 连接

**量化方案**：
- rotation-assisted 量化（线性层）
- power-of-two SSM 量化，大部分计算降至 4-bit

**FPGA 资源（VCK190）**：

| 资源 | 用量 |
|------|------|
| LUT | 107k |
| FF | 130k |
| DSP | 228 |
| BRAM | 912 |
| URAM | 61 |

**性能**：
- VCK190（W4A4）：7.21 tokens/s
- U280 模拟：93 tokens/s（1.43× GPU baseline）
- 能效比：4.65–6.06× 优于 GPU

**Baseline 对比**：RTX 2070 / RTX 4090 GPU

**对本项目的参考价值**：
- ✅ MMU 时间复用投影的设计与本项目 `UnifiedProjectionScheduler4` 思路相同，可作为直接类比
- ✅ SSMU 专用单元的设计可参考
- ⚠️ 针对纯 Mamba 模型，没有 Attention 和 MoE 路径的资源复用研究
- **Baseline 适用性**：★★★★☆，可作为 Mamba 部分的对比基准

---

### FastMamba（arxiv 2505.18975，IEEE FPGA 2025）

**全名**：FastMamba: A High-Speed and Efficient Mamba Accelerator on FPGA with Accurate Quantization

**目标平台**：Xilinx Virtex-7 VC709（28nm，250MHz）

**核心架构**：
- 定点计算组：Hadamard Linear 模块 + 卷积模块 + SSM 模块
- 浮点计算组：RMSNorm 模块 + SiLU 模块
- On-chip Buffer + Data Flow Handler 管理数据流
- SSM 模块：32 个并行乘加单元 + C 矩阵内积处理

**量化方案**：
- 线性层：Hadamard 变换消除 outlier + INT8
- SSM 块：power-of-two 细粒度量化
- 非线性函数（SoftPlus、exp 等）：**一阶线性近似**（这点和本项目 RMSNorm 近似思路一致）

**FPGA 资源（VC709）**：

| 资源 | 用量 | 利用率 |
|------|------|--------|
| LUT | 334,784 | 77.3% |
| FF | 354,464 | 40.9% |
| DSP | 3,333 | 92.5% |
| BRAM | 956 | 65.0% |

**性能**：
- Prefill 加速比：68.80× vs CPU，8.90× vs GPU（Mamba2-130M）
- 解码能效：6× 优于 RTX 3090

**Baseline 对比**：Intel Xeon 4210R CPU、NVIDIA RTX 3090 GPU

**对本项目的参考价值**：
- ✅ 非线性函数一阶近似的做法可直接参考（本项目 RMSNorm 近似、Attention 归一化近似）
- ✅ 资源数字详细，可作为综合后对比的参考标准
- ⚠️ 同样只针对 Mamba 模型，无混合路径
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

| 维度 | LightMamba / FastMamba | 本项目 |
|------|------------------------|--------|
| 模型覆盖 | 纯 Mamba / Mamba2 | **Mamba + Attention + MoE 混合** |
| 核心贡献 | 量化 + 专用硬件单元 | **异构算子间 MAC 资源复用 + 多层状态虚拟化** |
| 资源分析 | FPGA 综合（LUT/DSP/BRAM） | 结构代理（Mul-proxy，综合前） |
| 规模 | 实际模型参数（130M–2.7B） | Mini 原型（lanes=4，4×4 矩阵） |
| 动态控制 | 固定数据流，部分流水线 | 静态槽表 FSM（待加强动态控制） |

**论文定位**：本项目的核心贡献点（混合算子统一调度 + 层级状态虚拟化）在已有文献中尚无直接对应，LightMamba/FastMamba 可作为 Mamba 子路径的性能 baseline，本项目在此基础上扩展了 Attention 和 MoE 的统一支持。
