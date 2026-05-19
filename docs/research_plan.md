# Research Plan

## Target Paper Direction

Working title:

```text
A Resource-Reusable FPGA Accelerator Prototype for Jamba 2.0 Mini Style Hybrid Sequence Models
```

The paper goal is not to claim a full large-language-model accelerator. The goal is to show that Mamba, attention, convolution, linear projection, and MLP operators share enough arithmetic structure to justify a reusable hardware fabric.

Advisor clarification: "1:1" means supporting the complete algorithm semantics. Quantization and sparsification are allowed as hardware-aware algorithm variants. Therefore, the project should build an algorithm-complete mini baseline, then compare optimized quantized/sparse/resource-reused variants against it.

## Reference Models

Primary references:

- Transformer: `Attention Is All You Need`, arXiv:1706.03762.
- Mamba: `Mamba: Linear-Time Sequence Modeling with Selective State Spaces`, arXiv:2312.00752.
- Jamba: `Jamba: A Hybrid Transformer-Mamba Language Model`, arXiv:2403.19887.
- Samba: `Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling`, arXiv:2406.07522 and ICLR 2025 OpenReview.

These papers define the software-side operators. This project implements mini hardware versions of the core formulas to make the dataflow and resource pressure visible.

## Operator Milestones

The implementation order is:

```text
DotProduct
Linear / GEMM
RMSNorm
CausalConv1D
Mamba State Update
Selective Scan
Attention Decode
Sliding Window Attention
MLP
MoE-lite
Jamba2 Mini Layer
Jamba2 Mini Tile
```

Every new operator should include:

- a formula-level explanation.
- a Python golden model when the behavior is new.
- a Chisel module.
- a chiseltest test.
- a short Verilog correspondence note.
- a resource-reuse note.

## Research Contributions

The intended contributions are:

- A Chisel Jamba 2.0 Mini style accelerator prototype.
- A layer-by-layer operator taxonomy that separates common and non-common operators.
- One-to-one mini hardware implementations of the core Mamba, attention, and MLP operators.
- A reusable MAC/reduction fabric shared across different operator types.
- A unified datapath that executes the full mini model through shared arithmetic resources and operator-specific state.
- Baseline-versus-shared resource comparison from generated hardware.
- FPGA-ready SystemVerilog and a path to board synthesis.

## Advisor-Driven Architecture Method

The implementation should follow this order:

```text
1. Extract common and non-common operators from each layer.
2. Map common operators to reusable hardware resources.
3. Keep non-common behavior in operator-specific state/control blocks.
4. Preserve complete algorithm semantics in a mini baseline.
5. Build a unified datapath that can schedule these operators.
6. Add quantized, sparse, or approximate variants when they improve hardware cost.
7. Execute the whole Jamba 2.0 Mini style model through this structure.
```

The detailed taxonomy is documented in `docs/operator_taxonomy.md`. The 1:1 algorithm support policy is documented in `docs/algorithm_support_policy.md`.

## Evaluation Plan

The first evaluation uses generated hardware structure rather than board timing:

- baseline parallel operators.
- shared dot/linear fabric.
- shared Mamba and attention fabric.
- hybrid Jamba2 Mini layer and tile.

The reports should compare:

- generated Verilog size.
- module count.
- multiplier proxy count.
- adder proxy count.
- state and cache storage proxy count.
- trends across layer count, context length, and attention period.

After a concrete FPGA board is selected, the same designs should be synthesized to collect LUT, FF, DSP, BRAM, Fmax, and power estimates.

## Current Resource-Reuse Flow

The first automated comparison is:

```bash
./scripts/resource_reuse_analysis.sh
```

It emits small baseline and shared-fabric operators, then writes:

```text
generated/reports/resource_reuse_comparison.md
```

This report is the seed for the paper's resource-reuse table. It currently compares direct `DotProduct`, `Linear4`, serial/shared `Linear4`, generic, semantic, and unified layer-level projection schedulers, `AttentionDecodeTiny`, baseline/shared/semantic-serial `AttentionMixerMini`, direct/shared/serial causal convolution, direct/shared/serial `Jamba2MambaMixerMini`, `MambaStateUpdate`, `SelectiveScanTiny`, serial `SelectiveScanMini`, `TinyMambaBlock`, `TinyJambaBlock`, `DenseMLPMini`, `MoELiteMini`, `MlpPathMini`, and baseline/shared/semantic-serial `Jamba2MiniLayer` implementations. Later reports should add tile-level comparisons.
