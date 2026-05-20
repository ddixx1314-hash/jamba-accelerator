# Chapter 1: Introduction

## 1.1 Motivation

Large language models based on the Transformer architecture have demonstrated strong
performance across a wide range of sequence modeling tasks. However, the standard
self-attention mechanism scales quadratically with sequence length in both computation
and memory, creating a practical barrier for long-context inference on resource-constrained
hardware such as FPGAs.

Selective state space models (SSMs), in particular the Mamba architecture, offer
a recurrent formulation with linear-time inference that is well-suited to sequential
hardware deployment. Recent hybrid architectures such as Jamba and Jamba2 interleave
Mamba layers with sparse self-attention layers and mixture-of-experts (MoE) feed-forward
blocks. These hybrid designs achieve competitive model quality while preserving Mamba's
inference efficiency at long sequence lengths.

From a hardware design perspective, hybrid models present a distinct challenge: the three
operator families — Mamba (causal convolution + selective scan), attention (QKV projection
+ score computation + KV cache), and MoE (router + expert MLP projections) — have
heterogeneous compute requirements. A naive RTL implementation allocates dedicated
multiplier arrays for each operator, replicating compute fabric across all three families
and wasting hardware area. The alternative — sharing compute hardware across operator
families — requires explicit scheduling that is not present in standard operator-by-operator
RTL.

## 1.2 Problem Statement

The central question this work addresses is:

> **Can a single unified MAC lane serve all linear projection operators across the Mamba,
> Attention, and MoE-lite operator families in a Jamba2-style layer, and what is the
> resource-latency tradeoff of doing so?**

This question is studied at the mini-prototype scale (lanes=4, 4×4 weight matrices,
INT8 data path) using generated SystemVerilog structural analysis as the resource metric.
No FPGA synthesis results are included; all resource figures are pre-synthesis structural
proxies derived from the generated RTL (see Chapter 3 §3.4 for metric definitions).

## 1.3 Contributions

This work makes the following contributions:

1. **Operator taxonomy for Jamba2-style layers** (Chapter 3): We classify the operators
   in a Jamba2 Mini layer as *shared* (linear projections, RMSNorm, residual add) and
   *dedicated* (causal convolution, selective scan, attention score, SiLU gating). The
   10 linear projections across Mamba, Attention, and MLP paths are identified as the
   dominant compute operators and the primary candidates for hardware sharing.

2. **Four-tier hardware mapping framework** (Chapter 3): We implement and compare four
   tiers of resource-latency tradeoff for the Jamba2 Mini layer:
   - *Baseline*: dedicated parallel multiplier per projection (96 mul-proxy, 1 cycle/token)
   - *SharedFabric*: one MAC lane per operator (69 mul-proxy, ~16 cycles per projection)
   - *SemanticSerial*: one MAC lane per mixer type (42 mul-proxy)
   - *UnifiedSerial*: one MAC lane across all 10 projections (50 mul-proxy)

3. **`UnifiedProjectionScheduler4`** (Chapter 4): A slot-table FSM that schedules all
   10 linear projections from a single unified slot table, sharing one `MacLane` across
   Mamba, Attention, and MLP projection slots. This is the most unified projection-scheduling
   design, using one scheduler and one MAC lane across all three operator families at the
   layer level. (SemanticSerial achieves a lower per-layer mul-proxy of 42 by keeping
   separate MAC lanes per mixer type; UnifiedSerial trades a slightly higher count of 50
   for a single unified scheduler that enables tile-level sharing.)

4. **Quantization analysis** (Chapter 6 §6.2): A sweep across INT4, INT6, and INT8 data
   widths confirms that the multiply-line proxy (structural MAC count) is constant across
   precisions — quantization changes operand width, not the number of multipliers. Total
   total register bits scale roughly linearly with precision width — INT4 is about half of
   INT8 (6,104 vs 12,168 bits) — correlating with expected
   flip-flop savings at synthesis time.

5. **Zero-skip sparsification** (Chapter 6 §6.4): A `zeroSkip` parameter added to
   `MacLane` and `MacLaneMixed` inserts a comparator and Mux that bypasses the multiply
   when either operand is zero. The structural mul-proxy is unchanged (the multiplier
   remains in the RTL); the benefit is dynamic power reduction for sparse activation
   patterns.

6. **Chisel prototype with 208 tests** (Chapter 5): A complete Chisel RTL prototype
   covering all four tiers, `SinglePhysicalLayerTile` (M7-A), the memory subsystem
   (LayeredWeightStoreMini, address generator, sequential weight loader), and MoE expert
   weight decode. All 208 Chisel tests and 28 Python golden-model tests pass.

## 1.4 Scope and Limitations

This work focuses on a mini-scale prototype sufficient to demonstrate the resource-reuse
architecture and measure structural proxies. The following are explicitly out of scope
for the current prototype:

- **No FPGA synthesis**: mul-proxy and add-proxy are structural estimates from generated
  SystemVerilog, not post-synthesis LUT/FF/DSP counts.
- **Tile-level MAC sharing (M7-A, structure proof only)**: `SinglePhysicalLayerTile`
  achieves a constant instance-weighted mul-proxy (~92 for Context8) regardless of the
  number of layers (Section 6.3.3). Per-layer SSM state and KV cache are not yet
  virtualized between logical layers; full multi-token correctness requires M7-B
  (state-file save/restore).
- **Mini parameter scale**: lanes=4, weight matrices are 4×4, context length up to 16.
  Results demonstrate resource trends, not production-scale throughput.
- **Approximate attention**: the KV score normalization uses a right-shift approximation
  rather than softmax; this is sufficient for structural and functional verification.

## 1.5 Document Organization

- **Chapter 2** reviews the key algorithmic components: Mamba/SSM, Transformer attention,
  Jamba hybrid architecture, MoE-lite, fixed-point quantization, and FPGA accelerator design patterns.
- **Chapter 3** presents the operator taxonomy and four-tier hardware mapping, with
  generated-Verilog structural proxy metrics and the module hierarchy.
- **Chapter 4** describes the unified fabric architecture: the slot-table scheduler,
  weight subsystem, zero-skip MAC variant, and multi-layer sequencing.
- **Chapter 5** details the Chisel implementation: module inventory, key module descriptions,
  test strategy, and the relationship between Chisel constructs and generated Verilog proxies.
- **Chapter 6** presents the evaluation: four-tier resource comparison, quantization sweep,
  context/layer scale analysis, and sparsification analysis.
- **Chapter 7** summarizes findings, states limitations, and outlines future work toward
  a true single-physical-layer fabric and FPGA synthesis.
