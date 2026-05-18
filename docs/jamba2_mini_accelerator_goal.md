# Jamba 2.0 Mini Accelerator Goal

## Goal

The long-term goal of this repository is to build a Chisel-based Jamba 2.0 Mini style accelerator prototype.

This is an architecture and learning prototype, not a production Jamba 2 implementation. The design keeps the tensor shapes small so the hardware dataflow, generated SystemVerilog, tests, and FPGA direction remain inspectable.

## Research Position

The central research question is:

```text
Which hardware resources can be reused across different sequence-model operators?
```

The advisor-driven method is to first extract common and non-common operators from each layer, then design a unified hardware structure that supports those operators and executes the whole mini model.

The target operators come from Transformer, Mamba, Samba, and Jamba-style models:

- Transformer attention: dot-product scores and value accumulation.
- Mamba selective SSM: causal convolution, input-dependent state update, and gated output.
- Samba-style hybrid blocks: Mamba plus sliding-window attention.
- Jamba-style hybrid blocks: Mamba, attention, MLP, and MoE-lite composition.

## Mini Hardware Scope

The first complete accelerator target uses:

- 4 activation lanes.
- `SInt(8.W)` activations and weights.
- `SInt(32.W)` accumulators.
- integer or fixed-point arithmetic before BF16/FP16.
- token-stream valid/ready interfaces.
- small on-chip weight storage.
- generated SystemVerilog as the FPGA-ready artifact.

The mini scope is intentional. It lets each operator be implemented one-to-one against its formula before scaling or adding board-specific interfaces.

## Main Hardware Path

The formal accelerator path is:

```text
Jamba2MiniTile
 -> WeightStoreMini
 -> Jamba2MiniHybridCore
    -> Jamba2MiniLayer repeated by config
       -> RMSNorm
       -> Mixer: Mamba or Attention
       -> Residual Add
       -> RMSNorm
       -> MLP or MoE-lite
       -> Residual Add
```

Legacy modules remain useful as baselines, but new integration work should start from `Jamba2MiniTile`.

## FPGA Direction

The FPGA target is intentionally staged:

1. Generate clean SystemVerilog.
2. Pass Verilator lint.
3. Produce structural resource reports.
4. Add Vivado or Quartus synthesis once the board and toolchain are fixed.
5. Add board-level IO, constraints, and demos after the synthesis flow is stable.

The project should not add AXI, DDR, DMA, or a large memory system until the mini accelerator behavior and resource-reuse experiments are stable.
