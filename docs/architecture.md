# Architecture

This document explains the current mini accelerator structure and the target Jamba2 Mini accelerator direction.

## Design Intent

The current checked-in design is a small integer datapath for learning and experimentation. It is meant to answer:

- How do neural-network operators map into Chisel modules?
- How do Mamba-like state updates look in hardware?
- How can a Mamba path and a tiny attention path be composed?
- How do we wrap a compute core with simple streaming handshakes?

It is not intended to run production Jamba2 weights. The formal target is a Jamba2 Mini architecture-level accelerator prototype, not a full checkpoint-compatible model.

## Top-Level View

The current formal Jamba2 Mini accelerator shell is `Jamba2MiniTile`. The earlier `JambaMiniTile` remains as a legacy learning comparison top.

```text
upstream token source
 -> Jamba2MiniTile
      -> WeightStoreMini
      -> Jamba2MiniHybridCore
           -> Jamba2MiniLayer repeated by config
 -> downstream consumer
```

`Jamba2MiniTile` owns the formal token valid/ready boundary, command/status signals, weight load/read shell, and debug visibility for the hybrid layers. In v0.1, the weight store is exposed and tested, while the core still uses deterministic demo weights. Decoding the stored weight map into typed core ports is the next integration task.

The legacy learning top still exists:

```text
upstream token source
 -> JambaMiniTile
      -> Jamba2MiniStream
           -> legacy Jamba2MiniCore
 -> downstream consumer
```

`Jamba2MiniStream` and the legacy `Jamba2MiniCore` remain public in source and tests, but new formal integration work should start from `Jamba2MiniTile`.

The main compute layer is `Jamba2MiniCore`.

```text
x
 -> RmsNormApprox
 -> Linear4 input/gate/B/C projections
 -> TinyJambaBlock
      -> TinyMambaBlock
           -> CausalConv1D
           -> SelectiveScanTiny
      -> AttentionDecodeTiny optional path
 -> Linear4 output projection
 -> y
```

The lower-level stream wrapper is `Jamba2MiniStream`.

```text
upstream token source
 -> Jamba2MiniStream
      -> Jamba2MiniCore
 -> downstream consumer
```

`JambaMiniTile` forwards the same simple IO as `Jamba2MiniStream` and is now the legacy comparison point.

The formal `Jamba2MiniTile` uses a Jamba2-style layer structure through `Jamba2MiniHybridCore` and `Jamba2MiniLayer`:

```text
x
 -> RMSNorm
 -> Mixer(Mamba or Attention)
 -> Residual Add
 -> RMSNorm
 -> DenseMLP or MoELite
 -> Residual Add
 -> y
```

The default mixer schedule uses sparse attention with `attentionLayerPeriod = 8`, approximately one attention mixer for every seven Mamba mixers.

## Layer 1: Basic Hardware Blocks

These modules teach basic Chisel and hardware concepts:

- `Counter`: sequential register update.
- `PE`: combinational multiply-accumulate.
- `DotProduct`: parallel multipliers plus reduction.
- `SmallGemm4x4`: small matrix multiply.
- `VectorOps`: lane-wise arithmetic and mux-based ReLU.

These are not Jamba-specific. They are the hardware vocabulary used by later modules.

## Layer 2: Neural-Network Operators

These modules model common neural-network datapath stages:

- `RmsNormStats`: computes sum of squares and integer mean square.
- `RmsNormApprox`: integer-friendly RMSNorm placeholder.
- `Linear4`: 4-lane linear projection with bias.

In a real model, these stages prepare token activations and produce projected vectors for attention, gates, and SSM parameters.

## Layer 3: Mamba/SSM Path

The Mamba-like path is:

```text
token -> CausalConv1D -> MambaStateUpdate -> SelectiveScanTiny -> output
```

Roles:

- `CausalConv1D`: captures local history using the current and previous tokens.
- `MambaStateUpdate`: maintains recurrent state with a simplified update.
- `SelectiveScanTiny`: gates the visible state.
- `TinyMambaBlock`: composes convolution, scan, state projection, and residual gating.

The simplified state update is:

$$
h_{t+1} = a h_t + b x_t
$$

## Layer 4: Attention Path

`AttentionDecodeTiny` models a very small attention decode datapath:

$$
s_r = q \cdot k_r
$$

$$
y_c = \sum_r s_r v_{r,c}
$$

This intentionally omits:

- softmax
- masking
- scaling
- multi-head split/merge
- KV cache memory

The goal is to understand dot-product scores and weighted value sums in hardware.

## Layer 5: Jamba-Like Mixing

`TinyJambaBlock` combines:

- Mamba-like state path
- optional tiny attention path

When `useAttention` is false, output comes from the Mamba path. When `useAttention` is true, attention output is added to the Mamba output.

This is the tiny version of the Jamba idea: combine efficient state-space processing with attention-based context access.

## Layer 6: Core and Stream Wrapper

`Jamba2MiniCore` adds:

- RMSNorm approximation
- input projection
- gate projection
- B/C SSM-like projections
- tiny Jamba block
- output projection
- debug outputs

`Jamba2MiniStream` adds:

- input handshake
- output handshake
- one-entry output buffer
- backpressure handling

`JambaMiniTile` adds:

- formal engineering top-level boundary
- shared `JambaMiniConfig`
- a place to add future tile-level control without disturbing the core

## Layer 7: Formal Jamba2 Mini Shell

`Jamba2MiniTile` adds:

- token input/output valid-ready handshakes
- `start`, `clear`, and `enableMoE` controls
- `busy`, `done`, and `error` status outputs
- weight write/read shell backed by `WeightStoreMini`
- debug outputs for layer mixer schedule, selected expert, layer state, and layer outputs
- internal `Jamba2MiniHybridCore` execution using deterministic demo weights

## Important Simplifications

- Vector length is fixed at 4.
- Input data is `SInt(8.W)`.
- Accumulators are `SInt(32.W)`.
- RMSNorm is an integer approximation.
- Attention does not include softmax.
- The formal top has a small load/read weight shell, but stored weights are not yet decoded into all typed core ports.
- There is no AXI, DDR, DMA, or FPGA board integration.

These simplifications keep the project small enough to understand and verify.
