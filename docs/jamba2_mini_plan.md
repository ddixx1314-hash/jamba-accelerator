# Jamba2 Mini Accelerator Plan

This is the long-term implementation plan for the Chisel Jamba2 Mini hardware accelerator prototype.

The final target is an architecture-level hardware prototype. It should reflect the core Jamba2 ideas, but it is not intended to run full Jamba2 3B or Jamba2 Mini production checkpoints.

## Final Target

- Formal top: `Jamba2MiniTile`.
- Every layer uses `Mixer + MLP`.
- Mixer is either Mamba or Attention.
- Default sparse attention ratio is about 1:7 using `attentionLayerPeriod = 8`.
- Dense MLP is implemented first.
- MoE-lite is added later through reserved dispatch/combine boundaries.
- Fixed-point behavior is split by domain: activation, weight, accumulator, SSM state, and KV cache.
- Python golden traces define deterministic expected behavior.
- Chisel generates SystemVerilog and passes Verilator lint.

## Stage 1: Engineering Refactor

Status: complete.

- Moved source from `basic` into `jamba.*`.
- Added `JambaMiniConfig`.
- Added current `JambaMiniTile`.
- Kept existing tests and Verilog generation.

## Stage 2: Reproducibility Base

Status: complete.

- Added `scripts/check_env.sh`.
- Kept `scripts/run_test.sh` as the full verification command.
- Added reproducibility documentation.

## Stage 3: Jamba2 Mini Spec and Config

Status: complete.

- Added `Jamba2MiniConfig`.
- Added `docs/jamba2_mini_spec.md`.
- Defined `Mixer + MLP` layer structure.
- Defined sparse attention schedule.
- Documented token-serial SSM scan, circular KV cache, approximate attention normalization, and MoE-lite reservation.

## Stage 4: Golden Trace Infrastructure

Goal: make Python golden traces the behavior reference for future Chisel implementation.

Status: complete for v0.1.

Implemented:

- Added deterministic Jamba2 mini fixture.
- Added `mamba_mixer_step`.
- Added `attention_mixer_step`.
- Added `dense_mlp_step`.
- Added `jamba2_mini_layer_step`.
- Added `jamba2_mini_core_trace`.
- Added Python tests for Mamba layers, Attention layers, MLP residuals, KV cache index updates, and circular cache wrap behavior.
- Golden core traces now track SSM state, convolution history, and KV cache state per layer.

The deterministic trace helpers are now used by Chisel tests for mixers, layers, core behavior, and the tile demo path.

## Stage 5: Fixed-Point Policy

Goal: define all numeric-domain rules before replacing datapath math.

Status: complete for v0.1 policy and helper coverage.

Implemented:

- Added `FixedPointConfig`.
- Added Chisel `FixedPointMath`.
- Added Python golden fixed-point helpers.
- Defined saturation, away-from-zero rounded shifting, and multiply-rescale behavior.
- Added `docs/fixed_point.md`.

The policy and helper functions are in place. Wider adoption inside every datapath remains later cleanup work.

## Stage 6: Mamba Mixer

Goal: implement the Jamba2 Mini Mamba mixer.

Status: complete for v0.1.

Implemented:

- Added `CausalConvMini`.
- Added token-serial `SelectiveScanMini`.
- Added `Jamba2MambaMixerMini`.
- Added Chisel tests for convolution history, scan state, clear/hold behavior, and deterministic mixer outputs.

Additional fixed-point cleanup is tracked as later work.

## Stage 7: Attention Mixer

Goal: implement the Jamba2 Mini Attention mixer.

Status: complete for v0.1.

Implemented:

- Added `AttentionMixerMini`.
- Added Q/K/V and output projections.
- Added small circular KV cache.
- Added causal decode over the active cache window.
- Added shift-based approximate normalization.
- Added Chisel tests that match deterministic Python golden values across cache wrap.

The current attention mixer is connected into `Jamba2MiniLayer`. A fuller GQA-style interface remains later work.

## Stage 8: Dense MLP Integrated Layer

Goal: make every formal layer contain MLP.

Status: complete for v0.1.

Implemented:

- Added `DenseMLPMini`.
- Added `MlpPathMini`.
- Added `Jamba2MiniLayer`.
- Integrated `RMSNorm -> Mixer -> residual -> RMSNorm -> MLP -> residual`.
- Reserved dispatch/combine outputs for MoE-lite.
- Added Chisel tests for Dense MLP, Mamba-mode layer, and Attention-mode layer.

MoE-lite is now connected through `MlpPathMini`.

## Stage 9: Hybrid Core Scheduler

Goal: schedule Mamba and Attention layers with sparse attention.

Status: complete for v0.1.

Implemented:

- Added `Jamba2MiniHybridCore` as a safe formal scheduler prototype without replacing legacy `Jamba2MiniCore`.
- Schedules layer mixer type using `Jamba2MiniConfig.isAttentionLayer`.
- Uses `Jamba2MiniConfig.debug` for short tests with period 4.
- Exposes `layerUsesAttention`, `layerOutputs`, and `layerStateOut` debug outputs.
- Added Chisel tests for sparse attention scheduling, valid behavior, and clear behavior.

The v0.1 formal path uses `Jamba2MiniHybridCore` while keeping the legacy `Jamba2MiniCore` for comparison.

## Stage 10: MoE-Lite Interface and Implementation

Goal: add a small MoE-lite path without redesigning the layer.

Status: complete for v0.1.

Implemented:

- Added `RouterMini`.
- Added `ExpertMLPMini`.
- Added `MoELiteMini`.
- Connected MoE-lite into `MlpPathMini`.
- Threaded MoE-lite weights through `Jamba2MiniLayer` and `Jamba2MiniHybridCore`.
- Added token-serial top-1 routing tests.
- Added `docs/moe_lite.md`.

More experts and larger MoE schedules remain later work.

## Stage 11: Weight Storage and Load

Goal: stop passing all weights through large direct IO.

Status: complete for the first shell.

Implemented:

- Added `WeightStoreMini`.
- Added valid/ready write interface.
- Added combinational read interface.
- Documented first draft address ranges in `docs/weight_layout.md`.
- Added tests for write/read, overwrite, and clear-preserves-weights behavior.

Stored-weight decoding into typed core ports remains the main post-v0.1 integration task.

## Stage 12: Jamba2MiniTile Top

Goal: form the formal accelerator top.

Status: complete for the first formal shell.

Implemented:

- Added `Jamba2MiniTile`.
- Added token stream input/output with one-entry output buffering.
- Added `start`, `clear`, and `enableMoE` controls.
- Added `busy`, `done`, and `error` status outputs.
- Added weight write/read shell backed by `WeightStoreMini`.
- Connected internal `Jamba2MiniHybridCore` with deterministic demo weights.
- Exposed debug layer schedule, selected expert, layer states, and layer outputs.
- Updated Verilog generation and lint to include `Jamba2MiniTile.sv`.
- Added `Jamba2MiniTileSpec`.

Stored-weight decoding into typed core inputs remains post-v0.1 work.

## Stage 13: End-to-End Demo

Goal: run a complete deterministic mini inference trace.

Status: complete for the first demo-weight trace.

Implemented:

- Added `jamba2_mini_tile_demo_trace` to the Python golden model.
- Modeled the current Chisel-visible timing for registered output, SSM state update, and attention write-through behavior.
- Added Python tests for the two-token tile demo trace.
- Extended `Jamba2MiniTileSpec` with a two-token end-to-end demo.
- Verified weight shell write/read survives `clear`.
- Compared Chisel registered outputs against Python golden values.
- Checked SSM state and attention KV cache debug progress.
- Added `docs/demo.md`.

A larger fixture can be added after stored weights drive the core.

## Stage 14: Scale and Resource Analysis

Goal: make the project useful for architecture exploration.

Status: complete for lightweight generated-Verilog analysis.

Implemented:

- Added `GenerateScaleSweep`.
- Added `scripts/scale_analysis.sh`.
- Generated multiple `Jamba2MiniTile` variants for context length, layer count, and attention-period comparisons.
- Reported generated Verilog bytes, line count, module count, assign count, and register declaration count.
- Added `docs/scale_analysis.md`.

Remaining:

- Add synthesis-backed LUT/FF/BRAM/DSP estimates if a stable local synthesis flow is available.
- Sweep wider hidden sizes after the current 4-lane `Linear4` restriction is removed.

## Stage 15: Final Documentation and Release

Goal: package the completed Jamba2 Mini accelerator prototype.

Status: complete for v0.1 after final verification and local tag.

Implemented:

- Updated README for v0.1 status.
- Added `docs/limitations.md`.
- Added `docs/release_v0.1.md`.
- Kept architecture, interface, spec, fixed-point, MoE-lite, weight-layout, demo, scale-analysis, and reproducibility docs linked from README.
- Final verification command is `./scripts/run_test.sh`.
- Release tag is `jamba2-mini-accelerator-v0.1`.

## Final Acceptance

```bash
./scripts/check_env.sh
./scripts/run_test.sh
git diff --check
git status --short --branch
```
