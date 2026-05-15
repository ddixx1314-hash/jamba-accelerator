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

Status: in progress.

Implemented:

- Added deterministic Jamba2 mini fixture.
- Added `mamba_mixer_step`.
- Added `attention_mixer_step`.
- Added `dense_mlp_step`.
- Added `jamba2_mini_layer_step`.
- Added `jamba2_mini_core_trace`.
- Added Python tests for Mamba layers, Attention layers, MLP residuals, KV cache index updates, and circular cache wrap behavior.
- Golden core traces now track SSM state, convolution history, and KV cache state per layer.

Remaining:

- Reuse selected deterministic expected values in future Chisel tests once the Jamba2 mini hardware modules exist.

## Stage 5: Fixed-Point Policy

Goal: define all numeric-domain rules before replacing datapath math.

Status: in progress.

Implemented:

- Added `FixedPointConfig`.
- Added Chisel `FixedPointMath`.
- Added Python golden fixed-point helpers.
- Defined saturation, away-from-zero rounded shifting, and multiply-rescale behavior.
- Added `docs/fixed_point.md`.

Remaining:

- Reuse the fixed-point helpers inside future Mamba, Attention, MLP, and top-level datapath modules.

## Stage 6: Mamba Mixer

Goal: implement the Jamba2 Mini Mamba mixer.

Status: in progress.

Implemented:

- Added `CausalConvMini`.
- Added token-serial `SelectiveScanMini`.
- Added `Jamba2MambaMixerMini`.
- Added Chisel tests for convolution history, scan state, clear/hold behavior, and deterministic mixer outputs.

Remaining:

- Replace the first integer/narrowing datapath with the shared fixed-point helpers where appropriate.
- Reuse selected Python golden trace values directly in future layer/core tests.

## Stage 7: Attention Mixer

Goal: implement the Jamba2 Mini Attention mixer.

Status: in progress.

Implemented:

- Added `AttentionMixerMini`.
- Added Q/K/V and output projections.
- Added small circular KV cache.
- Added causal decode over the active cache window.
- Added shift-based approximate normalization.
- Added Chisel tests that match deterministic Python golden values across cache wrap.

Remaining:

- Extend from the current mini single-KV-head behavior toward the planned GQA-style interface.
- Reuse the mixer inside the future `Jamba2MiniLayer`.

## Stage 8: Dense MLP Integrated Layer

Goal: make every formal layer contain MLP.

Status: in progress.

Implemented:

- Added `DenseMLPMini`.
- Added `MlpPathMini`.
- Added `Jamba2MiniLayer`.
- Integrated `RMSNorm -> Mixer -> residual -> RMSNorm -> MLP -> residual`.
- Reserved dispatch/combine outputs for MoE-lite.
- Added Chisel tests for Dense MLP, Mamba-mode layer, and Attention-mode layer.

Remaining:

- Reconcile layer timing with future full-core golden traces.
- Replace current dense-only wrapper with MoE-lite when Stage 10 lands.

## Stage 9: Hybrid Core Scheduler

Goal: schedule Mamba and Attention layers with sparse attention.

Status: in progress.

Implemented:

- Added `Jamba2MiniHybridCore` as a safe formal scheduler prototype without replacing legacy `Jamba2MiniCore`.
- Schedules layer mixer type using `Jamba2MiniConfig.isAttentionLayer`.
- Uses `Jamba2MiniConfig.debug` for short tests with period 4.
- Exposes `layerUsesAttention`, `layerOutputs`, and `layerStateOut` debug outputs.
- Added Chisel tests for sparse attention scheduling, valid behavior, and clear behavior.

Remaining:

- Decide when to promote `Jamba2MiniHybridCore` into the formal `Jamba2MiniCore` name.
- Reconcile full-core timing with Python golden traces before top-level integration.

## Stage 10: MoE-Lite Interface and Implementation

Goal: add a small MoE-lite path without redesigning the layer.

Status: in progress.

Implemented:

- Added `RouterMini`.
- Added `ExpertMLPMini`.
- Added `MoELiteMini`.
- Connected MoE-lite into `MlpPathMini`.
- Threaded MoE-lite weights through `Jamba2MiniLayer` and `Jamba2MiniHybridCore`.
- Added token-serial top-1 routing tests.
- Added `docs/moe_lite.md`.

Remaining:

- Extend beyond two experts when the weight storage stage lands.
- Add Python golden MoE-lite trace helpers before larger core-level golden comparison.

## Stage 11: Weight Storage and Load

Goal: stop passing all weights through large direct IO.

Status: complete for the first shell.

Implemented:

- Added `WeightStoreMini`.
- Added valid/ready write interface.
- Added combinational read interface.
- Documented first draft address ranges in `docs/weight_layout.md`.
- Added tests for write/read, overwrite, and clear-preserves-weights behavior.

Remaining:

- Decode stored weights into typed layer/core ports.
- Integrate stored weight decoding into the end-to-end demo path.

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

Remaining:

- Decode stored weights into typed core inputs instead of using demo weights.
- Add an end-to-end trace that loads weights, feeds tokens, and compares against Python golden output.

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

Remaining:

- Decode stored weights into typed core inputs instead of using demo weights.
- Add a larger fixture once the weight map is connected to the core.

## Stage 14: Scale and Resource Analysis

Goal: make the project useful for architecture exploration.

- Add parameter sweep script.
- Report generated Verilog size and module counts.
- Optionally add synthesis estimates if local tools are available.

## Stage 15: Final Documentation and Release

Goal: package the completed Jamba2 Mini accelerator prototype.

- Finish architecture, interface, fixed-point, MoE-lite, weight-layout, demo, scale-analysis, and limitations docs.
- Run full verification.
- Tag `jamba2-mini-accelerator-v0.1`.

## Final Acceptance

```bash
./scripts/check_env.sh
./scripts/run_test.sh
git diff --check
git status --short --branch
```
