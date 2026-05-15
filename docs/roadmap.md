# Roadmap

This roadmap describes how to move from the current teaching prototype toward a Chisel Jamba2 Mini hardware accelerator prototype.

## Current Stage: Teaching Prototype

Already implemented:

- basic arithmetic modules
- RMSNorm statistics and approximation
- linear projection
- causal convolution
- Mamba-like state update
- tiny selective scan
- tiny attention decode
- Jamba-like mixed block
- mini core top
- valid/ready stream wrapper
- Chisel tests
- Python golden tests
- SystemVerilog generation
- Verilator lint
- English and Chinese learning notes

This stage is useful for learning and validating datapath ideas.

## Stage 1: Engineering Refactor

Goal: turn the copied learning prototype into a maintainable engineering workspace.

Implemented in this stage:

- migrated modules from `basic` into `jamba.*` packages
- added `JambaMiniConfig`
- added formal top `JambaMiniTile`
- kept existing mini-core behavior and tests
- updated Verilog generation and lint to include `JambaMiniTile`

Acceptance:

```bash
./scripts/run_test.sh
```

Chisel tests, Python tests, Verilog generation, and Verilator lint must pass.

## Stage 2: Documentation and Reproducibility

Goal: make the repository easy for another person to understand and run.

Implemented in this stage:

- added `scripts/check_env.sh` for read-only tool/version checks
- documented reproducible setup and generated artifacts in `docs/reproducibility.md`
- kept `./scripts/run_test.sh` as the single full verification command
- clarified that `JambaMiniTile` is the formal top and lower modules are internal engineering layers
- kept the stage lightweight without Docker, Nix, devcontainers, or GitHub Actions

Acceptance:

```bash
./scripts/check_env.sh
./scripts/run_test.sh
git diff --check
git status --short --branch
```

Environment check passes, tests pass, formatting check passes, and git is clean.

## Stage 3: Jamba2 Mini Spec and Config

Goal: formally define the Jamba2 Mini accelerator target before replacing the legacy mini datapath.

Implemented in this stage:

- added `Jamba2MiniConfig`
- added `docs/jamba2_mini_spec.md`
- fixed the formal layer shape as `Mixer + MLP`
- set default sparse attention scheduling to `attentionLayerPeriod = 8`
- documented token-serial SSM scan policy
- documented circular sliding-window KV cache policy
- documented shift-based attention normalization direction
- documented MoE-lite dispatch/combine reservation
- added a Jamba2 3B vs Mini prototype comparison table

Acceptance:

```bash
./scripts/run_test.sh
git diff --check
```

The legacy mini tests still pass, and the new Jamba2 Mini target is documented.

## Stage 4: Golden Trace Infrastructure

Goal: make Python golden traces define Jamba2 Mini layer behavior before major Chisel rewrites.

Possible tasks:

- add `jamba2_mini_layer_step`
- add `jamba2_mini_core_trace`
- add Mamba mixer, attention mixer, and dense MLP golden functions
- record mixer type, residuals, SSM state, KV cache index, and MoE debug fields
- reuse deterministic expected values in Chisel tests

Acceptance:

- Python tests cover Jamba2 Mini deterministic traces.
- Chisel expected values can be derived from golden traces.

## Stage 5: Fixed-Point Policy

Goal: move beyond the current integer placeholder math.

Possible tasks:

- define Q-format conventions
- add rounding policy
- add saturation or explicit wrap policy
- replace `RmsNormApprox` with reciprocal-square-root approximation
- add scale factors to projection outputs

Acceptance:

- every narrowing point is documented
- Python golden model matches fixed-point behavior
- tests cover overflow and negative values

## Stage 6: Mamba Mixer

Goal: implement the Jamba2 Mini Mamba mixer.

Possible tasks:

- add parameterized causal convolution
- add token-serial selective scan
- use the SSM state fixed-point domain
- add Python golden and Chisel tests

Acceptance:

- Mamba mixer tests pass.
- Golden and Chisel behavior match.

## Stage 7: Attention Mixer

Goal: implement the Jamba2 Mini attention mixer.

Possible tasks:

- add Q/K/V projections
- add small circular KV cache
- add GQA-style mini decode
- add shift-based approximate normalization

Acceptance:

- Attention mixer tests pass.
- KV cache wrap behavior is deterministic.

## Stage 8: Dense MLP Integrated Layer

Goal: make every layer use `Mixer + MLP`.

Possible tasks:

- add `DenseMLPMini`
- add `MlpPathMini`
- add `Jamba2MiniLayer`
- reserve dispatch/combine fields for MoE-lite

Acceptance:

- Layer tests cover both Mamba and Attention mixer modes.
- Every formal layer includes MLP.

## Stage 9: Hybrid Core Scheduler

Goal: schedule Mamba and Attention layers with sparse attention.

Possible tasks:

- add formal `Jamba2MiniCore`
- default to `attentionLayerPeriod = 8`
- allow period 4 for test/debug
- expose debug attention layer indices

Acceptance:

- Core trace matches Python golden.

## Stage 10: MoE-Lite Interface and Implementation

Goal: add a small MoE-lite path without large refactors.

Possible tasks:

- add `RouterMini`
- add `ExpertMLPMini`
- add `MoELiteMini`
- use token-serial top-1 routing
- document vectorized dispatch/combine extension path

Acceptance:

- Dense and MoE-lite MLP paths both test cleanly.

## Stage 11: Weight Storage and Load

Goal: stop passing every weight through top-level IO.

Possible tasks:

- add small register-file-backed weight storage
- add weight load valid/ready interface
- document weight map

Acceptance:

- demo can run with loaded internal weights
- tests verify weight load/read behavior

## Stage 12: Jamba2MiniTile Top

Goal: form the formal accelerator top.

Implemented:

- added `Jamba2MiniTile`
- added token stream input/output
- added weight load/read shell
- added command/status
- added debug layer outputs
- generated and linted `Jamba2MiniTile.sv`

Acceptance:

- `Jamba2MiniTile` tests pass.
- Verilator lints generated top.

## Stage 13: End-to-End Demo

Goal: run a complete mini inference trace.

Implemented:

- added deterministic Python tile demo trace
- loaded a weight through the top-level weight shell
- fed two tokens through `Jamba2MiniTile`
- compared registered Chisel outputs against Python golden values
- checked SSM state and attention KV cache debug progress
- documented demo flow in `docs/demo.md`

Acceptance:

- end-to-end demo test passes.

## Stage 14: Scale and Resource Analysis

Goal: make the project useful for architecture exploration.

Implemented:

- added `GenerateScaleSweep`
- added `scripts/scale_analysis.sh`
- generated debug/formal/context/attention-period tile variants
- reported generated Verilog size and simple structural counts
- documented the flow in `docs/scale_analysis.md`

Acceptance:

- sweep script runs without breaking full verification.

## Stage 15: Final Documentation and Release

Goal: package the completed Jamba2 Mini accelerator prototype.

Implemented:

- updated README for v0.1 status
- added `docs/limitations.md`
- added `docs/release_v0.1.md`
- linked architecture, interface, fixed-point, MoE-lite, weight layout, demo, scale-analysis, limitations, and release docs
- tagged `jamba2-mini-accelerator-v0.1`

Acceptance:

```bash
./scripts/check_env.sh
./scripts/run_test.sh
git diff --check
git status --short --branch
```

## Later Work: Larger Datapath

Goal: move from 4-lane teaching vectors toward more realistic vector widths.

Possible tasks:

- parameterize modules beyond length 4
- add tests for different lengths
- separate compile-time parameters from fixed module names
- consider resource sharing versus full parallelism

Acceptance:

- current 4-lane tests still pass
- at least one larger configuration can elaborate and pass tests

## Later Work: System Interface

Goal: make the accelerator easier to integrate with a larger hardware system.

Possible tasks:

- extend `Jamba2MiniStream`
- add packetized input/output
- add simple command/status registers
- later consider AXI-lite or AXI-stream

Acceptance:

- valid/ready behavior remains tested
- interface timing is documented

## Later Work: Toward Real Mamba/Jamba

Goal: reduce the gap between the prototype and real model kernels.

Possible tasks:

- implement a closer selective scan
- add true causal convolution shape
- add multi-head attention structure
- add softmax or an approximation
- add KV-cache-like storage for decode experiments

Acceptance:

- each added feature has a Python golden model
- each feature has Chisel tests
- generated Verilog passes lint

## Not in Scope Until Much Later

Avoid these until the smaller stages are solid:

- full Jamba 2 model support
- real HuggingFace checkpoint loading
- BF16/FP16 production units
- DDR controller
- full DMA
- full AXI4 memory subsystem
- FPGA board bring-up
- ASIC timing closure
- large-context production cache

These are real accelerator-system tasks, not first-stage learning tasks.
