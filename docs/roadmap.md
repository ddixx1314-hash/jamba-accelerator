# Roadmap

This roadmap describes how to move from the current teaching prototype toward a more realistic accelerator project.

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

## Stage 3: Better Golden Models

Goal: make Python references closer to the Chisel behavior.

Possible tasks:

- add multi-token stream golden tests
- add clear/reset trace tests
- add attention enabled/disabled comparisons
- add negative-input tests for the full core
- emit JSON traces that Chisel tests can reuse

Acceptance:

- Python tests cover the same scenarios as top-level Chisel tests.
- Chisel expected values can be derived from golden traces.

## Stage 4: More Realistic Fixed-Point Math

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

## Stage 5: Larger Datapath

Goal: move from 4-lane teaching vectors toward more realistic vector widths.

Possible tasks:

- parameterize modules beyond length 4
- add tests for different lengths
- separate compile-time parameters from fixed module names
- consider resource sharing versus full parallelism

Acceptance:

- current 4-lane tests still pass
- at least one larger configuration can elaborate and pass tests

## Stage 6: Weight Storage

Goal: stop passing every weight through top-level IO.

Possible tasks:

- add small ROM-backed demo weights
- add simple SRAM-like register file
- add load interface for weights
- document weight layout

Acceptance:

- demo can run with internal weights
- tests verify weight load/read behavior

## Stage 7: System Interface

Goal: make the accelerator easier to integrate with a larger hardware system.

Possible tasks:

- extend `Jamba2MiniStream`
- add packetized input/output
- add simple command/status registers
- later consider AXI-lite or AXI-stream

Acceptance:

- valid/ready behavior remains tested
- interface timing is documented

## Stage 8: Toward Real Mamba/Jamba

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
