# Jamba Accelerator

This repository is the **engineering version** of the project. It starts from the stable learning prototype in `/home/dong/jamba-accelerator-learning` and now contains a Chisel **Jamba 2.0 Mini style hardware accelerator prototype**.

A Chisel hardware accelerator project for building from a tiny Jamba/Mamba-like prototype toward a Jamba2-style mini accelerator architecture.

This repository is **not** a production Jamba2 accelerator. It is a verified architecture-level mini prototype for learning, experimenting, generating SystemVerilog, and studying Jamba2-style hardware structure.

## What This Project Can Do

- Run a tiny integer Jamba-like datapath with 4-lane `SInt(8.W)` token vectors and `SInt(32.W)` accumulators.
- Demonstrate RMSNorm approximation, linear projections, causal convolution, Mamba-like state update, selective scan, tiny attention decode, and output projection.
- Provide a simple `valid` / `ready` streaming wrapper around the mini core.
- Define the formal `Jamba2MiniConfig` target and provide the first `Jamba2MiniTile` accelerator shell.
- Run Jamba2-style `Mixer + MLP` hybrid layers through the `Jamba2MiniHybridCore` demo-weight path.
- Expose token stream IO, command/status, debug outputs, and a small weight load/read register file.
- Run a two-token end-to-end `Jamba2MiniTile` demo trace against Python golden values.
- Generate SystemVerilog for the top-level modules.
- Run Chisel tests, Python golden-model tests, and Verilator lint.
- Generate lightweight scale-analysis reports from elaborated Verilog.

## What This Project Cannot Do Yet

- Run real Jamba 2 model weights.
- Load HuggingFace checkpoints.
- Accelerate a real LLM faster than a GPU.
- Support BF16/FP16, AXI, DDR, DMA, large hidden sizes, production MoE, or full softmax attention.
- Deploy directly to an FPGA or ASIC as a complete system.

## Main Hardware Tops

- `Jamba2MiniTile`: the current formal Jamba2 Mini accelerator shell.
- `JambaMiniTile`: the legacy/learning engineering top kept for comparison.
- `Jamba2MiniCore`: the main mini datapath.
- `Jamba2MiniStream`: a token-level valid/ready wrapper around `Jamba2MiniCore`.
- `Jamba2MiniAccelerator`: an earlier simpler top kept for comparison.

High-level datapath:

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

## Repository Structure

```text
src/main/scala/jamba/   Chisel hardware modules organized by subsystem
src/test/scala/jamba/   Chisel/chiseltest unit tests organized by subsystem
python/golden/          Python reference models
python/tests/           Python golden-model tests
scripts/                Test and Verilog generation scripts
docs/                   Architecture, interface, roadmap, and learning notes
generated/              Regenerated Verilog and wave outputs, ignored by git
```

## Quick Start

Check required tools:

```bash
./scripts/check_env.sh
```

Run the full project verification:

```bash
./scripts/run_test.sh
```

This runs:

```text
sbt test
python3 -m pytest python/tests/ -v
sbt "runMain jamba.top.GenerateVerilog"
verilator --lint-only generated/verilog/JambaMiniTile.sv
verilator --lint-only generated/verilog/Jamba2MiniTile.sv
verilator --lint-only generated/verilog/Jamba2MiniAccelerator.sv
verilator --lint-only generated/verilog/Jamba2MiniCore.sv
verilator --lint-only generated/verilog/Jamba2MiniStream.sv
./scripts/scale_analysis.sh
```

Generate SystemVerilog only when you do not need to rerun tests:

```bash
./scripts/generate_verilog.sh
```

Generated files appear under:

```text
generated/verilog/
```

They are ignored by git because they are reproducible from Chisel source.

## Recommended Reading Order

If you are learning Chisel with Verilog background:

1. `Counter`
2. `PE`
3. `DotProduct`
4. `SmallGemm4x4`
5. `VectorOps`
6. `RmsNormStats`
7. `RmsNormApprox`
8. `Linear4`
9. `MambaStateUpdate`
10. `CausalConv1D`
11. `SelectiveScanTiny`
12. `AttentionDecodeTiny`
13. `TinyMambaBlock`
14. `TinyJambaBlock`
15. `Jamba2MiniCore`
16. `Jamba2MiniStream`

## Documentation

- [Architecture](docs/architecture.md)
- [Interface](docs/interface.md)
- [Jamba2 Mini spec](docs/jamba2_mini_spec.md)
- [Jamba2 Mini implementation plan](docs/jamba2_mini_plan.md)
- [Jamba2 Mini accelerator goal](docs/jamba2_mini_accelerator_goal.md)
- [Research plan](docs/research_plan.md)
- [Resource reuse architecture](docs/resource_reuse_architecture.md)
- [Fixed-point policy](docs/fixed_point.md)
- [MoE-lite](docs/moe_lite.md)
- [Weight layout](docs/weight_layout.md)
- [End-to-end demo](docs/demo.md)
- [Scale analysis](docs/scale_analysis.md)
- [Limitations](docs/limitations.md)
- [Release v0.1](docs/release_v0.1.md)
- [Reproducibility](docs/reproducibility.md)
- [Roadmap](docs/roadmap.md)
- [English learning notes](docs/learning_notes.md)
- [中文学习笔记](docs/learning_notes_zh.md)

## v0.1 Status

The current repository state is a complete `v0.1` architecture prototype. The highest-value next work is decoding stored weights into typed core ports, replacing more datapath narrowing with shared fixed-point helpers, and parameterizing beyond the current 4-lane design.
