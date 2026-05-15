# Jamba Accelerator

This repository is the **next engineering version** of the project. It starts from the stable learning prototype in `/home/dong/jamba-accelerator-learning` and is intended to evolve toward a more realistic Jamba/Mamba-like hardware accelerator.

A Chisel hardware accelerator project for building from a mini Jamba/Mamba-like prototype toward a more complete accelerator architecture.

This repository is **not yet** a production Jamba 2 accelerator. The current codebase still contains the teaching mini core, but this copy is now the workspace for turning that prototype into a more realistic engineering project.

## What This Project Can Do

- Run a tiny integer Jamba-like datapath with 4-lane `SInt(8.W)` token vectors and `SInt(32.W)` accumulators.
- Demonstrate RMSNorm approximation, linear projections, causal convolution, Mamba-like state update, selective scan, tiny attention decode, and output projection.
- Provide a simple `valid` / `ready` streaming wrapper around the mini core.
- Generate SystemVerilog for the top-level modules.
- Run Chisel tests, Python golden-model tests, and Verilator lint.

## What This Project Cannot Do Yet

- Run real Jamba 2 model weights.
- Load HuggingFace checkpoints.
- Accelerate a real LLM faster than a GPU.
- Support BF16/FP16, AXI, DDR, DMA, large hidden sizes, MoE, or full softmax attention.
- Deploy directly to an FPGA or ASIC as a complete system.

## Main Hardware Tops

- `JambaMiniTile`: the formal first-stage engineering top.
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
verilator --lint-only generated/verilog/Jamba2MiniAccelerator.sv
verilator --lint-only generated/verilog/Jamba2MiniCore.sv
verilator --lint-only generated/verilog/Jamba2MiniStream.sv
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
- [Reproducibility](docs/reproducibility.md)
- [Roadmap](docs/roadmap.md)
- [English learning notes](docs/learning_notes.md)
- [中文学习笔记](docs/learning_notes_zh.md)

## Current Project Goal

The current goal is to provide a clean, understandable, and fully testable mini accelerator foundation. The next major stage would be turning this learning prototype into a more realistic accelerator by adding larger dimensions, fixed-point scaling policy, weight storage, memory interfaces, and more accurate Mamba/attention kernels.
