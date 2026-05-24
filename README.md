# Jamba Accelerator

A Chisel RTL prototype of a **unified resource-reuse fabric** for Jamba2-style hybrid
sequence model inference (Mamba + Attention + MoE). The project studies the
resource-latency tradeoff of sharing one MAC lane across heterogeneous operator families,
measured via pre-synthesis structural proxy analysis of generated SystemVerilog.

This is **not** a production Jamba2 accelerator. It is an architecture-level mini
prototype (lanes=4, INT8 data path, 4×4 weight matrices) for hardware design
experimentation, Chisel learning, and paper-baseline evaluation.

---

## Key Contributions

| Milestone | Description |
|---|---|
| **Four-tier framework** | Baseline / SharedFabric / SemanticSerial / UnifiedSerial: mul-proxy 96 → 69 → 42 → 50 |
| **UnifiedProjectionScheduler4** | One slot-table FSM schedules all 10 projections across Mamba, Attention, MLP, and MoE |
| **SinglePhysicalLayerTile (M7-A+B)** | One physical layer serves L logical layers; instance-weighted proxy O(L) → O(1); per-layer SSM/KV state virtualized |
| **projectionMacLanes sweep (M8-O)** | macLanes=1/2/4 Pareto: Context8 resource +6.5%, latency −53% |
| **Quantization + zero-skip** | INT4/6/8 mul-proxy invariant; zero-skip MAC reduces dynamic power for sparse activations |

**Test coverage**: 219 Chisel tests (69 suites) + 28 Python golden-model tests, all pass.

---

## Repository Structure

```text
src/main/scala/jamba/
  common/     Config, fixed-point helpers, SignedMath
  math/       Combinational PE, DotProduct, GEMM, Linear4 (learning primitives)
  norm/       RMSNorm approximation
  mamba/      Combinational Mamba/SSM/CausalConv blocks
  attention/  Combinational attention decode and mixer
  moe/        Router and expert MLP
  core/       Early hybrid core compositions
  fabric/     Resource-reuse implementations (shared MAC, serial schedulers, unified layer)
  memory/     Layered weight store, address generator, sequential loader
  stream/     Token-streaming wrapper
  top/        Tile-level shells and Verilog generators

src/test/scala/jamba/   Unit tests mirroring the main hierarchy
python/golden/          Python reference models
python/tests/           Python golden-model cross-checks
scripts/                Analysis and Verilog generation scripts
docs/                   Paper chapters, roadmap, learning notes, technical references
generated/reports/      Generated resource/latency analysis reports (tracked)
generated/verilog/      Generated SystemVerilog (git-ignored, reproducible)
```

---

## Quick Start

```bash
# Check required tools (sbt, scala, python3, pytest)
./scripts/check_env.sh

# Run all Chisel tests
sbt test

# Run Python golden-model tests
python3 -m pytest python/tests/ -v

# Generate SystemVerilog for key modules
sbt "runMain jamba.top.GenerateVerilog"

# Resource-reuse structural analysis (four-tier + quantization)
bash scripts/resource_reuse_analysis.sh

# Scale analysis (context length + layer count sweep)
bash scripts/scale_analysis.sh

# M8-O projection MAC parallelism Pareto sweep
bash scripts/optimization_sweep.sh
```

Generated reports appear in `generated/reports/`.

---

## Primary Modules (Current)

### Core fabric (`jamba.fabric`)

| Module | Role |
|---|---|
| `MacLane` / `MacLaneMixed` | Atomic MAC unit; optional `zeroSkip` |
| `SerialSharedLinear4` | 4×4 matrix-vector multiply, 16 cycles, 1 MAC lane |
| `ConfigurableSerialLinear4` | Same as above but `macLanes=1/2/4`; M8-O Pareto engine |
| `UnifiedProjectionScheduler4` | Slot-table FSM for all 10 named projection slots |
| `SerialCausalConvMini` | Multi-tap depthwise conv; state save/restore ports |
| `SerialSelectiveScanMini` | SSM recurrent update; state save/restore ports |
| `UnifiedJamba2MiniLayer` | Full Jamba2 layer (Mamba / Attention / MoE) with unified scheduler |

### Tile level (`jamba.top`)

| Module | Role |
|---|---|
| `UnifiedJamba2MiniFullTile` | L-instance tile with layered weight store (reference baseline) |
| `SinglePhysicalLayerTile` | **Main contribution**: 1 physical layer, L logical layers, full state virtualization |
| `UnifiedJamba2MiniAcceleratorTile` | Single-layer shell with token + weight interface |

### Memory (`jamba.memory`)

| Module | Role |
|---|---|
| `LayeredWeightStoreMini` | Flat-write / typed-read per-layer weight register file |
| `WeightAddressGenMini` | `(layer, field, element)` → flat address |
| `SequentialWeightLoaderMini` | BRAM-style sequential field loader |

---

## Recommended Reading Order

**Learning path** (Chisel/hardware concepts):
```
Counter → PE → DotProduct → Linear4 → RmsNormApprox
→ MambaStateUpdate → CausalConv1D → SelectiveScanTiny
→ SerialSharedLinear4 → SerialCausalConvMini → SerialSelectiveScanMini
```

**Architecture path** (resource-reuse contributions):
```
UnifiedProjectionScheduler4
→ UnifiedJamba2MiniLayer
→ UnifiedJamba2MiniFullTile (L-instance baseline)
→ SinglePhysicalLayerTile  (O(1) tile, state virtualization)
```

---

## Documentation

### Paper chapters
- [Abstract](docs/paper_abstract.md)
- [Ch1 Introduction](docs/paper_ch1_introduction.md)
- [Ch2 Background & Related Work](docs/paper_ch2_background.md)
- [Ch3 Operator Taxonomy](docs/paper_ch3_operator_taxonomy.md)
- [Ch4 Architecture](docs/paper_ch4_architecture.md)
- [Ch5 Implementation](docs/paper_ch5_implementation.md)
- [Ch6 Evaluation](docs/paper_ch6_evaluation.md)
- [Ch7 Conclusion](docs/paper_ch7_conclusion.md)

### Project reference
- [Progress summary (中文)](docs/current_progress_summary_zh.md)
- [Development roadmap (中文)](docs/thesis_roadmap_zh.md)
- [Source file overview (中文)](docs/src_file_overview.md)
- [Weight layout](docs/weight_layout.md)
- [Latency budget](docs/latency_budget.md)
- [Interface spec](docs/interface.md)

### Learning notes
- [English learning notes](docs/learning_notes.md)
- [中文学习笔记](docs/learning_notes_zh.md)

### Generated reports
- [Optimization sweep (M8-O Pareto)](generated/reports/optimization_sweep.md)
- [Resource reuse comparison](generated/reports/resource_reuse_comparison.md)
- [Scale analysis](generated/reports/scale_analysis.md)
- [Sparsification analysis](generated/reports/sparsification_comparison.md)

---

## Current Status

Tag `m8o-complete` (2026-05-21). All structural proxy experiments complete;
FPGA synthesis (M9/M10) is deferred future work.
