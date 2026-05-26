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
| **Dynamic projection bypass (M10-D)** | Runtime zero-input detection skips full MAC loop (17 cycles → 1 cycle per bypassed slot) |
| **Fused operator scheduling (M11-F)** | CODA-inspired FSM fusion saves 2 cycles/token (Mamba 158→156; Attention 149→147) |
| **Power-of-two SSM A (M12-P)** | `useShiftA=true` replaces state×A MAC with arithmetic right shift; saves 4 Mamba cycles/token |
| **Sliding window attention (M12-K)** | Samba-style `attentionWindowSize` limits KV context; verified divergence vs full context |
| **Sparse projection (M12-A)** | `columnSkip=true` in `ConfigurableSerialLinear4`; k×lanes+2 cycles for k non-zero input columns |
| **Analytical latency model (M12-C)** | `scripts/latency_model.py`: closed-form T_scheduler, T_mamba, T_attention formulas validated against 5 empirical measurements |
| **M12-K integration (MoE)** | `attentionWindowSize` masking verified in Attention+MoE combined mode; window=1 diverges from full-context through both mixer and MoE MLP |
| **lanes=8 parallelism sweep (M13-L)** | `ConfigurableSerialLinear4` tested at lanes=8, macLanes=1/2/4; correct output + Pareto latency ordering confirmed |
| **Sparse-aware scheduler (M13-S)** | `columnSkip` parameter propagated through `UnifiedProjectionScheduler4`; k=2 sparse slot: 14 cycles vs 21 (dense), saves 7 cycles per slot |
| **SSM lanes=8 scale expansion (M14-C)** | `SerialSelectiveScanMini` validated at lanes=8; standard 25 cy / useShiftA 17 cy; saved=8=lanes confirms N×lanes formula |
| **Launch-state fusion (M14-F)** | `fuseInnerLaunch=true` absorbs launchConv/launchScan/launchAttentionOut into preceding wait states; Mamba −2 cy (158→156), Attention −1 cy (149→148); combined with M11-F: Mamba 154, Attention 146 cy |

**Test coverage**: 256 Chisel tests (29 suites) + 28 Python golden-model tests, all pass.

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

# M12-C analytical latency model
python3 scripts/latency_model.py
```

Generated reports appear in `generated/reports/`.

---

## Primary Modules (Current)

### Core fabric (`jamba.fabric`)

| Module | Role |
|---|---|
| `MacLane` / `MacLaneMixed` | Atomic MAC unit; optional `zeroSkip` |
| `SerialSharedLinear4` | 4×4 matrix-vector multiply, 16 cycles, 1 MAC lane |
| `ConfigurableSerialLinear4` | Same as above but `macLanes=1/2/4`; M8-O Pareto engine; `columnSkip` for sparse input (M12-A) |
| `UnifiedProjectionScheduler4` | Slot-table FSM for all 10 named projection slots |
| `SerialCausalConvMini` | Multi-tap depthwise conv; state save/restore ports |
| `SerialSelectiveScanMini` | SSM recurrent update; state save/restore ports; `useShiftA` for PoT A matrix (M12-P) |
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

### Technical reference
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

> Paper draft chapters (`docs/paper/`) and internal progress documents
> are kept locally and not included in this repository.

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

Tag `m12-advisor-extensions` (2026-05-25). M12 advisor extensions complete:
M12-P (useShiftA PoT A matrix), M12-K (sliding window attention), M12-A (columnSkip
sparse projection), M12-C (analytical latency model). FPGA synthesis (M13) is deferred future work.
