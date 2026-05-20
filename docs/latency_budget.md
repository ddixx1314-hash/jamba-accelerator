# Latency Cycle Budget

This document derives clock-cycle latency for each serial module and hardware tier.
Counts are from FSM inspection and are verified by `LatencyBudgetSpec`.
All counts assume default parameters (lanes=4, taps=4, contextLength=4, dataWidth=8).

---

## Baseline Tier: 1 Cycle

All Baseline modules are fully combinational. Any operation completes in one clock cycle
(or is a purely wire-level computation). This is the resource-maximum, latency-minimum tier.

---

## Atomic Serial Units

### `SerialSharedLinear4` — **16 cycles**

One MAC lane reused across a 4×4 matrix multiply.

```
FSM: (row, col) iterate nested 0..3 × 0..3
Cycles = lanes × lanes = 4 × 4 = 16
```

Source: [SerialSharedLinear4.scala](../src/main/scala/jamba/fabric/SerialSharedLinear4.scala)

### `SerialCausalConvMini` (taps=4, lanes=4) — **16 cycles**

One MAC lane reused across all tap-lane pairs.

```
FSM: (lane, tap) iterate 0..3 × 0..3
Cycles = lanes × taps = 4 × 4 = 16
```

Source: [SerialCausalConvMini.scala](../src/main/scala/jamba/fabric/SerialCausalConvMini.scala)

### `SerialSelectiveScanMini` (lanes=4) — **12 cycles**

One `MacLaneMixed` reused across 3 operations per lane.

```
Op schedule per lane: op0 (recurrent), op1 (state update), op2 (output)
Cycles = 3 × lanes = 3 × 4 = 12
```

Source: [SerialSelectiveScanMini.scala](../src/main/scala/jamba/fabric/SerialSelectiveScanMini.scala)

---

## Projection Scheduler

### `UnifiedProjectionScheduler4` (N slots) — **3 + 16×N cycles**

The scheduler searches for the next pending slot (`findSlot`, 1 cycle), launches it (`launch`,
1 cycle), waits for `SerialSharedLinear4` to complete (16 cycles), then repeats. A final
`findSlot` with no remaining slots transitions to done (1 cycle).

```
Overhead per slot transition: ≈1 cycle (findSlot+launch amortized)
Linear compute per slot: 16 cycles
Formula: 3 + 16×N  (3 = launch overhead + done-detect cycle)
```

| Slot count N | Cycles |
|---|---|
| 1 | 19 |
| 2 | 35 |
| 3 | 51 |
| 4 | 67 |
| 10 | 163 |

Mamba projection (3 slots: mambaInput, mambaB, mambaC) = **51 cycles**  
Attention projection (3 slots: Q, K, V) = **51 cycles**  
Out projection (1 slot each) = **19 cycles**  
MLP gate+up (2 slots) = **35 cycles**  
MLP down (1 slot) = **19 cycles**

Source: [UnifiedProjectionScheduler4.scala](../src/main/scala/jamba/fabric/UnifiedProjectionScheduler4.scala)

---

## Mixer Level

### `SerialMambaMixerMini` — **~79 cycles**

Sequential stages:

| Stage | Cycles | Source |
|---|---|---|
| Mamba projection (3 slots) | 51 | `UnifiedProjectionScheduler4` |
| Causal convolution | 16 | `SerialCausalConvMini` |
| Selective scan | 12 | `SerialSelectiveScanMini` |
| FSM transitions | ~0 | (pipelined overlap) |
| **Total** | **~79** | |

Source: [SerialMambaMixerMini.scala](../src/main/scala/jamba/fabric/SerialMambaMixerMini.scala)

### `SerialAttentionMixerMini` — **~69 cycles**

Sequential stages:

| Stage | Cycles | Source |
|---|---|---|
| Q projection (1 slot) | 19 | `UnifiedProjectionScheduler4` |
| K projection (1 slot) | 19 | |
| V projection (1 slot) | 19 | |
| Attention score + value | 1 | combinational |
| Out projection (1 slot) | 19 | |
| FSM transitions | ~5 | |
| **Total** | **~82** | |

Note: score/value computation is purely combinational (one-cycle wire).
Effective throughput is bounded by the 4 serial projections.

Source: [SerialAttentionMixerMini.scala](../src/main/scala/jamba/fabric/SerialAttentionMixerMini.scala)

---

## Layer Level

### `UnifiedJamba2MiniLayer` — **~135–143 cycles**

The layer runs one mixer path (Mamba or Attention), then the MLP path. The MLP portion
uses the same `UnifiedProjectionScheduler4`.

**Mamba path:**

| Stage | Cycles |
|---|---|
| Mamba mixer (proj 51 + conv 16 + scan 12) | 79 |
| First residual (combinational) | 1 |
| MLP gate+up (2 slots) | 35 |
| SiLU-like gate (combinational) | 1 |
| MLP down (1 slot) | 19 |
| FSM transitions | ~8 |
| **Total** | **~143** |

**Attention path:**

| Stage | Cycles |
|---|---|
| Q+K+V projection (3 slots) | 51 |
| Attention score+value (combinational) | 1 |
| Out projection (1 slot) | 19 |
| First residual (combinational) | 1 |
| MLP path (same as above) | 55 |
| FSM transitions | ~8 |
| **Total** | **~135** |

Source: [UnifiedJamba2MiniLayer.scala](../src/main/scala/jamba/fabric/UnifiedJamba2MiniLayer.scala)

---

## Full Tile Level

### `UnifiedJamba2MiniFullTile` (L layers) — **~139×L cycles**

Layers run sequentially via `UnifiedJamba2MiniTileScheduler`. The tile scheduler itself
adds ~3 cycles of overhead (launch, wait, done detect). The average per-layer latency is
~139 cycles (midpoint between Mamba ~143 and Attention ~135).

| Config | Layers | Approx cycles/token |
|---|---|---|
| 2L Mamba | 2 | ~286 |
| 4L debug (mix) | 4 | ~556 |
| 8L formal (mix) | 8 | ~1,112 |

For comparison:

| Tier | Per-token latency (4L) | Mul-line proxy (layer) |
|---|---|---|
| Baseline | 1 cycle | 96 |
| SharedFabric | 16 cycles/proj × ≥6 projs | 69 |
| SemanticSerial | ~556 cycles | 42 |
| UnifiedSerial | ~556 cycles | 50 |

The UnifiedSerial tile uses a **single** `UnifiedJamba2MiniLayer` module time-multiplexed
across all layers. Only per-layer state (SSM hidden state, KV cache) scales with L.
The compute fabric registers are shared.

Source: [UnifiedJamba2MiniFullTile.scala](../src/main/scala/jamba/top/UnifiedJamba2MiniFullTile.scala)

---

## Weight Loading Path

### `SequentialWeightLoaderMini` (one field) — **numElements + 2 cycles**

The loader walks a flat address space: one element captured per cycle in the `running`
state, plus one cycle to transition `idle→running` and one cycle for `done`.

| Field | numElements | Load cycles |
|---|---|---|
| Data vector (bias, 4 lanes) | 4 | 6 |
| Data matrix (weight, 4×4) | 16 | 18 |
| Conv kernel (4 taps × 4 lanes) | 16 | 18 |
| Router weight (4×4) | 16 | 18 |
| Router bias (4 lanes) | 4 | 6 |
| Expert matrix (2 experts × 4×4) | 32 | 34 |
| Expert bias (2 experts × 4 lanes) | 8 | 10 |

### `SequentialWeightLoadPathMini` (one field end-to-end) — **numElements + ~3 cycles**

Chains `SequentialWeightCaptureMini` and `FieldWeightBufferMini`. The buffer accepts one
element per cycle concurrently with the capture, so total latency equals the loader plus
one extra handshake cycle.

### Full tile weight load (all fields, L layers) — **~(18×8 + 6×4 + 34×6 + 10×6) × L + overhead**

For the default config (lanes=4, 2 experts, numLayers=4), sequential loading of all
fields per layer:

| Field type | Count per layer | Cycles each | Subtotal |
|---|---|---|---|
| Data matrices (weights) | 8 | 18 | 144 |
| Data vectors (biases) | 4 | 6 | 24 |
| Expert matrices | 6 | 34 | 204 |
| Expert vectors (biases) | 6 | 10 | 60 |
| **Per-layer total** | | | **~432** |
| **4-layer tile** | | | **~1,728** |

This is the weight-reload cost per inference token (amortized to zero when weights are
static and loaded once at startup).

Source: [SequentialWeightLoadPathMini.scala](../src/main/scala/jamba/memory/SequentialWeightLoadPathMini.scala)

---

## Summary Table

| Component | Cycles | Notes |
|---|---|---|
| Baseline (any op) | 1 | combinational |
| `SerialSharedLinear4` | 16 | 4×4, one MAC lane |
| `SerialCausalConvMini` (4 taps, 4 lanes) | 16 | one MAC lane |
| `SerialSelectiveScanMini` (4 lanes) | 12 | one MacLaneMixed |
| `UnifiedProjectionScheduler4` (1 slot) | 19 | 3+16×1 |
| `UnifiedProjectionScheduler4` (3 slots) | 51 | 3+16×3 |
| `UnifiedProjectionScheduler4` (10 slots) | 163 | 3+16×10 |
| `SerialMambaMixerMini` | ~79 | proj+conv+scan |
| `SerialAttentionMixerMini` | ~82 | 4 proj + score |
| `UnifiedJamba2MiniLayer` (Mamba) | ~143 | mixer + MLP |
| `UnifiedJamba2MiniLayer` (Attention) | ~135 | mixer + MLP |
| `UnifiedJamba2MiniFullTile` (4L) | ~556 | 4 × ~139 avg |
| `UnifiedJamba2MiniFullTile` (8L) | ~1,112 | 8 × ~139 avg |

Cycle counts are structural (deterministic FSM) and do not depend on data values.
They were verified by simulation in `LatencyBudgetSpec`.
