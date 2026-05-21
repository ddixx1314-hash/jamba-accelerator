# Chapter 3: Operator Taxonomy and Hardware Mapping

## 3.1 Jamba2 Mini Operator Taxonomy

The Jamba2 Mini layer implements a hybrid recurrent-attention architecture. Its forward pass
decomposes into three categories of operations:

### 3.1.1 Linear Projections

The layer contains up to **10 linear projections**, each a 4×4 matrix-vector multiply with
optional bias:

| Projection | Used In | Dimensions |
|---|---|---|
| `mambaInput` | Mamba mixer | 4×4 |
| `mambaB` | Mamba mixer | 4×4 |
| `mambaC` | Mamba mixer | 4×4 |
| `Q` | Attention mixer | 4×4 |
| `K` | Attention mixer | 4×4 |
| `V` | Attention mixer | 4×4 |
| `attentionOut` | Attention mixer | 4×4 |
| `mlpGate` | MLP path | 4×4 |
| `mlpUp` | MLP path | 4×4 |
| `mlpDown` | MLP path | 4×4 |

At inference, only one mixer path executes per layer (Mamba or Attention), so at most
7 projections are active per token pass. The choice is fixed at elaboration time via
`Jamba2MiniConfig.attentionLayerPeriod`.

### 3.1.2 Recurrent Operations

Two stateful operators carry hidden state across tokens:

**Causal convolution** (`CausalConv1D`): A 1D depthwise convolution over the `mambaInput`
projection output. The delay buffer (`history`) stores the last `convTaps−1` input vectors.
For `convTaps=4` and `lanes=4`, this is a 4-element delay line per lane.

**Selective scan** (`SelectiveScan`): The SSM state update equation
`state ← A·state + B·x; y = C·state`. The state vector is persistent across tokens. Each
lane maintains one scalar state of `stateWidth` bits.

### 3.1.3 Attention

For attention layers, a scaled dot-product attention computes score logits `S = Q·Kᵀ` and
context vectors `ctx = S·V` over a key-value cache of length `contextLength`. The KV cache
is implemented as a circular buffer per lane.

### 3.1.4 Feed-Forward MLP

After the mixer (Mamba or Attention), a gated MLP applies:
```
hidden = SiLU(gate) ⊙ up
out = mlpDown · hidden
```
where gate, up, and down are the three MLP projections. An MoE-Lite routing layer
(top-1 expert selection) can replace the standard MLP when `enableMoE=true`.

### 3.1.5 Layer Normalization

RMSNorm is applied before each mixer and before the MLP. In the mini prototype, RMSNorm
uses a per-lane scalar weight (no learned shift) and a fixed normalization shift:
`x_norm = (weight · x) >> normShift`.

---

## 3.2 Four-Tier Hardware Mapping

We map the 10-projection layer to hardware at four levels of resource/latency tradeoff.
All numbers are from generated Verilog structural analysis (see Section 6.1).

### Tier 1: Baseline (Parallel)

Each projection is a dedicated combinational multiplier array. All 10 projections and both
recurrent operators are instantiated in parallel. This maximizes throughput (1 cycle/token)
at the cost of the full operator set being replicated in silicon.

```
Jamba2MiniLayer_Baseline:  96 multiply-line proxy, 73 add-line proxy
```

The baseline is the comparison anchor. It corresponds to a direct RTL translation of the
mathematical forward pass with no resource sharing.

### Tier 2: SharedFabric

Each operator is refactored to use a single shared MAC lane (`MacLane`). Projections that
previously used 16 parallel multipliers now serialize: a 4×4 matrix multiply takes **16 MAC
cycles** through one shared lane, controlled by a small FSM.

```
Jamba2MiniLayer_SharedFabric:  69 multiply-line proxy, 59 add-line proxy
```

The mul-proxy reduction (96 → 69) comes from replacing multi-lane parallel multiplier blocks
with single-MAC serial blocks. The module boundaries are explicit in the generated Verilog,
so the SharedFabric design is slightly larger in total file size before synthesis.

Key module: `SerialSharedLinear4` — one `MacLane` reused across 16 cycles per 4×4 projection.

### Tier 3: SemanticSerial

The serial projection units are grouped by mixer type and scheduled in semantic order:
a `SerialMambaProjectionGroup` handles the 3 Mamba projections sequentially, and a
`SerialAttentionProjectionGroup` handles Q, K, V, Out. Each group uses one shared MAC lane,
reducing the total MAC count per mixer from N-parallel to 1.

```
Jamba2MiniLayer_SemanticSerial:  42 multiply-line proxy, 53 add-line proxy
```

This is the lowest mul-proxy tier because each mixer type has its own dedicated (but
minimal) MAC unit. The two mixer types are structurally independent — the Mamba path and
the Attention path each have their own scheduler.

### Tier 4: UnifiedSerial

A single `UnifiedProjectionScheduler4` schedules **all 10 projections** from a unified
slot table. One MAC lane is shared across Mamba, Attention, and MLP projections. The
scheduler walks a priority-encoded slot table, launching `ConfigurableSerialLinear4` for each
pending projection in turn. The default `projectionMacLanes=1` preserves the original
one-MAC behavior; M8-O additionally evaluates `projectionMacLanes=2/4`.

```
Jamba2MiniLayer_UnifiedSerial:  50 multiply-line proxy, 53 add-line proxy
```

The mul-proxy (50) is slightly higher than SemanticSerial (42) because both Mamba and
Attention projection paths are instantiated at elaboration time (the scheduler contains
both slot sets), even though only one path is active per token. However, the compute fabric
— the single `MacLane` — is shared across all 10 slots.

At the **tile level**, `UnifiedJamba2MiniTileScheduler` sequences L layers one at a time,
but currently instantiates **one `UnifiedJamba2MiniLayer` per logical layer** (via
`Seq.tabulate(numLayers)`). This means the compute fabric (MAC lane, projection scheduler)
scales linearly with L in area, while tokens are processed sequentially to avoid
concurrency overhead. The current design avoids pipeline stalls but does not yet achieve
compute-fabric sharing across layers.

`SinglePhysicalLayerTile` (M7-A+B, completed) reduces this to one physical layer
instance, time-multiplexing it across all L layers. A per-layer state file (M7-B) saves
and restores SSM hidden state, causal-conv history, and KV cache on each layer transition,
making MAC count independent of L while preserving full multi-token functional correctness.

---

## 3.3 Key Module Hierarchy

```
UnifiedJamba2MiniFullTile
└── UnifiedJamba2MiniTileScheduler       (layer-dispatch FSM)
    ├── UnifiedJamba2MiniLayer           (layer 0 instance)
    ├── UnifiedJamba2MiniLayer           (layer 1 instance)
    ├── ...                              (one instance per logical layer, sequenced)
    └── UnifiedJamba2MiniLayer           (layer L-1 instance)
        ├── UnifiedProjectionScheduler4  (slot-dispatch for all 10 projections)
        │   └── ConfigurableSerialLinear4 (Mac1/Mac2/Mac4 4×4 projection)
        │       └── MacLane(s) + reduction adder tree
        ├── SerialCausalConvMini         (16-cycle conv, one MacLane)
        ├── SerialSelectiveScanMini      (12-cycle SSM, one MacLaneMixed)
        └── LayeredWeightStoreMini       (per-layer weight bank, flat write interface)
```

---

## 3.4 Generated-Verilog Proxy Metrics

Because synthesis tools are FPGA-board-dependent, we use two structural proxies from the
generated SystemVerilog as pre-synthesis resource estimates:

**Multiply-line proxy**: count of lines containing ` * ` in the flattened `.sv` file. Each
such line corresponds to one structural multiplier instance after FIRRTL lowering. This
counts MAC units present in the RTL, not post-synthesis DSP blocks.

**Add-line proxy**: count of lines containing ` + `. Analogous to the multiply proxy.

**Total register bits**: sum of all `reg [N:0]` bit widths in the file. This is a proxy
for total flip-flop usage and scales directly with quantization precision.

These proxies are structural (not data-dependent) and are computed automatically by
`scripts/resource_reuse_analysis.sh` and `scripts/scale_analysis.sh`. They correlate with
but do not replace post-synthesis LUT/FF/BRAM/DSP reports.

**Limitation**: the multiply-line proxy does not distinguish combinational multipliers from
pipelined DSP-backed multipliers. The ZeroSkip variant (Section 6.4) adds a Mux gate
without removing the structural multiplier, so its proxy is identical to the baseline — the
power saving is dynamic, not structural.
