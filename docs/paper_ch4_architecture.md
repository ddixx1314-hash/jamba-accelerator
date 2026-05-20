# Chapter 4: Unified Fabric Architecture

## 4.1 Design Goal

The unified fabric targets one primary objective: **minimize the number of structural
multipliers in the generated RTL while preserving correct token-level semantics for all
three operator families** (Mamba, Attention, MoE-lite).

The secondary objective is a clean separation between the compute fabric (shared MAC lane)
and the control/weight fabric (slot-table FSM + layered weight store), so that each can
be analyzed and extended independently.

The key insight enabling this architecture: the 10 linear projections across Mamba (3),
Attention (4), and MLP/MoE (3) are all 4×4 integer matrix-vector multiplies with bias.
They differ only in their weight matrices and in where their inputs and outputs are
connected in the data path. A single serial MAC that can be pointed at different weight
matrices in sequence serves all 10 projections.

## 4.2 SharedFabric: One MAC Lane Per Operator

The `SerialSharedLinear4` module implements a 4×4 matrix-vector multiply over 16 clock
cycles using a single `MacLane`. The `MacLane` accumulates:

```
acc = 0
for col in 0..4:
  for row in 0..4:
    acc[row] += x[col] * weight[row][col]
```

One `MacLane` is shared across all 16 inner-loop iterations; the FSM advances `col` and
`row` each cycle.

In the SharedFabric tier, each operator (each projection) gets its own `SerialSharedLinear4`
instance. For the Mamba mixer, three separate `SerialSharedLinear4` modules handle the
three Mamba projections. This reduces the mul-proxy from 16 (Baseline, parallel multipliers)
to 1 per operator, but the total MAC count at the layer level is still proportional to
the number of distinct operator instances.

## 4.3 UnifiedSerial: One MAC Lane Across All 10 Projections

The `UnifiedProjectionScheduler4` advances beyond SharedFabric by sharing one
`SerialSharedLinear4` across all 10 projection slots. It does this with a slot-table FSM:

```
slotTable = Seq(
  Slot(MAMBA_INPUT, mambaInputWeight, mambaInputBias, xReg),
  Slot(MAMBA_B,     mambaBWeight,     mambaBBias,     xReg),
  Slot(MAMBA_C,     mambaCWeight,     mambaCBias,     xReg),
  Slot(Q_PROJ,      qWeight,          qBias,          norm1Reg),
  Slot(K_PROJ,      kWeight,          kBias,          norm1Reg),
  ...
)
```

The FSM walks the slot table, enabling `SerialSharedLinear4` for each slot in turn and
routing the correct weight matrix and input vector. Intermediate results are stored in
named registers (`projInputReg`, `projBReg`, `projCReg`, `projQReg`, etc.) between slots.

**Why the mul-proxy is 50, not 42**: the slot table contains entries for both Mamba and
Attention projections, because both path types are present in the elaborated module. Only
one path is active per token at runtime, but both sets of weight connections are structurally
present in the generated SV. This is why the UnifiedSerial mul-proxy (50) is slightly
higher than the SemanticSerial mul-proxy (42), which has separate modules per mixer type
that do not both appear in the same instantiation hierarchy.

## 4.4 Module Hierarchy

The full hierarchy from top to bottom:

```
UnifiedJamba2MiniFullTile
└── LayeredWeightStoreMini          (weight register file, flat write, typed read)
└── UnifiedJamba2MiniTileScheduler  (layer-dispatch FSM)
    ├── UnifiedJamba2MiniLayer       (layer 0 instance)
    │   ├── UnifiedProjectionScheduler4  (slot-dispatch for all 10 projections)
    │   │   └── SerialSharedLinear4      (16-cycle 4×4 multiply)
    │   │       └── MacLane             (1 MAC per cycle, optional zeroSkip)
    │   ├── SerialCausalConvMini     (16-cycle conv, one MacLane)
    │   ├── SerialSelectiveScanMini  (12-cycle SSM, one MacLaneMixed)
    │   └── UnifiedMoEPathMini       (router + top-1 expert dispatch)
    ├── UnifiedJamba2MiniLayer       (layer 1 instance)
    │   └── ...
    └── UnifiedJamba2MiniLayer       (layer L-1 instance)
        └── ...
```

**Important**: the current `UnifiedJamba2MiniTileScheduler` uses `Seq.tabulate(numLayers)`
to create **one physical `UnifiedJamba2MiniLayer` instance per logical layer**. Tokens
are processed sequentially through all L layers (layer 0 first, then layer 1, etc.).
This means the compute fabric (MAC lane, projection scheduler) is replicated L times in
area, even though only one layer is active per token. The instance-weighted mul proxy
grows as ~92L for Context8 configs.

The planned `SinglePhysicalLayerTile` (Section 4.7, not yet implemented) will reduce
this to one physical instance with per-layer state/weight multiplexing.

## 4.5 Weight Subsystem

### 4.5.1 LayeredWeightStoreMini

`LayeredWeightStoreMini` is a register-file-backed weight storage module with a flat
write interface and typed per-layer read outputs. The flat address space uses a fixed
stride of 512 per layer:

```
physical_address = layer_index × 512 + local_offset
```

Each layer occupies local offsets 0–365 (366 addresses); offsets 366–511 are reserved.
The full address map is documented in `docs/weight_layout.md`.

On the read side, `LayeredWeightStoreMini` decodes the flat store into typed output
ports for each weight field. The tile-level scheduler selects the active layer's decoded
outputs and routes them to the layer's IO ports.

### 4.5.2 Expert Weight Decode

MoE-lite adds six weight fields per layer per expert (gate/up/down weight and bias for
each of two experts). These are decoded by `LayeredWeightStoreMini` and wired through
`connectLoadedWeights` when `useLoadedWeights=true`. When `useLoadedWeights=false`,
`connectDemoExpertWeights` provides deterministic identity-matrix fixtures.

### 4.5.3 Sequential Weight Loader

`SequentialWeightLoaderMini` provides an alternative BRAM-style loading path: given a
`(layer, field)` pair, it emits one address per accepted cycle in address order. The
address stream feeds the flat read port of `LayeredWeightStoreMini`, which returns the
element at each address. `SequentialWeightCaptureMini` and `FieldWeightBufferMini` chain
these into a complete field capture path.

## 4.6 Zero-Skip MacLane Variant

When `zeroSkip=true`, `MacLane` inserts a comparator and Mux before the multiplier:

```
skipMul = (io.a === 0.S) || (io.b === 0.S)
result = Mux(skipMul, io.accIn, io.a * io.b + io.accIn)
```

The structural multiplier remains in the generated SV (the `Mux` selects between two
inputs, one of which is the multiply output). The mul-proxy is therefore unchanged.
The benefit is dynamic: when an operand is zero, the multiply's switching activity is
suppressed, reducing dynamic power consumption proportionally to sparsity.

`zeroSkip` is available on `MacLane`, `MacLaneMixed` (used by `SerialSelectiveScanMini`),
and propagates through `SerialSharedLinear4`, `SerialSelectiveScanMini`, and the unified
layer via the `zeroSkipScan` parameter.

## 4.7 Multi-Layer Sequencing and the SinglePhysicalLayerTile Gap

The current `UnifiedJamba2MiniTileScheduler` sequences L layers using L physical
`UnifiedJamba2MiniLayer` instances. The tile-level FSM enables each layer in turn,
waits for its `done` signal, then enables the next layer. Each layer's SSM hidden state
and KV cache are maintained in its own registers across tokens.

This design avoids synchronization overhead: no state needs to be saved and restored
between layers, because each layer has its own state registers. The cost is that
instance-weighted mul count is O(L).

The `SinglePhysicalLayerTile` design (next architectural milestone, not yet implemented)
would reduce this to one physical `UnifiedJamba2MiniLayer` instance shared across all L
logical layers. Per-layer state would be stored in a small state file (one entry per
layer), and the active layer's state would be loaded into the physical layer's registers
at the start of each layer invocation, then saved back when the layer is done.

The expected resource trade-off:

| Metric | Current (L-instance) | SinglePhysicalLayerTile |
|---|---|---|
| Instance-weighted mul-proxy | ~92L (Context8) | ~92 (constant) |
| State file size | 0 (state in layer regs) | L × (SSM state + KV cache) |
| Weight MUX logic | none | 1 L-way mux per weight field |

Whether the state-file and weight-mux overhead is smaller than the replicated-layer
overhead is the key engineering question that motivates `SinglePhysicalLayerTile`.
