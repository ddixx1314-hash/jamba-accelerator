# Chapter 5: Chisel Implementation

## 5.1 Module Inventory

The prototype is organized into eight packages. Key modules are listed below with their
source file and primary test file.

### Compute Fabric (`jamba.fabric`)

| Module | Source | Purpose |
|---|---|---|
| `MacLane` | [fabric/MacLane.scala](../src/main/scala/jamba/fabric/MacLane.scala) | Atomic MAC: one `a * b + accIn` per cycle; optional `zeroSkip` |
| `MacLaneMixed` | [fabric/MacLaneMixed.scala](../src/main/scala/jamba/fabric/MacLaneMixed.scala) | Mixed-width MAC for SSM (A×state + B×x); optional `zeroSkip` |
| `SerialSharedLinear4` | [fabric/SerialSharedLinear4.scala](../src/main/scala/jamba/fabric/SerialSharedLinear4.scala) | 4×4 matrix-vector multiply over 16 MAC cycles |
| `SerialProjectionScheduler4` | [fabric/SerialProjectionScheduler4.scala](../src/main/scala/jamba/fabric/SerialProjectionScheduler4.scala) | N-slot projection scheduler (generic slot table) |
| `UnifiedProjectionScheduler4` | [fabric/UnifiedProjectionScheduler4.scala](../src/main/scala/jamba/fabric/UnifiedProjectionScheduler4.scala) | Jamba2-named slot dispatcher for all 10 projections |
| `SerialCausalConvMini` | [fabric/SerialCausalConvMini.scala](../src/main/scala/jamba/fabric/SerialCausalConvMini.scala) | 16-cycle depthwise conv (one MacLane) |
| `SerialSelectiveScanMini` | [fabric/SerialSelectiveScanMini.scala](../src/main/scala/jamba/fabric/SerialSelectiveScanMini.scala) | 12-cycle SSM update (one MacLaneMixed) |
| `SerialMambaMixerMini` | [fabric/SerialMambaMixerMini.scala](../src/main/scala/jamba/fabric/SerialMambaMixerMini.scala) | Full Mamba path (conv + scan + 3 projections) |
| `SerialAttentionMixerMini` | [fabric/SerialAttentionMixerMini.scala](../src/main/scala/jamba/fabric/SerialAttentionMixerMini.scala) | Full attention path (QKV + score + out) |
| `SerialJamba2MiniLayer` | [fabric/SerialJamba2MiniLayer.scala](../src/main/scala/jamba/fabric/SerialJamba2MiniLayer.scala) | Complete layer (both mixer types + MLP) |
| `UnifiedJamba2MiniLayer` | [fabric/UnifiedJamba2MiniLayer.scala](../src/main/scala/jamba/fabric/UnifiedJamba2MiniLayer.scala) | **Core contribution**: unified projection scheduler drives all operators |
| `UnifiedMoEPathMini` | [fabric/UnifiedMoEPathMini.scala](../src/main/scala/jamba/fabric/UnifiedMoEPathMini.scala) | Router + top-1 expert dispatch (unified scheduling) |

### Memory Subsystem (`jamba.memory`)

| Module | Source | Purpose |
|---|---|---|
| `WeightStoreMini` | [memory/WeightStoreMini.scala](../src/main/scala/jamba/memory/WeightStoreMini.scala) | Flat register-file weight store with `readAll` bus |
| `LayeredWeightStoreMini` | [memory/LayeredWeightStoreMini.scala](../src/main/scala/jamba/memory/LayeredWeightStoreMini.scala) | Per-layer banked decode; flat write, typed read |
| `WeightAddressGenMini` | [memory/WeightAddressGenMini.scala](../src/main/scala/jamba/memory/WeightAddressGenMini.scala) | (layer, field, element) → flat address mapping |
| `SequentialWeightLoaderMini` | [memory/SequentialWeightLoaderMini.scala](../src/main/scala/jamba/memory/SequentialWeightLoaderMini.scala) | Ready/valid address stream for one field |
| `SequentialWeightCaptureMini` | [memory/SequentialWeightCaptureMini.scala](../src/main/scala/jamba/memory/SequentialWeightCaptureMini.scala) | Captures element stream from loader + BRAM read port |
| `FieldWeightBufferMini` | [memory/FieldWeightBufferMini.scala](../src/main/scala/jamba/memory/FieldWeightBufferMini.scala) | Accumulates captured elements into typed register bank |
| `SequentialWeightLoadPathMini` | [memory/SequentialWeightLoadPathMini.scala](../src/main/scala/jamba/memory/SequentialWeightLoadPathMini.scala) | Chains Capture + Buffer into one-field load path |

### Top-Level Tiles (`jamba.top`)

| Module | Source | Purpose |
|---|---|---|
| `UnifiedJamba2MiniAcceleratorTile` | [top/UnifiedJamba2MiniAcceleratorTile.scala](../src/main/scala/jamba/top/UnifiedJamba2MiniAcceleratorTile.scala) | Single-layer accelerator shell (token + weight interface) |
| `UnifiedJamba2MiniTileScheduler` | [top/UnifiedJamba2MiniTileScheduler.scala](../src/main/scala/jamba/top/UnifiedJamba2MiniTileScheduler.scala) | Multi-layer sequential dispatcher (one instance per layer) |
| `UnifiedJamba2MiniFullTile` | [top/UnifiedJamba2MiniFullTile.scala](../src/main/scala/jamba/top/UnifiedJamba2MiniFullTile.scala) | Scheduler + LayeredWeightStoreMini + weight/load interface |
| `Jamba2MiniTile` | [top/Jamba2MiniTile.scala](../src/main/scala/jamba/top/Jamba2MiniTile.scala) | Legacy SharedFabric tile (comparison baseline) |

### Verilog Generators

| Script | Source | Output directory |
|---|---|---|
| `GenerateResourceReuseSweep` | [top/GenerateResourceReuseSweep.scala](../src/main/scala/jamba/top/GenerateResourceReuseSweep.scala) | `generated/resource_reuse/` |
| `GenerateScaleSweep` | [top/GenerateScaleSweep.scala](../src/main/scala/jamba/top/GenerateScaleSweep.scala) | `generated/scale/` |

---

## 5.2 MacLane and MacLaneMixed

`MacLane` is the atomic compute primitive. Its IO:

```scala
val a     = Input(SInt(dataWidth.W))
val b     = Input(SInt(dataWidth.W))
val accIn = Input(SInt(accWidth.W))
val out   = Output(SInt(accWidth.W))
```

In every clock cycle: `out = a * b + accIn`. When `zeroSkip=true`:

```scala
val skipMul = (io.a === 0.S) || (io.b === 0.S)
io.out := Mux(skipMul, io.accIn, io.a * io.b + io.accIn)
```

`MacLaneMixed` supports different widths for the two multiplicands (used in the SSM
update where `A` is a data-width value but `state` is accumulator-width). It also
accepts the `zeroSkip` parameter.

In generated SystemVerilog, each `MacLane` elaborates to one line containing ` * `.
This is the basis of the mul-proxy metric.

---

## 5.3 SerialSharedLinear4

`SerialSharedLinear4` iterates a single `MacLane` over all 16 positions of a 4×4
matrix-vector multiply. The FSM has three states:

```
idle → running (16 cycles) → done
```

In the `running` state, a column counter `col` and row counter `row` advance each
cycle. The current weight element `weight(row)(col)` and input element `x(col)` are
fed to the `MacLane`. Accumulator registers `acc(row)` accumulate the partial products.
After 16 cycles, `acc` holds the full matrix-vector product; the FSM transitions to
`done` and asserts `io.outValid`.

The `zeroSkip` parameter is passed through to the `MacLane` instance.

The total cycle count is exactly `lanes × lanes = 16` for the default 4×4 configuration.

---

## 5.4 UnifiedProjectionScheduler4

The slot table is a Scala `Seq` of `ProjectionSlot` case class instances:

```scala
case class ProjectionSlot(
  name:   String,
  weight: Vec[Vec[SInt]],
  bias:   Vec[SInt],
  xSrc:   Vec[SInt]
)

val slots = Seq(
  ProjectionSlot("mambaInput", io.mambaInputWeight, io.mambaInputBias, norm1Out),
  ProjectionSlot("mambaB",     io.mambaBWeight,     io.mambaBBias,     norm1Out),
  ...
)
```

At elaboration time, Chisel unrolls the `Seq` into hardware: one `MuxLookup` per IO
port (`weight`, `bias`, `xSrc`) controlled by the FSM's slot index register `slotReg`.

The FSM:
1. On `start`: reset `slotReg` to 0, enable `SerialSharedLinear4`
2. On `SerialSharedLinear4` done: store result in the named register, advance `slotReg`
3. When `slotReg` reaches `numActiveSlots`: assert `done`

The `numActiveSlots` depends on the layer type (Mamba: 6 projections; Attention: 7
projections; both if useAttention is determined at elaboration time through config).

---

## 5.5 UnifiedJamba2MiniLayer

`UnifiedJamba2MiniLayer` composes all operators for one Jamba2-style layer:

```scala
val projScheduler = Module(new UnifiedProjectionScheduler4(...))
val conv          = Module(new SerialCausalConvMini(...))
val scan          = Module(new SerialSelectiveScanMini(...))
val moePath       = Module(new UnifiedMoEPathMini(...))
val norm1         = Module(new RmsNormApprox(...))
val norm2         = Module(new RmsNormApprox(...))
```

The layer-level FSM sequences these in operator order:
- Mamba path: norm1 → [mambaInput, mambaB, mambaC projections] → conv → scan → norm2 →
  [gate, up projections] → SiLU → [down projection] → residual
- Attention path: norm1 → [Q, K, V projections] → KV write + score → [out projection] →
  norm2 → [gate, up] → SiLU → [down] → residual

The KV circular buffer is implemented as a `Reg(Vec(contextLength, Vec(lanes, SInt)))`.
A write-index register advances each token; reads scan all `contextLength` entries.

---

## 5.6 LayeredWeightStoreMini and Address Decode

`LayeredWeightStoreMini` maintains per-layer register banks for all weight fields. The
layer count and field list are determined at elaboration time by `Jamba2MiniConfig`.

The flat write interface accepts `(addr, data, valid)`. On write, the module decodes
`addr` as:

```
layerIndex = addr / 512
localOffset = addr % 512
```

Then selects the target register bank based on `localOffset` ranges from the address
map (e.g., offsets 16–31 → `mambaInputWeight`). All weight registers are readable
simultaneously via per-layer typed output ports.

`WeightAddressGenMini` implements the inverse mapping: given `(layer, field, row, col)`
it returns the flat address and a flag indicating whether the field uses `accWidth` or
`dataWidth` elements. `SequentialWeightLoaderMini` uses this to emit addresses for all
elements of a requested field in row-major order.

---

## 5.7 Test Strategy

### ChiselTest peek/poke

All modules are tested using the ChiselTest framework (`chiseltest`). Tests poke inputs
cycle by cycle and peek outputs, verifying correct behavior against hardcoded expected
values or Python golden model outputs.

Example test structure for `SerialSharedLinear4`:

```scala
it("should compute a 4x4 matmul in 16 cycles") {
  test(new SerialSharedLinear4(8, 32)) { dut =>
    // poke weights and input
    dut.io.weight.zipWithIndex.foreach { ... }
    dut.io.x.zipWithIndex.foreach { ... }
    dut.io.start.poke(true.B)
    dut.clock.step()
    dut.io.start.poke(false.B)
    // wait for done
    while (!dut.io.outValid.peek().litToBoolean) { dut.clock.step() }
    // verify output
    dut.io.out(0).expect(expectedOut0.S)
  }
}
```

### Python golden model cross-check

`python/golden/mamba_ops.py` implements the same computations in NumPy. Key functions:
- `mamba_step(x, a, b, c, state, data_bits, acc_bits)`: one SSM step with quantization
- `attention_step(q, k, v, kv_cache, data_bits, acc_bits)`: one attention step

The golden model uses the same fixed-point rules (multiply, accumulate in `acc_bits`,
narrow back to `data_bits`), enabling bit-exact comparison with the Chisel output for
small integer inputs.

### Test statistics (as of 2026-05-20)

- 208 Chisel tests across 68 test suites
- 28 Python tests in `python/tests/`
- All pass on `sbt test` and `python3 -m pytest`

---

## 5.8 FIRRTL Lowering and Proxy Metrics

Chisel module hierarchy → FIRRTL → SystemVerilog lowering has several properties
relevant to the structural proxy metrics:

1. **One SV module per Chisel Module class**: a `Module(new SerialSharedLinear4(...))` call
   becomes a SV `module SerialSharedLinear4 ... endmodule` block. If the same class is
   instantiated 10 times, there is one SV module *definition* but 10 *instantiation* lines.

2. **Arithmetic operators lower to SV operators**: `io.a * io.b` lowers to a SV
   `assign out = a * b;` line. The mul-proxy counts these lines.

3. **Parameterized types widen to bit-vectors**: `SInt(8.W)` becomes `wire [7:0]` in SV.
   The bit widths appear in `reg [N:0]` declarations, which the reg-bit proxy counts.

4. **Mux lowers to SV ternary**: `Mux(sel, a, b)` becomes `assign out = sel ? a : b`.
   The zero-skip comparator adds one `|` or `==` line but does not add a ` * ` line,
   which is why the zero-skip mul-proxy equals the non-zero-skip mul-proxy.

These properties are why the structural proxies are conservative estimates: they count
structural multipliers in the RTL, not the synthesized DSP blocks that a tool might infer
after optimization passes.
