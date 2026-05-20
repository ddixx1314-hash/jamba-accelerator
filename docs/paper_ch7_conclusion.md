# Chapter 7: Conclusion

## 7.1 Summary of Contributions

This work presents a Chisel prototype of a unified resource-reuse fabric for Jamba2-style
hybrid sequence model inference, targeting the resource-latency tradeoff inherent in
hardware implementations of models that mix Mamba, Attention, and MoE operators.

The main contributions are:

**Operator taxonomy**: We identified that the 10 linear projections across the Mamba,
Attention, and MoE-lite paths are structurally identical 4×4 integer matrix-vector
multiplies, making them the primary candidates for hardware sharing. Dedicated operators
(causal convolution, selective scan, attention KV score) require separate units with
small MAC lanes, but the projection operators dominate the MAC count.

**Four-tier hardware mapping**: We implemented and compared four resource-latency tiers
for the Jamba2 Mini layer (lanes=4, contextLength=4):

| Tier | Mul-proxy | Latency |
|---|---|---|
| Baseline (parallel) | 96 | 1 cycle/token |
| SharedFabric (1 MAC/op) | 69 | ~16 cycles/projection |
| SemanticSerial (1 MAC/mixer) | 42 | ~556 cycles/token |
| UnifiedSerial (1 MAC/layer) | 50 | ~556 cycles/token |

The UnifiedSerial tier achieves a 48% reduction in structural mul-proxy compared to
Baseline, at the cost of ~556 cycle latency per token. SemanticSerial has the lowest
per-layer mul-proxy (42) but does not share the compute fabric across layers. UnifiedSerial
has a slightly higher per-layer count (50) because both Mamba and Attention slot sets are
elaborated simultaneously, but enables tile-level fabric sharing.

**Instance-weighted resource analysis**: We identified a critical distinction between
file-level and instance-weighted mul-proxy metrics for multi-layer designs. The
file-level `grep -cF ' * '` counts each module definition once; for a 4-layer
`UnifiedFullTile`, this gives 82 (one definition of `UnifiedJamba2MiniLayer`), while the
instance-weighted proxy correctly reports 368 (4 × ~92 per layer instance). The
instance-weighted metric is the correct hardware area surrogate.

**Quantization and sparsification**: The mul-proxy is invariant to quantization precision
(INT4/INT6/INT8): the number of structural multipliers does not change when operand width
changes. Total register bits scale roughly linearly with precision width: INT4 is approximately
half of INT8 (6,104 vs 12,168 bits), though the per-step reductions (−25% INT8→INT6,
−33% INT6→INT4) are not equal. This reflects narrower state and accumulator registers. Zero-skip sparsification adds a comparator Mux
without removing the structural multiplier; the dynamic power benefit for sparse activations
is not visible in structural proxies.

**Memory subsystem**: A layered weight storage design (`LayeredWeightStoreMini`) decodes
a flat write address space into per-layer typed register banks for all weight fields,
including the six MoE expert weight fields. A sequential load path (`SequentialWeightLoaderMini`
→ `SequentialWeightCaptureMini` → `FieldWeightBufferMini`) provides an alternative
BRAM-style field loading interface.

## 7.2 Key Numerical Results

| Experiment | Key Result |
|---|---|
| Four-tier resource comparison | Mul-proxy: 96 (Baseline) → 69 (Shared) → 42 (Semantic) → 50 (Unified) |
| Tile-level (4L, Context8) | File-level: 82; instance-weighted: 368 (= 4 × 92) |
| SinglePhysicalLayerTile (M7-A+B) | Instance-weighted proxy constant ~92 (Context8) regardless of L; 4L: 368 → 92, 8L: 1,248 → 156; M7-B adds per-layer state file for multi-token correctness |
| Quantization (INT4–INT8) | Mul-proxy: constant at 82 (Context8); reg bits: 6,104 → 12,168 |
| Context length sweep | Mul-proxy grows ~linearly: 50 (ctx4) → 82 (ctx8) → 146 (ctx16) |
| Latency budget | Tier 4 layer: ~143 cycles; 4-layer tile: ~556 cycles |
| Test suite | 211 Chisel tests, 28 Python tests; all pass |

## 7.3 Limitations

**No post-synthesis results**: All resource figures are structural proxies from generated
SystemVerilog. The mul-proxy counts SystemVerilog lines containing ` * `, not synthesized DSP blocks. A
synthesis tool may share or eliminate multipliers through constant propagation and
resource sharing that are invisible at the RTL level.

**State virtualization implemented (M7-B)**: `SinglePhysicalLayerTile` achieves constant
instance-weighted mul-proxy (~92 for Context8, independent of L) and now correctly
saves/restores per-layer SSM hidden state, conv history, and KV cache between logical layers
via a per-layer state file. Multi-token functional correctness is verified by direct comparison
against `UnifiedJamba2MiniFullTile` (§6.3.3). The remaining limitation is that resource
figures are structural proxies only, not post-synthesis results.

**Mini parameter scale**: lanes=4, 4×4 weight matrices. Resource trends are demonstrated
but throughput and efficiency numbers do not extrapolate directly to production-scale
dimensions (e.g., 4096-dimensional hidden state, 32 layers).

**Approximate attention**: KV score normalization uses a right-shift approximation instead
of softmax. This is sufficient for structural analysis and functional testing but would
require a proper softmax implementation for accuracy-sensitive deployment.

**No real sparsity injection**: The zero-skip analysis is structural. No activation
statistics from a real model are used to estimate actual zero-skip rates.

## 7.4 Future Work

**M7-A (completed)**: `SinglePhysicalLayerTile` replaces the L-instance
`UnifiedJamba2MiniTileScheduler` with one physical `UnifiedJamba2MiniLayer` and a
tile-level FSM that sequences through all L logical layers. `LayeredWeightStoreMini`
already handles per-layer weight selection via its `activeLayer` input. This reduces
the instance-weighted mul-proxy from ~92L to a constant ~92 (Context8), confirmed
by direct comparison in §6.3.3.

**M7-B (completed)**: per-layer state virtualization. A state file (one entry per logical
layer) saves/restores SSM hidden state, conv history, and KV cache on each layer transition.
The `restoreState` FSM phase drives all `loadState`/`loadHistory`/`loadKvState` signals for
one cycle before `launchLayer`, correctly reconstructing each layer's runtime context. The
state-file cost (L × (SSM state + conv history + KV cache) bits) is sub-linear in area
compared to replicating the full compute fabric L times. Multi-token correctness verified
by 2-token trace comparison against `UnifiedJamba2MiniFullTile` (211 Chisel tests, all pass).

Additional future milestones:

- **M8: BRAM-style weight/state memory** — replace register-file assumptions with
  `SyncReadMem` to enable BRAM inference in Vivado synthesis. The `SequentialWeightLoadPathMini`
  infrastructure provides a starting point for the address-generation side.

- **M9: Verilator lint and synthesis-ready cleanup** — resolve any synthesis warnings
  from FIRRTL lowering (undriven wires, latch inference risks) before submitting to
  Vivado.

- **M10: FPGA synthesis and board demo** — run Vivado synthesis on a Xilinx Ultrascale+
  or Alveo device, collect LUT/FF/DSP/BRAM/Fmax reports, and compare DSP counts across
  the four tiers. This provides the post-synthesis resource comparison needed to validate
  the structural proxy against actual FPGA resource usage.

A secondary research direction is **algorithm-hardware co-design**: adjusting the Jamba2
attention period or MoE sparsity to match the hardware's latency budget, or exploring
wider data paths (BF16, FP8) using Xilinx DSP cascade mode for higher model quality at
the same hardware cost.
