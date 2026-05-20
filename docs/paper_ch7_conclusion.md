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
changes. Total register bits scale approximately 50% per 2-bit reduction, reflecting
narrower state and accumulator registers. Zero-skip sparsification adds a comparator Mux
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
| Quantization (INT4–INT8) | Mul-proxy: constant at 82 (Context8); reg bits: 6,104 → 12,168 |
| Context length sweep | Mul-proxy grows ~linearly: 50 (ctx4) → 82 (ctx8) → 146 (ctx16) |
| Latency budget | Tier 4 layer: ~143 cycles; 4-layer tile: ~556 cycles |
| Test suite | 202 Chisel tests, 28 Python tests; all pass |

## 7.3 Limitations

**No post-synthesis results**: All resource figures are structural proxies from generated
SystemVerilog. The mul-proxy counts `SV ` * `` lines, not synthesized DSP blocks. A
synthesis tool may share or eliminate multipliers through constant propagation and
resource sharing that are invisible at the RTL level.

**Tile-level MAC not yet shared**: The current `UnifiedJamba2MiniTileScheduler` creates
one physical `UnifiedJamba2MiniLayer` instance per logical layer. Instance-weighted mul
count is O(L). The contribution of the UnifiedSerial tier is at the *layer* level (one
MAC lane across 10 projections); the *tile* level remains L-proportional.

**Mini parameter scale**: lanes=4, 4×4 weight matrices. Resource trends are demonstrated
but throughput and efficiency numbers do not extrapolate directly to production-scale
dimensions (e.g., 4096-dimensional hidden state, 32 layers).

**Approximate attention**: KV score normalization uses a right-shift approximation instead
of softmax. This is sufficient for structural analysis and functional testing but would
require a proper softmax implementation for accuracy-sensitive deployment.

**No real sparsity injection**: The zero-skip analysis is structural. No activation
statistics from a real model are used to estimate actual zero-skip rates.

## 7.4 Future Work

The most impactful next step is the **SinglePhysicalLayerTile** (M7):

> Replace the L-instance `UnifiedJamba2MiniTileScheduler` with a design that has one
> physical `UnifiedJamba2MiniLayer`. Per-layer state (SSM hidden state, KV cache) is
> stored in a small state file. On each layer invocation, the active layer's state is
> loaded into the physical layer's registers; on completion, state is saved back. Weight
> multiplexing selects the active layer's weight bank.

This would reduce the instance-weighted mul-proxy from ~92L to a constant ~92,
independent of the number of layers. The cost is additional state-file registers and
weight-MUX logic that scales with L in area (but sub-linearly compared to replicating
the full compute fabric L times).

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
