# Resource Reuse Architecture

## Core Idea

The bottleneck in a Jamba 2.0 Mini style accelerator is not just implementing each operator. The harder problem is reusing limited arithmetic and storage resources across operators with different shapes.

Following the advisor's feedback, the design starts by classifying every layer into common operators and non-common operators. Common operators map to reusable arithmetic and storage resources. Non-common operators keep their own state and control.

The shared resources are:

- MAC lanes.
- reduction adders.
- state registers.
- KV cache storage.
- weight storage.
- activation buffers.
- control scheduling.

The operator taxonomy is maintained in `docs/operator_taxonomy.md`.

## Operator-to-Resource Map

| Operator | Main Formula Shape | Reusable Resources |
| --- | --- | --- |
| DotProduct | `sum(a(i) * b(i))` | MAC lanes, reduction |
| Linear / GEMM | matrix-vector dot products | MAC lanes, reduction, weight buffer |
| RMSNorm stats | `sum(x(i)^2)` | multiplier lanes, reduction |
| CausalConv1D | windowed multiply-accumulate | MAC lanes, activation history |
| Mamba state update | `h = a*h + b*x` | MAC lanes, state register file |
| Selective scan | recurrent state plus gate | state register file, MAC lanes |
| Attention score | `q dot k` | MAC lanes, reduction, KV cache |
| Attention value sum | `score * v` accumulation | MAC lanes, reduction, KV cache |
| MLP | linear, activation, linear | MAC lanes, reduction, activation buffer |
| MoE-lite | router plus selected expert MLP | MAC lanes, weight buffer, control scheduler |

## Baseline Versus Shared Fabric

The baseline design keeps each operator mostly independent:

```text
Linear has its own dot products.
Attention has its own score and value accumulation.
CausalConv has its own multiply-add lanes.
Mamba update has its own lane-wise arithmetic.
Mamba mixer has its own input/B/C projections.
Attention mixer has its own Q/K/V/out projections.
MLP has its own linear layers.
MoE has its own router and expert MLPs.
```

The shared design introduces common building blocks:

```text
MacLane
SharedReduction
SharedDotProduct
SharedLinear4
```

Later stages map attention, convolution, Mamba update, and MLP onto the same style of fabric.

## First Fabric Version

The first fabric version is intentionally combinational:

```text
input operands
 -> MacLane chain or reduction network
 -> output result
```

This keeps behavior easy to compare with existing baseline modules. Time-multiplexed scheduling comes later, after correctness and structural comparison are stable.

## FPGA Relevance

On FPGA, MAC-heavy operators compete for DSP blocks and routing. A shared fabric can reduce replicated arithmetic, but it may increase control complexity and latency.

The project should measure this tradeoff in three steps:

1. Baseline parallel hardware.
2. Shared combinational fabric with equivalent behavior.
3. Time-multiplexed fabric with explicit scheduling.

Only after these three steps should the project add board-specific synthesis and timing closure work.

## Generated Report

Run:

```bash
./scripts/resource_reuse_analysis.sh
```

The script generates baseline and shared-fabric operator SystemVerilog into:

```text
generated/resource_reuse/
```

and writes:

```text
generated/reports/resource_reuse_comparison.md
```

The first report compares:

- `AttentionDecodeTiny_Baseline`
- `AttentionDecodeTiny_SharedFabric`
- `CausalConv1D_Baseline`
- `CausalConv1D_SharedFabric`
- `CausalConvMini_SerialSharedFabric`
- `MambaStateUpdate_Baseline`
- `MambaStateUpdate_SharedFabric`
- `SelectiveScanTiny_Baseline`
- `SelectiveScanTiny_SharedFabric`
- `TinyMambaBlock_Baseline`
- `TinyMambaBlock_SharedFabric`
- `TinyJambaBlock_Baseline`
- `TinyJambaBlock_SharedFabric`
- `DenseMLPMini_Baseline`
- `DenseMLPMini_SharedFabric`
- `MoELiteMini_Baseline`
- `MoELiteMini_SharedFabric`
- `MlpPathMini_Baseline`
- `MlpPathMini_SharedFabric`
- `Jamba2MiniLayer_Baseline`
- `Jamba2MiniLayer_SharedFabric`
- `Jamba2MambaMixerMini_Baseline`
- `Jamba2MambaMixerMini_SharedFabric`
- `AttentionMixerMini_Baseline`
- `AttentionMixerMini_SharedFabric`
- `MacLane_ResourceReuse`
- `MacLaneMixed_ResourceReuse`
- `SharedReduction4_ResourceReuse`
- `DotProduct_Baseline`
- `DotProduct_SharedFabric`
- `Linear4_Baseline`
- `Linear4_SharedFabric`
- `Linear4_SerialSharedFabric`
- `MambaProjectionGroup_SerialSharedFabric`
- `AttentionProjectionGroup_SerialSharedFabric`
- `MambaProjectionGroup_SemanticSerial`
- `AttentionProjectionGroup_SemanticSerial`
- `Jamba2LayerProjectionGroup_UnifiedSerial`
- `Jamba2MambaMixerMini_SemanticSerial`
- `Jamba2MiniLayer_UnifiedSerial`
- `SelectiveScanMini_SerialSharedFabric`
- `AttentionMixerMini_SemanticSerial`
- `Jamba2MiniLayer_SemanticSerial`

The shared attention decode maps score calculation to `SharedDotProduct` and weighted value accumulation to `MacLaneMixed`, because attention multiplies accumulator-width scores by data-width values.

The shared causal convolution keeps its operator-specific delay history but maps each tap accumulation to `MacLane` chains. This is the first Mamba/Samba local-history operator in the shared-fabric report.

The shared Mamba state update keeps the recurrent state register file as operator-specific state, while mapping `state * a + x * b` to `MacLaneMixed` plus `MacLane`.

The shared selective scan composes the shared state update with mixed-width gate MACs. This is the first combined Mamba scan operator in the shared-fabric report.

The shared tiny Mamba block composes shared causal convolution, shared selective scan, state projection, and residual gating. It is the first end-to-end Mamba-style path in the optimized track.

The shared tiny Jamba block composes the shared Mamba path with the shared attention decode path. This is the first hybrid Mamba-plus-attention block in the optimized track.

The shared dense MLP maps gate, up, and down projections to `SharedLinear4` and keeps the activation/hidden path as lane-local arithmetic.

`SerialSharedLinear4` is the first explicit time-multiplexed fabric module. It latches one 4x4 projection, reuses one `MacLane` for 16 cycles, and writes four accumulator-width outputs. This gives the project a concrete parallel-vs-serial resource/latency comparison.

`SerialProjectionScheduler4` schedules several `SerialSharedLinear4` operations over the same serial fabric. The 3-projection instance models Mamba input/B/C projection reuse, and the 4-projection instance models attention Q/K/V/out projection reuse.

`SerialMambaProjectionGroup` and `SerialAttentionProjectionGroup` add model-level names around the serial schedules. The attention wrapper keeps Q/K/V tied to the token input and uses a separate input for the final output projection, matching the real attention mixer dataflow.

`UnifiedProjectionScheduler4` generalizes these wrappers into named layer-level projection slots. Each slot can use a different input vector, which is necessary for a real layer because Mamba projections consume `norm1`, attention out consumes the decoded attention value, and MLP down consumes the hidden activation.

`SerialCausalConvMini` reuses one `MacLane` across lanes and taps, then updates causal history after the token's convolution finishes.

`SerialSelectiveScanMini` reuses one mixed-width MAC lane for the recurrent Mamba scan: recurrent state update, input contribution, and output gate are scheduled over lanes. This keeps the full scan recurrence while making the state path time-multiplexed.

`SerialMambaMixerMini` connects the semantic serial Mamba projection group to serial causal convolution and serial selective scan. It reports outputs only after the token has completed the multi-cycle projection, multi-cycle convolution, and scan schedules, so it is a latency/resource comparison point rather than a drop-in same-cycle replacement.

`SerialAttentionMixerMini` applies the semantic serial projection fabric to Q/K/V and output projection while preserving KV-cache semantics. In the current mini implementation, the score/value decode remains combinational, so it is the next attention-side target for deeper reuse.

`SerialJamba2MiniLayer` combines serial Mamba/attention mixers with the existing shared MLP path. This is the first full layer-level semantic-serial datapoint: it supports the mini layer algorithm, but its MLP stage is not yet fully serial.

`UnifiedJamba2MiniLayer` reuses the same `UnifiedProjectionScheduler4` across Mamba/Attention mixer projections and dense MLP projections. This is the first layer-level unified-serial datapoint; MoE-lite routing and experts remain a follow-on integration target.

The shared MoE-lite path maps router logits to shared dot products and maps each expert MLP to `SharedDenseMLPMini`. `SharedMlpPathMini` then preserves the dense-or-MoE selection contract used by the formal Jamba2 mini layer.

The shared Jamba2 Mamba mixer maps the input, B, and C projections to `SharedLinear4` while preserving causal convolution and selective scan state behavior.

The shared attention mixer maps Q, K, V, and output projections to `SharedLinear4` while preserving KV cache update/read behavior.

The shared Jamba2 mini layer uses shared Mamba mixer, shared attention mixer, and shared MLP path modules. This creates the first layer-level baseline/shared comparison with all major linear projection groups mapped to the shared fabric.

The multiply and add counts are line-based generated-Verilog proxies. They are useful for early architecture comparison, but they are not a substitute for post-synthesis DSP, LUT, FF, BRAM, timing, or power reports.
