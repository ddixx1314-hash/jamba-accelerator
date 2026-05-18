# Resource Reuse Architecture

## Core Idea

The bottleneck in a Jamba 2.0 Mini style accelerator is not just implementing each operator. The harder problem is reusing limited arithmetic and storage resources across operators with different shapes.

The shared resources are:

- MAC lanes.
- reduction adders.
- state registers.
- KV cache storage.
- weight storage.
- activation buffers.
- control scheduling.

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
MLP has its own linear layers.
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

- `MacLane_ResourceReuse`
- `SharedReduction4_ResourceReuse`
- `DotProduct_Baseline`
- `DotProduct_SharedFabric`
- `Linear4_Baseline`
- `Linear4_SharedFabric`

The multiply and add counts are line-based generated-Verilog proxies. They are useful for early architecture comparison, but they are not a substitute for post-synthesis DSP, LUT, FF, BRAM, timing, or power reports.
