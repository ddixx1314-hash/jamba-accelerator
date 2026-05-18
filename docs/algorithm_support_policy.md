# Algorithm Support Policy

This document records the clarified advisor guidance for "1:1 implementation".

## Meaning of 1:1

In this project, 1:1 means:

```text
support the complete algorithm semantics
```

It does not mean immediately matching the full production model scale, parameter count, hidden size, or deployment stack.

For a Jamba 2.0 Mini style accelerator, complete algorithm support means the hardware architecture should cover the algorithmic components required by the model family:

- normalization.
- linear projections.
- Mamba/SSM mixer.
- causal convolution or local sequence mixing.
- selective scan/state update.
- attention or sliding-window attention.
- KV cache behavior.
- MLP.
- MoE-lite routing and expert path.
- residual connections.
- layer scheduling.
- weight loading.

The mini implementation may use smaller dimensions, fewer layers, smaller context, and simplified numeric formats while preserving the algorithmic roles and data dependencies.

## Quantization and Sparsification

Quantization and sparsification are allowed and expected as hardware-aware algorithm modifications.

They should be treated as optimized variants, not as abandoning 1:1 support:

```text
complete algorithm baseline
 -> hardware bottleneck analysis
 -> quantized / sparse / approximate variant
 -> compare accuracy or numerical drift against baseline
 -> compare resource, latency, bandwidth, and FPGA feasibility
```

Examples:

| Algorithm Part | Complete Support | Hardware-Aware Variant |
| --- | --- | --- |
| Linear / GEMM | matrix-vector projection | INT8 weights, shared MAC fabric, sparse weights |
| RMSNorm | norm statistics and scaling | fixed-point reciprocal approximation |
| Attention | score, value accumulation, KV cache | sliding window, sparse attention, shift normalization |
| Mamba scan | token-state recurrence | fixed-point state, scheduled shared MAC use |
| MLP | gate/up/down projections | quantized activations, sparse weights |
| MoE | router and selected expert path | top-1 routing, sparse expert activation |

## Baseline and Optimized Tracks

The project should keep two tracks:

```text
Baseline track:
  complete mini algorithm behavior, easy to compare with Python golden models.

Optimized track:
  quantized, sparse, or resource-reused hardware-friendly variants.
```

The paper value comes from comparing these tracks.

## Practical Boundary

The current scope is:

```text
algorithm-complete mini Jamba 2.0 style accelerator
```

not:

```text
full-scale checkpoint-compatible AI21 Jamba2 Mini reproduction
```

If the project later targets real model compatibility, that should be a separate scaling stage after the mini architecture is correct and measurable.
