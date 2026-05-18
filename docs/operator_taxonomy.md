# Operator Taxonomy and Unified Datapath

This note captures the advisor's direction: first identify common and non-common operators in each layer, then design reusable hardware resources and data paths that can execute the whole mini model.

The intended 1:1 target is complete algorithm support. The implementation may use mini dimensions, quantization, and sparsification as long as the algorithmic roles and dependencies are represented and compared against a baseline.

## Layer-Level View

The Jamba 2.0 Mini style layer is treated as:

```text
x
 -> RMSNorm
 -> Mixer: Mamba or Attention
 -> Residual Add
 -> RMSNorm
 -> MLP or MoE-lite
 -> Residual Add
 -> y
```

The hardware question is:

```text
Which parts are shared arithmetic/storage patterns, and which parts are operator-specific control or state?
```

## Common Operators

These appear across multiple layer types and should map to reusable hardware.

| Common Operator | Appears In | Hardware Resource |
| --- | --- | --- |
| Dot product | Linear, attention score, router | MAC lanes, reduction |
| Matrix-vector projection | Q/K/V/O, Mamba B/C/input, MLP | MAC fabric, weight buffer |
| Weighted sum | attention value accumulation | mixed-width MAC fabric |
| Element-wise multiply | gates, Mamba update, MLP hidden | lane-wise multiplier |
| Element-wise add | residual, bias, state update | lane-wise adder |
| Reduction sum | RMSNorm stats, dot product | reduction tree |
| Narrow/resize | fixed-point boundaries | signed resize/saturate unit |
| Register buffering | state, history, output | register file / small SRAM |

## Non-Common Operators

These are specific to one model family or layer role.

| Non-Common Operator | Appears In | Why It Is Special |
| --- | --- | --- |
| SSM recurrent state update | Mamba | token-serial state dependency |
| Causal convolution history | Mamba/Samba | sliding activation window |
| KV cache update and read | Attention/Samba/Jamba | context-indexed memory |
| Attention schedule | Hybrid Jamba layer | sparse attention layer selection |
| MoE routing | Jamba MoE-lite | expert selection and dispatch |
| Expert weight selection | MoE-lite | dynamic weight source |
| Softmax or normalization approximation | full attention future work | nonlinear and precision-sensitive |

## Resource Reuse Map

The first reusable hardware resources are:

```text
MacLane
MacLaneMixed
SharedDotProduct
SharedReduction
SharedLinear4
SharedDenseMLPMini
SharedRouterMini
SharedExpertMLPMini
SharedMoELiteMini
SharedMlpPathMini
SharedJamba2MiniLayer
SharedAttentionDecodeTiny
SharedCausalConv1D
SharedMambaStateUpdate
SharedSelectiveScanTiny
SharedTinyMambaBlock
SharedTinyJambaBlock
```

They cover:

```text
Linear / GEMM
Attention scores
Attention value accumulation
Router logits
MoE expert MLPs
Dense-or-MoE MLP path selection
Layer-level Mixer plus MLP composition
Convolution taps
Mamba update arithmetic
Selective scan gate
Mamba block composition
Mamba-plus-attention block composition
MLP projections
```

## Unified Datapath Proposal

The long-term mini accelerator should have this structure:

```text
Input token buffer
 -> Operand select / scheduler
 -> Shared MAC + reduction fabric
 -> Operator-specific post-processing
 -> State / KV / activation buffers
 -> Output / residual path
```

### Shared Datapath

The shared datapath should support:

- vector dot product.
- matrix-vector projection.
- weighted accumulation.
- lane-wise multiply-add.
- reduction sum.

### Operator-Specific State

The non-common state should stay outside the shared arithmetic fabric:

- Mamba state register file.
- causal convolution history buffer.
- attention KV cache.
- MoE router/expert control.

This separation keeps arithmetic resources reusable while preserving each operator's unique behavior.

## Execution Strategy

The first version keeps operators combinational and parallel for clarity. The next resource-reuse version should add time multiplexing:

```text
cycle group 0: load operands
cycle group 1: run shared MAC/reduction
cycle group 2: write result to state/KV/output buffer
cycle group 3: advance scheduler
```

This makes the same fabric execute multiple operator types over time.

## Research Implication

The paper should not only report that individual operators are implemented. It should show:

```text
operator taxonomy
 -> common resource extraction
 -> unified datapath
 -> baseline vs shared resource comparison
 -> full mini model execution through the unified structure
```

This is the path from core operators to a complete Jamba 2.0 Mini style accelerator.

Quantized and sparse versions should be presented as hardware-aware variants of the complete algorithm, with numerical and resource comparisons against the mini baseline.
