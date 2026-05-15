# Jamba2 Mini Accelerator Spec

This document defines the target architecture for the Chisel Jamba2 Mini hardware accelerator prototype.

The goal is not to run full Jamba2 3B or Jamba2 Mini production checkpoints. The goal is to build a small, verifiable hardware model that preserves the core structure of Jamba2-style hybrid SSM/Transformer layers.

Public references used for direction:

- AI21 describes Jamba2 as available in 3B and Mini variants, with Mini as an MoE model.
- Hugging Face documents Jamba as a hybrid Transformer-Mamba/MoE model family.
- Hugging Face `JambaConfig` uses `attn_layer_period = 8`, meaning one attention layer per eight layers in the reference-style configuration.

## Mini Profile

The first formal target config is represented by `Jamba2MiniConfig`.

| Field | Mini Default | Meaning |
| --- | ---: | --- |
| `hiddenSize` | 8 | Token hidden dimension for the mini prototype |
| `lanes` | 4 | Processing lanes used by the current hardware base |
| `numLayers` | 8 | Number of mini layers |
| `attentionLayerPeriod` | 8 | One attention mixer per eight layers |
| `attentionLayerOffset` | 7 | Attention layer is the last layer in each period |
| `mambaStateSize` | 8 | Per-layer SSM state size |
| `convTaps` | 4 | Causal convolution tap count |
| `numAttentionHeads` | 2 | Mini query head count |
| `numKvHeads` | 1 | Mini grouped-query KV head count |
| `contextLength` | 16 | Small decode context for hardware tests |
| `mlpExpansion` | 2 | Dense MLP expansion factor |
| `enableMoE` | false | Dense MLP is the default first implementation |
| `numExperts` | 0 | No experts until MoE-lite is enabled |

## Layer Structure

Every formal Jamba2 mini layer has the same high-level shape:

```text
x
 -> RMSNorm
 -> Mixer(Mamba or Attention)
 -> Residual Add
 -> RMSNorm
 -> DenseMLP or MoELite
 -> Residual Add
 -> y
```

The MLP is part of every layer. It is not a separate optional layer.

## Mixer Schedule

The default schedule is sparse attention:

```text
attentionLayerPeriod = 8
```

This gives approximately one attention mixer and seven Mamba mixers per eight-layer block.

The attention rule is:

```text
layerIndex % attentionLayerPeriod == attentionLayerOffset
```

With the default config, layer 7 is the attention layer in an 8-layer mini core.

For faster tests, use:

```text
attentionLayerPeriod = 4
attentionLayerOffset = 3
```

`attentionLayerPeriod = 2` is smoke-test-only and should not be treated as the formal architecture.

## Fixed-Point Domains

The fixed-point plan must separate numeric domains:

| Domain | Config Field | Reason |
| --- | --- | --- |
| Activation | `activationBits` | Regular layer inputs and outputs |
| Weight | `weightBits` | Projection and kernel weights |
| Accumulator | `accumulatorBits` | Dot products and reductions |
| SSM State | `ssmStateBits` | Recurrent state can grow over time |
| KV Cache | `kvCacheBits` | Stored attention keys and values |

The first implementation keeps these fields as config metadata. Later fixed-point stages will define rounding, saturation, and rescale rules.

## SSM Scan Policy

The first `SelectiveScanMini` implementation will use token-serial scan:

```text
one token step -> update SSM state -> emit mixer output
```

This is area-friendly and easier to verify. A future parallel prefix scan engine may replace it without changing the layer-level interface.

## KV Cache Policy

The first attention cache will use a circular sliding window:

```text
write pointer advances each accepted token
when full, the newest token overwrites the oldest token
```

The cache length is `contextLength`. Tests must make the write index and wrap behavior deterministic.

## Attention Normalization Policy

The first attention mixer will not implement production softmax. It will use a deterministic shift-based approximate normalization so that Python golden and Chisel behavior are identical.

The exact shift and saturation rules will be specified in the fixed-point stage before implementation.

## MoE Reservation Policy

The first formal layer uses Dense MLP. The MLP path must still reserve the MoE boundary:

```text
MLP input
 -> optional Dispatch
 -> DenseMLP or ExpertMLP
 -> optional Combine
 -> MLP output
```

The first MoE-lite implementation will be:

- token-serial
- top-1 routing
- 2 or 4 experts

The dispatch/combine interface should leave room for future vectorized routing.

## Jamba2 3B vs This Mini Prototype

| Item | Public Jamba2 Direction | This Mini Prototype |
| --- | --- | --- |
| Purpose | Production language model | Hardware architecture prototype |
| Scale | Billions of parameters | Tiny deterministic test config |
| Context | Long-context model family | 8-16 token hardware demo context |
| Layer Mix | Hybrid SSM/Transformer | Hybrid Mamba/Attention mixers |
| Attention Period | Reference configs use sparse attention, e.g. period 8 | Default period 8, debug period 4 |
| MLP | Present after mixer blocks | Present in every formal mini layer |
| MoE | Jamba2 Mini is MoE | MoE-lite added after dense path |
| Numeric Format | Model-dependent | Fixed-point hardware prototype |
| Goal | Inference quality and deployment | Chisel verification and Verilog generation |

## Current Implementation Status

The checked-in hardware still contains the earlier `JambaMiniTile` and tiny datapath for legacy comparison. The formal Jamba2 mini path is now built around `Jamba2MiniTile`, `Jamba2MiniHybridCore`, Jamba2 mini mixers, integrated MLP, MoE-lite, and weight storage shell components.

The Python golden model now includes deterministic Jamba2 mini trace helpers for:

- Mamba mixer steps
- Attention mixer steps with circular KV cache behavior
- Dense MLP steps
- full `Mixer + MLP` layer steps
- multi-token sparse-attention core traces

The Chisel source now includes the first Jamba2 mini Mamba mixer building blocks:

- `CausalConvMini`
- `SelectiveScanMini`
- `Jamba2MambaMixerMini`

These modules implement the initial token-serial Mamba mixer path and are kept separate from the older tiny teaching modules.

The Chisel source also includes the first attention mixer building block:

- `AttentionMixerMini`

It implements Q/K/V projection, circular KV cache update, causal decode over the active cache window, and deterministic shift-based approximate normalization.

The Chisel source now includes the first formal layer building blocks:

- `DenseMLPMini`
- `MlpPathMini`
- `Jamba2MiniLayer`

`Jamba2MiniLayer` follows the target `Mixer + MLP` structure and reserves MoE dispatch/combine outputs while running the dense MLP path.

The first hybrid core scheduler is implemented as `Jamba2MiniHybridCore`. It keeps the legacy `Jamba2MiniCore` intact while introducing the sparse attention schedule and multi-layer `Jamba2MiniLayer` composition.

The first MoE-lite path is implemented as `RouterMini`, `ExpertMLPMini`, and `MoELiteMini`. `MlpPathMini` can now select either Dense MLP or token-serial top-1 MoE-lite using the previously reserved dispatch/combine boundary.

The first formal accelerator shell is implemented as `Jamba2MiniTile`. It provides token valid/ready IO, `start`/`clear`/`enableMoE` command controls, `busy`/`done`/`error` status, debug outputs, and a `WeightStoreMini` load/read shell. The Stage 13 demo runs a two-token trace against `jamba2_mini_tile_demo_trace` and checks output, SSM state, and attention KV cache progress. The internal hybrid core still uses deterministic demo weights; decoding the stored weight map into typed core ports is the next integration step.
