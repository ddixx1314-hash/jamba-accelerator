# Chapter 2: Background

## 2.1 Transformer Self-Attention

The Transformer architecture processes a sequence of token vectors using multi-head
self-attention. Given input sequence length N and model dimension D, the attention
operation computes query, key, and value projections:

```
Q = X W_Q,  K = X W_K,  V = X W_V
Attn(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

The dot-product score matrix `Q K^T` is O(N²) in both computation and memory, making
standard attention expensive for long sequences. FPGA implementations of attention
typically exploit the structured sparsity of the score matrix or resort to blockwise
computation to fit within on-chip BRAM.

In this prototype, attention uses a simplified score normalization: the dot-product
scores `S = Q·K^T` are divided by a fixed right-shift instead of the full softmax
exponential. This approximation is sufficient for structural and functional RTL verification.

## 2.2 Selective State Space Models (Mamba/S4)

State space models (SSMs) implement a linear recurrence over the hidden state:

```
h[t] = A h[t-1] + B x[t]
y[t] = C h[t]
```

where `A`, `B`, `C` are parameters and `h[t]` is the per-channel hidden state. In the
structured SSM (S4), `A` is a fixed diagonal or low-rank matrix. In Mamba, `B` and `C`
are input-dependent (selected) at each time step, enabling the model to selectively
compress or pass through information.

From a hardware perspective, the SSM recurrence is a scalar multiply-accumulate per
channel per step, making it naturally amenable to serial implementation. The hidden
state is small (one scalar per channel in the mini prototype) and must persist across
tokens — it is stored in registers, not recomputed.

The Mamba layer additionally applies a causal depthwise convolution to the input
projection before the SSM, using a kernel of `convTaps` weights per channel.

## 2.3 Jamba: Hybrid Mamba + Attention Architecture

Jamba interleaves Mamba and Transformer attention layers in a fixed schedule. In the
Jamba2 design, one out of every seven layers is an attention layer; the remaining six
are Mamba layers. All layers share a common two-norm + residual structure and a
feed-forward MLP block after the mixer.

The key hardware implication is that the hardware must support both recurrent (Mamba)
and attention computation paths, selected per layer at inference time. From a resource
perspective, the linear projection matrices in both paths are structurally identical
4×4 integer multiplies (at mini scale), making them candidates for a shared MAC unit.

In the mini prototype:
- `attentionLayerPeriod = 4` selects which layers use attention vs. Mamba
- The selection is fixed at elaboration time via `Jamba2MiniConfig`
- Both paths share the same projection MAC lane via `UnifiedProjectionScheduler4`

## 2.4 Mixture-of-Experts (MoE-Lite)

The MoE-lite block replaces the dense MLP after the mixer with a top-1 expert routing
layer. A router network selects one of `numExperts` expert MLPs per token:

```
gate_logits = x W_router + b_router
expert_idx  = argmax(gate_logits)
y = expert[expert_idx].mlp(x)
```

Each expert MLP has its own gate, up, and down weight matrices. In the mini prototype,
`numExperts = 2` and each expert is a 4×4 gate/up/down structure. The router output
is a 1-bit index selecting between the two experts.

From a hardware perspective, MoE introduces six additional weight fields per layer
(gate/up/down weight and bias for each expert) that must be stored and selectively
decoded at inference time. The `LayeredWeightStoreMini` and `WeightAddressGenMini`
modules handle this address decoding (see Chapter 5 §5.6).

## 2.5 Fixed-Point Quantization

This prototype uses signed fixed-point arithmetic throughout:
- **Data width**: `dataWidth` bits for weights and activations (default 8, swept over 4, 6, 8)
- **Accumulator width**: `accWidth = 2 × dataWidth` bits for accumulator outputs

Signed N-bit integers represent values in the range `[-(2^(N-1)), 2^(N-1) - 1]`.
For INT4, this is `[-8, 7]`; the asymmetric range (absolute maximum is 8, not 7) is
relevant when checking saturation bounds in tests.

After each projection, the accumulator output is narrowed back to `dataWidth` bits by
dropping the low-order bits (right shift). This approximates the effect of quantization
at each layer boundary. A full saturation-aware quantization pass (clamp before narrow)
is modeled in `python/golden/mamba_ops.py`.

The key quantization result (Section 6.2): reducing data width from INT8 to INT4 halves
the register bit count (proportional to flip-flop usage) while leaving the multiply-line
proxy constant. The number of multipliers does not change; only their bit width changes.

## 2.6 FPGA Accelerator Design Patterns

FPGA implementations of DNN inference typically exploit three resource-sharing strategies:

**DSP tile reuse**: Xilinx DSP48E2 tiles support `27×18` signed multiplications natively.
A single DSP tile can be time-multiplexed across many sequential MAC operations, reducing
the total DSP count at the cost of increased latency.

**Serial MAC with FSM control**: A single MAC unit iterates over matrix rows and columns
controlled by a finite state machine. For a 4×4 matrix multiply, this takes 16 MAC cycles.
The `SerialSharedLinear4` module in this prototype implements this pattern.

**Shared compute with weight multiplexing**: A single MAC unit can serve multiple operators
if the weight inputs are multiplexed in the correct order. This requires a scheduler that
knows when each operator's weights are needed, and registers to hold intermediate results
between operator invocations.

The `UnifiedProjectionScheduler4` module in this prototype implements the third pattern,
scheduling 10 distinct projection operators through a single MAC lane using a slot-table FSM.

## 2.7 Chisel HDL and Structural Verilog Analysis

Chisel (Constructing Hardware in a Scala Embedded Language) generates RTL through the
FIRRTL (Flexible Intermediate Representation for RTL) compiler. FIRRTL lowering flattens
Chisel's parameterized module hierarchy into synthesizable SystemVerilog.

A key property of FIRRTL lowering: each Chisel `Module` class produces one SV `module`
definition, regardless of how many times it is instantiated. This has an important
consequence for resource counting: a `grep -cF ' * '` over the generated SV file counts
each module definition once, even if that module is instantiated 10 times in the hierarchy.

This work uses two structural proxy metrics derived from generated SV (defined in Chapter 3
§3.4):

- **File-level mul-proxy**: `grep -cF ' * '` across the entire file — counts one definition
  per module, independent of instance count
- **Instance-weighted mul-proxy**: a Python graph traversal that multiplies each module's
  own mul count by the number of times it appears in the instantiation hierarchy — the
  correct area surrogate

For flat designs (no sub-module reuse), both metrics agree. For hierarchical designs
with shared sub-modules, the instance-weighted proxy correctly reflects that a sub-module
instantiated N times contributes N × its own resource cost.

## 2.8 Related Work

**FPGA acceleration of Transformer attention.** FTRANS [Li et al., 2020] and FA-BERT
[Liu et al., 2021] exploit structured weight sparsity and INT8 quantization to fit
Transformer inference within FPGA DSP and bandwidth constraints; both use time-multiplexed
DSP tiles — the same serial-MAC philosophy adopted here. A key difference is that those
works optimize a single attention mechanism; this work schedules three heterogeneous
operator families (Mamba, Attention, MoE) through a unified MAC lane.

**SSM / Mamba hardware.** Mamba [Gu and Dao, 2023] and S4 [Gu et al., 2022] are
amenable to hardware because the recurrent state update is a scalar MAC per channel per
token with no sequence-length-dependent memory growth. Dedicated SSM accelerators can
concentrate on the state update and causal convolution alone. The Jamba hybrid model
prevents this specialization: any token may require attention or MoE computation, so the
accelerator must support all three families efficiently.

**Hybrid model accelerators.** Jamba [Lieber et al., 2024] and Jamba2 [Team Jamba, 2024]
interleave Mamba and attention layers at a fixed period, preserving Mamba's O(1)-state
inference advantage while adding periodic attention. Hardware support for the hybrid
schedule requires per-layer mode selection at elaboration time (our `Jamba2MiniConfig`)
or at runtime (deferred to future work). To our knowledge, a systematic resource-latency
tier comparison for unified Mamba/Attention/MoE RTL has not been published for this
architecture family.

**MoE FPGA accelerators.** Sparse MoE [Shazeer et al., 2017; Fedus et al., 2022] reduces
active parameter count per token by routing to one or a few expert sub-networks. FPGA MoE
designs typically allocate separate weight banks per expert and use router-controlled
multiplexing to activate the correct bank. Our `LayeredWeightStoreMini` implements the
same pattern at mini scale.

**Resource sharing in ML accelerators.** Time-multiplexed DSP architectures for DNN
inference [Umuroglu et al., 2017; Blott et al., 2018; Colangelo et al., 2018] show that
serializing MAC operations through a shared unit achieves significant area savings at the
cost of proportional throughput reduction. The key novelty of this work is applying the
sharing framework across operator families that differ structurally (recurrent vs. attention
vs. MLP projection), not just scaling a single operator type.

*Citation metadata (author lists, venues, DOIs) will be completed during final reference
selection.*
