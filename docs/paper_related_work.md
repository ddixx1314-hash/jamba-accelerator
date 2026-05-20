# Chapter 2: Background and Related Work

## 2.1 Transformer Attention and FPGA Acceleration

The dominant approach to sequence modeling uses scaled dot-product self-attention,
which scales quadratically in computation and memory with sequence length. FPGA
implementations such as FTRANS [Li et al., 2020] and FA-BERT [Liu et al., 2021]
exploit structured weight sparsity and INT8 quantization to fit Transformer inference
within FPGA on-chip bandwidth and DSP constraints. A recurring theme is DSP block reuse:
multiple logical multiplications share one physical DSP48 by serializing the operand
stream, trading throughput for area. This work adopts the same serial-MAC philosophy
but applies it across the heterogeneous operators of a hybrid Mamba/Attention/MoE model
rather than to a single attention mechanism.

KV-cache management is a central concern in attention FPGA designs. FPGA inference
engines typically allocate BRAM for the key and value caches, with address generators
tracking the write pointer and valid-entry count per sequence. Our prototype implements
a register-file KV cache with explicit save/restore ports for state virtualization;
BRAM inference (M8) is deferred to future work.

## 2.2 Selective State Space Models (Mamba / S4)

Mamba [Gu and Dao, 2023] is a selective state space model with an input-dependent
recurrent update that enables linear-time sequential inference. Unlike attention, Mamba
does not require a growing KV cache; its state is a fixed-size hidden vector updated
at every token. The recurrent structure maps naturally to hardware: one state-update
multiply-accumulate per state dimension per token, with no sequence-length-dependent
memory growth.

S4 [Gu et al., 2022] and subsequent SSM variants (H3, Hyena) share the same structural
feature: a causal convolution followed by a state recurrence. In hardware implementations,
the causal convolution over a short kernel (4 taps in Jamba2 Mini) is the other major
compute operator alongside the state update. Our `SerialCausalConvMini` and
`SerialSelectiveScanMini` modules implement these two stages as time-multiplexed
serial units.

Hardware accelerators for pure SSM models (e.g., dedicated Mamba inference chips) can
concentrate design effort on the recurrent update and convolution alone. The Jamba2
hybrid model prevents this specialization: any token may require attention or MoE
computation, so the accelerator must support all three operator families efficiently.

## 2.3 Jamba: Hybrid Mamba/Attention Architecture

Jamba [Lieber et al., 2024] and Jamba2 [Team Jamba, 2024] interleave Mamba and
Transformer attention layers at a fixed period (e.g., one attention layer every 8
Mamba layers). This hybrid design achieves competitive language model quality while
preserving Mamba's O(1)-state inference advantage at long sequences: the KV cache
only grows for the attention layers, which are a small fraction of total layers.

From a hardware perspective, the interleaving period is a design parameter. A large
`attentionLayerPeriod` reduces the fraction of attention layers and thus the KV cache
memory requirement, at the cost of some model quality. Our prototype exposes
`attentionLayerPeriod` and `attentionLayerOffset` as configuration parameters,
allowing the physical hardware mapping to be studied at different hybrid ratios.

## 2.4 Mixture of Experts (MoE)

Sparse mixture-of-experts layers [Shazeer et al., 2017; Fedus et al., 2022] replace
a dense feed-forward block with a router that selects one or more expert sub-networks
per token. In Jamba2, MoE is applied to a subset of layers using top-1 routing with
two experts (gate/up/down MLPs per expert), which we call "MoE-lite." The hardware
implication is that the weight store must hold parameters for all experts simultaneously,
and the router output selects which expert's weights are loaded for a given token.

FPGA accelerators for MoE models typically allocate separate weight banks per expert
and use a routing-controlled multiplexer to select the active bank. Our prototype
implements this with `LayeredWeightStoreMini` holding expert weights in separate
register banks, selected by the router output before the projection stage runs.

## 2.5 Fixed-Point Quantization

Fixed-point quantization maps floating-point model weights and activations to low-bit
integer representations (INT8, INT4, INT6) to reduce memory bandwidth, on-chip storage,
and DSP usage. A key property of uniform symmetric quantization is that it changes
operand bit width but not the number of multiply operations: a layer with N multiplies
at INT8 still requires N multiplies at INT4, only with narrower operands.

Our quantization analysis (§6.2) confirms this structural invariance: the structural
mul-proxy (multiply-line count in generated SystemVerilog) is identical across INT4,
INT6, and INT8 configurations, while total register bits scale roughly linearly with
precision width. FPGA synthesis would show DSP usage reduction for narrower precisions
when the synthesis tool infers sub-DSP-width multipliers from LUT logic; this effect
is invisible at the pre-synthesis structural level.

## 2.6 Resource Sharing in ML Accelerators

Resource sharing — reusing one physical arithmetic unit for multiple logical operations
— is a standard technique in area-constrained hardware design. Time-multiplexed DSP
architectures for neural network inference (e.g., [Umuroglu et al., 2017; Blott et al.,
2018] for binarized networks; [Colangelo et al., 2018] for floating-point CNNs) achieve
significant area savings by serializing MAC operations through a shared unit.

For FPGA neural network inference, the tradeoff between parallelism and resource use
is often managed through a "resource factor" or "folding factor" that determines how
many operations share each DSP. Our four-tier framework formalizes this tradeoff for
the heterogeneous Jamba2 operator mix: Baseline (no sharing), SharedFabric (one MAC
per operator), SemanticSerial (one MAC per mixer type), and UnifiedSerial (one MAC
per layer). The key novelty is applying this sharing framework across operator families
that differ structurally (recurrent vs. attention vs. MLP), not just scaling a single
operator type.

## 2.7 Chisel HDL and Structural Proxy Metrics

Chisel [Bachrach et al., 2012] is a hardware construction language embedded in Scala
that generates FIRRTL intermediate representation, which is then lowered to synthesizable
SystemVerilog. Chisel's parameterized module generation (Scala `case class` configs,
`Seq.tabulate` for multi-instance designs) allows systematic exploration of design
variants without manual RTL duplication.

Because post-synthesis FPGA results (LUT/FF/DSP/BRAM/Fmax) require Vivado or Quartus
tool runs that are expensive and tool-version-dependent, we use a structural proxy
metric: the count of lines containing ` * ` in the generated SystemVerilog. This
multiply-line count is a conservatively stable pre-synthesis surrogate for DSP or
LUT-multiplier usage, with the understood limitation that a synthesis tool may share
or constant-propagate multipliers in ways not visible at the RTL level. Chapter 3
§3.4 defines the metric formally and discusses its limitations.

---

*Note: Reference list is placeholder; specific citation details to be filled in from
actual papers during final draft preparation.*
