## DotProduct

### Function
Computes the signed dot product of two fixed-length vectors: `y = sum(a(i) * b(i))`.

### Role in the Accelerator
Dot product is the basic compute pattern behind GEMM, attention score computation, and many linear projections used in Mamba/Jamba-style accelerators.

### Chisel Concepts
`Module`, `Bundle`, `Input`, `Output`, `Vec`, `SInt`, `Wire`, `for` loop hardware generation, `reduce`, combinational logic.

### Verilog Correspondence
`Input` and `Output` map to module ports, `Vec` maps to packed/unpacked arrays or repeated ports, the loop generates repeated multipliers, and `reduce` builds the adder network.

### Common Pitfalls
- Forgetting that Chisel `for` loops elaborate hardware instead of running at circuit runtime.
- Letting multiply/add widths grow unexpectedly or truncate without making the intended accumulator width clear.
- Expecting combinational logic to need `clock.step()` in the test.

## SmallGemm4x4

### Function
Computes a 4x4 signed matrix multiply: `c(row)(col) = sum(a(row)(k) * b(k)(col))`.

### Role in the Accelerator
Small GEMM is the next step after dot product and is the core pattern behind linear layers, attention projections, and many dense transforms.

### Chisel Concepts
Nested `Vec`, nested `for` loops, repeated multiply-add hardware, combinational matrix outputs.

### Verilog Correspondence
The nested loops elaborate repeated multipliers and adders for each output element, similar to manually instantiating 16 dot-product datapaths.

### Common Pitfalls
- Mixing up `b(row)(col)` with `b(k)(col)`.
- Forgetting that a 4-term sum needs more accumulator width than one product.
- Expecting the nested loops to run over time instead of generating parallel hardware.

## VectorOps

### Function
Computes element-wise vector add, subtract, multiply, and ReLU.

### Role in the Accelerator
Vector operations are used around matrix/SSM kernels for residuals, gates, activations, and simple post-processing.

### Chisel Concepts
Parameterized `Vec`, element-wise combinational assignments, `Mux` for conditional selection.

### Verilog Correspondence
Each vector lane maps to its own add/sub/mul logic, and `Mux` maps to a conditional operator or mux gate.

### Common Pitfalls
- Confusing element-wise multiply with dot product.
- Forgetting ReLU is combinational compare plus mux.
- Using too narrow an output for multiplication.

## RmsNormStats

### Function
Computes integer RMSNorm statistics: sum of squares and floor mean square.

### Role in the Accelerator
RMSNorm needs a sum-of-squares reduction before reciprocal square root and scaling; this module captures that first hardware step.

### Chisel Concepts
Squaring signed inputs, reduction with `reduce`, constant division, combinational outputs.

### Verilog Correspondence
The module is a bank of squarers feeding an adder reduction and a constant divider.

### Common Pitfalls
- Thinking this is full RMSNorm; it only computes the statistics.
- Forgetting negative inputs still produce positive squares.
- Ignoring the accumulator width needed for squared sums.

## MambaStateUpdate

### Function
Updates a small signed SSM state vector with `state := a * state + b * x`.

### Role in the Accelerator
This is the recurrent state update pattern used by tiny Mamba-like scan blocks.

### Chisel Concepts
`RegInit`, `VecInit`, sequential update with `when` / `.elsewhen`, signed multiply-add.

### Verilog Correspondence
The state vector maps to registers, `clear` maps to synchronous clearing logic, and `en` controls whether new next-state values are loaded.

### Common Pitfalls
- Expecting the updated state to appear before a clock edge.
- Forgetting to clear all lanes of a vector register.
- Truncating recurrent products without choosing an accumulator width.

## CausalConv1D

### Function
Computes a three-tap per-channel causal convolution using the current token and two delayed tokens.

### Role in the Accelerator
Mamba-style blocks often use a local causal convolution before the selective scan path.

### Chisel Concepts
Delay registers, per-lane combinational multiply-add, synchronous enable and clear.

### Verilog Correspondence
The two delay vectors map to shift-register stages, while the three taps map to parallel multipliers feeding adders.

### Common Pitfalls
- Updating delay registers before understanding which cycle the output observes.
- Mixing up tap order: current, previous, previous-previous.
- Forgetting that `en` should hold the delay state.

## SelectiveScanTiny

### Function
Wraps the state update and applies an element-wise gate to the visible state.

### Role in the Accelerator
This approximates the Mamba selective scan datapath in a small integer-friendly form.

### Chisel Concepts
Module instantiation, connecting child IO, vectorized combinational post-processing.

### Verilog Correspondence
The child module is like an instantiated Verilog submodule, and the gate is a bank of multipliers.

### Common Pitfalls
- Reading registered state as if it updated combinationally.
- Confusing the state output with the gated output.
- Letting gate multiplication overflow silently.

## AttentionDecodeTiny

### Function
Computes tiny dot-product attention scores and weighted value sums without softmax.

### Role in the Accelerator
Jamba-like blocks mix SSM-style computation with attention-style decode computation.

### Chisel Concepts
Nested `Vec`, nested loops, dot-product reductions, matrix-vector style combinational logic.

### Verilog Correspondence
The loops generate repeated multipliers and adders for every score and output lane.

### Common Pitfalls
- Treating this as full attention; it intentionally omits softmax and scaling.
- Mixing up key rows and value rows.
- Under-sizing the accumulator for score times value.

## TinyMambaBlock

### Function
Combines causal convolution, tiny selective scan, state projection, and residual gating.

### Role in the Accelerator
This is a compact Mamba-like block suitable for early accelerator prototyping.

### Chisel Concepts
Composing modules, vector wires, registered state, combinational residual output.

### Verilog Correspondence
This maps to several submodules wired together with explicit vector buses.

### Common Pitfalls
- Forgetting the convolution and scan state each advance on clock edges.
- Assuming this is numerically equivalent to production Mamba.
- Narrowing convolution output back to input width without considering range.

## TinyJambaBlock

### Function
Selects between a Mamba-only path and a Mamba-plus-tiny-attention path.

### Role in the Accelerator
This gives a small prototype of the Jamba idea: combine SSM-style sequence processing with attention-style decode.

### Chisel Concepts
`Mux`, child module composition, shared vector inputs, exposing debug scores/state.

### Verilog Correspondence
The attention path and Mamba path are parallel hardware blocks, with a mux selecting or adding the active result.

### Common Pitfalls
- Thinking `useAttention` prevents attention hardware from existing; it only controls the output mux.
- Forgetting both paths are combinationally evaluated.
- Not exposing internal state when debugging sequential behavior.

## Jamba2MiniAccelerator

### Function
Top-level wrapper around `TinyJambaBlock` with enable, clear, output state, attention scores, and valid.

### Role in the Accelerator
This is the first runnable top-level mini accelerator block in the project.

### Chisel Concepts
Top-level module wrapping, IO forwarding, registered valid signal.

### Verilog Correspondence
The wrapper is a Verilog top module that instantiates the block and registers a one-cycle valid flag.

### Common Pitfalls
- Treating this as a full Jamba 2 implementation rather than a tiny fixed-point prototype.
- Forgetting `valid` is registered while `y` is driven by the current datapath.
- Expecting memory, AXI, softmax, or BF16 support at this learning stage.

## Linear4

### Function
Computes a four-lane signed linear projection with per-output bias.

### Role in the Accelerator
Linear projections create the token, gate, SSM, attention, and output vectors used around a Jamba-style block.

### Chisel Concepts
Nested `Vec`, row-wise dot products, bias addition, combinational projection logic.

### Verilog Correspondence
Each row maps to four multipliers plus an adder tree and a bias adder.

### Common Pitfalls
- Mixing up weight row/column order.
- Forgetting bias width should match the accumulator.
- Assuming this stores weights internally; weights are still input ports in this prototype.

## RmsNormApprox

### Function
Approximates RMSNorm using integer mean-square division and a per-lane weight.

### Role in the Accelerator
Jamba/Mamba blocks usually normalize token activations before projections; this adds that stage in a simple fixed-point form.

### Chisel Concepts
Module reuse, protected division by zero, narrowing accumulator results back to lane width.

### Verilog Correspondence
The stats module feeds a divider and per-lane multiply/divide units.

### Common Pitfalls
- Treating this as numerically exact floating-point RMSNorm.
- Forgetting division truncates toward zero for integer hardware.
- Not handling the all-zero input case.

## Jamba2MiniCore

### Function
Connects integer RMSNorm, input/gate/SSM projections, the tiny Jamba block, and an output projection.

### Role in the Accelerator
This is the more complete mini accelerator datapath: normalize token, project parameters, run Mamba/attention mix, project output.

### Chisel Concepts
Larger module composition, multiple projection submodules, debug outputs, registered `valid`.

### Verilog Correspondence
This becomes a top-level datapath with many submodule instances and explicit buses for weights, activations, state, and scores.

### Common Pitfalls
- Thinking weight input ports are the same as an on-chip SRAM weight system.
- Forgetting the recurrent state updates only after a clock edge.
- Ignoring range loss when accumulator outputs are narrowed back to 8-bit lanes.

## Jamba2MiniStream

### Function
Wraps `Jamba2MiniCore` with a one-entry valid/ready output buffer for token-level streaming.

### Role in the Accelerator
This is the first system-boundary wrapper, making the mini core easier to connect to an upstream token source and downstream consumer.

### Chisel Concepts
Handshake signals, output buffering with `RegInit`, fire conditions, backpressure handling.

### Verilog Correspondence
`inValid && inReady` is the input transfer condition, `outValid && outReady` is the output consume condition, and the output register holds data while stalled.

### Common Pitfalls
- Forgetting the core state advances when a token is accepted.
- Overwriting output data while `outValid` is high and `outReady` is low.
- Letting `inReady` stay high during clear.
