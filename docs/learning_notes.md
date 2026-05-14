# Jamba Accelerator Learning Notes

This project is a small Chisel learning accelerator for Mamba/Jamba-like blocks. It is not a production Jamba 2 implementation. The design intentionally uses small vector sizes, integer arithmetic, and simple IO so each hardware idea is easy to inspect in Chisel, FIRRTL/SystemVerilog, and tests.

## Current Mini Accelerator Shape

The most complete top-level datapath is `Jamba2MiniCore`.

```text
x
 -> RmsNormApprox
 -> Linear4 input/gate/B/C projections
 -> TinyJambaBlock
      -> TinyMambaBlock
           -> CausalConv1D
           -> SelectiveScanTiny
      -> AttentionDecodeTiny optional path
 -> Linear4 output projection
 -> y
```

`Jamba2MiniStream` wraps the core with a one-entry valid/ready output buffer. This makes the core look more like a token-processing hardware block, while keeping weights on simple input ports.

Useful commands:

```bash
sbt test
./scripts/run_test.sh
./scripts/generate_verilog.sh
```

`./scripts/run_test.sh` runs Chisel tests, Python golden tests, SystemVerilog generation, and Verilator lint.

## Counter

### Function
`Counter` is an enable-controlled unsigned counter. When `en` is true, the register increments by one each clock cycle. When `en` is false, it holds its previous value.

### Role in the Accelerator
Counters are basic control building blocks. Later accelerator blocks need counters for token positions, loop indices, pipeline timing, memory addresses, and sequence length tracking.

### Chisel Concepts
`RegInit` creates a register with a reset value. `when(io.en)` describes conditional sequential update logic. `UInt(width.W)` makes the counter width explicit.

### Verilog Correspondence
This maps to an `always_ff @(posedge clock)` register. `RegInit(0.U(width.W))` becomes a reset-to-zero register. `when(io.en)` is like `if (en) cnt <= cnt + 1;`.

### Common Pitfalls
- Expecting `out` to change immediately after poking `en`; the counter updates on a clock edge.
- Forgetting that unsigned fixed-width counters wrap around on overflow.
- Using a width of zero or forgetting to make the width parameter legal.

## PE

### Function
`PE` is a combinational multiply-accumulate processing element:

$$
\text{acc\_out} = a \times b + \text{acc\_in}
$$

### Role in the Accelerator
Multiply-accumulate is the core operation behind dot products, GEMM, linear projections, attention score calculation, and many SSM datapaths.

### Chisel Concepts
`SInt` represents signed arithmetic. `Wire(SInt(accWidth.W))` names the multiply result. Explicit `dataWidth` and `accWidth` separate input precision from accumulation precision.

### Verilog Correspondence
This is a signed multiplier feeding a signed adder. There is no register in `PE`, so it is combinational logic, similar to `assign acc_out = a * b + acc_in;`.

### Common Pitfalls
- Making `accWidth` too small and truncating the product.
- Assuming `valid` means pipelined completion; here it is always true because the module is combinational.
- Mixing signed and unsigned values accidentally.

## DotProduct

### Function
Computes the signed dot product of two fixed-length vectors:

$$
y = \sum_i a_i b_i
$$

### Role in the Accelerator
Dot product is the basic compute pattern behind GEMM, attention scores, linear layers, and output projections.

### Chisel Concepts
`Vec` represents repeated hardware lanes. A Scala `for` loop elaborates multiple multipliers. `reduce(_ +& _)` builds an adder tree. `SignedMath.resize` makes the intended output width explicit.

### Verilog Correspondence
The loop generates parallel multipliers. `reduce` becomes a chain or tree of adders. The vector ports map to repeated signed input/output signals in generated SystemVerilog.

### Common Pitfalls
- Thinking the Scala loop runs over time; it generates parallel hardware at elaboration time.
- Forgetting product width is larger than input width.
- Expecting a combinational module to need `clock.step()` in tests.

## SmallGemm4x4

### Function
Computes a 4x4 signed matrix multiply:

$$
c_{row,col} = \sum_k a_{row,k} b_{k,col}
$$

### Role in the Accelerator
Small GEMM is the next step after dot product and models the dense compute pattern used in projections and neural network layers.

### Chisel Concepts
Nested `Vec` builds 2D hardware structures. Nested Scala loops elaborate repeated multiply-add datapaths.

### Verilog Correspondence
The module is similar to manually writing 16 dot-product circuits, one for each output element.

### Common Pitfalls
- Mixing up `b(row)(col)` and `b(k)(col)`.
- Forgetting that four products need more accumulator width than one product.
- Writing expected test values without hand-checking the row-column dot product.

## VectorOps

### Function
Computes element-wise add, subtract, multiply, and ReLU for two signed vectors.

### Role in the Accelerator
Vector operations appear around the main compute kernels for residual paths, gates, activations, and simple post-processing.

### Chisel Concepts
Element-wise `Vec` assignments create independent hardware lanes. `Mux` builds conditional selection logic for ReLU.

### Verilog Correspondence
Each lane maps to its own add/sub/mul logic. ReLU maps to a signed comparison and mux.

### Common Pitfalls
- Confusing element-wise multiply with dot product.
- Forgetting ReLU is combinational compare plus mux.
- Using an output width too narrow for multiplication.

## SignedMath

### Function
`SignedMath.resize` centralizes signed width conversion.

### Role in the Accelerator
Tiny accelerator blocks repeatedly narrow or widen signed arithmetic results. A helper makes these choices visible instead of hiding them inside scattered casts.

### Chisel Concepts
The helper creates a typed `Wire(SInt(width.W))` and assigns the source value into it.

### Verilog Correspondence
This corresponds to signed extension or truncation in generated hardware, depending on source and target widths.

### Common Pitfalls
- Treating resize as saturation; it is not saturation.
- Forgeting narrowing can drop high bits.
- Using `.asTypeOf` when the intent is numeric resize rather than bit reinterpretation.

## RmsNormStats

### Function
Computes integer RMSNorm statistics: sum of squares and floor mean square.

$$
\text{sumSquares} = \sum_i x_i^2
$$

$$
\text{meanSquare} = \left\lfloor \frac{\text{sumSquares}}{\text{length}} \right\rfloor
$$

### Role in the Accelerator
RMSNorm needs a sum-of-squares reduction before reciprocal square-root and scaling. This module captures that first hardware step.

### Chisel Concepts
Signed squaring, `Vec` of intermediate squares, reduction with `reduce`, and constant division.

### Verilog Correspondence
The module becomes a bank of squarers feeding an adder reduction and constant divider.

### Common Pitfalls
- Thinking this is full RMSNorm; it only computes statistics.
- Forgetting negative inputs still produce positive squares.
- Ignoring accumulator width for squared sums.

## RmsNormApprox

### Function
Approximates RMSNorm using integer mean-square division and per-lane weights.

$$
y_i = \frac{x_i \times \text{weight}_i}{\text{meanSquare}}
$$

### Role in the Accelerator
Jamba/Mamba blocks typically normalize token activations before projections. This module adds a simple fixed-point-friendly normalization stage.

### Chisel Concepts
Module reuse, protected division by zero, signed multiply/divide, and narrowing accumulator results back to lane width.

### Verilog Correspondence
`RmsNormStats` feeds a denominator. Each lane multiplies by a weight and divides by the denominator.

### Common Pitfalls
- Treating this as numerically exact floating-point RMSNorm.
- Forgetting integer division truncates.
- Not protecting against all-zero input.

## Linear4

### Function
Computes a four-lane signed linear projection with per-output bias.

$$
y_{row} = \text{bias}_{row} + \sum_{col} \text{weight}_{row,col} x_{col}
$$

### Role in the Accelerator
Linear projections create token features, gates, SSM parameters, attention vectors, and output activations.

### Chisel Concepts
Nested `Vec`, row-wise dot products, bias addition, and combinational projection logic.

### Verilog Correspondence
Each row becomes four multipliers, an adder reduction, and a bias adder.

### Common Pitfalls
- Mixing up weight row/column order.
- Forgetting bias should use accumulator width.
- Assuming weights are stored internally; in this prototype they are input ports.

## MambaStateUpdate

### Function
Updates a small signed SSM state vector.

$$
\text{state}_{next} = a \times \text{state}_{current} + b \times x
$$

### Role in the Accelerator
This is the recurrent state update pattern used by tiny Mamba-like scan blocks.

### Chisel Concepts
`RegInit` stores state. `VecInit` initializes all state lanes. `when` / `.elsewhen` implement clear and enable priority.

### Verilog Correspondence
The state vector maps to registers. `clear` is synchronous clearing logic. `en` controls whether next-state values are loaded.

### Common Pitfalls
- Expecting updated state before a clock edge.
- Forgetting clear has priority over enable.
- Truncating recurrent products without choosing an accumulator width.

## CausalConv1D

### Function
Computes a three-tap per-channel causal convolution using the current token and two delayed tokens.

### Role in the Accelerator
Mamba-style blocks often use local causal convolution before selective scan.

### Chisel Concepts
Delay registers, per-lane combinational multiply-add, synchronous enable, and synchronous clear.

### Verilog Correspondence
`delay1` and `delay2` are shift-register stages. The three taps are parallel multipliers feeding adders.

### Common Pitfalls
- Mixing up tap order: current, previous, previous-previous.
- Forgetting delay registers update on the clock edge.
- Letting delays change while `en` should hold them.

## SelectiveScanTiny

### Function
Wraps state update and applies an element-wise gate to the visible state.

### Role in the Accelerator
This approximates the Mamba selective scan datapath in a small integer-friendly form.

### Chisel Concepts
Child module instantiation, IO connection, vectorized combinational post-processing.

### Verilog Correspondence
`MambaStateUpdate` becomes an instantiated submodule. The gate is a bank of signed multipliers.

### Common Pitfalls
- Reading registered state as if it updated combinationally.
- Confusing raw state output with gated output.
- Under-sizing the gate multiplication result.

## AttentionDecodeTiny

### Function
Computes dot-product attention scores and weighted value sums without softmax.

### Role in the Accelerator
Jamba-like blocks mix SSM-style computation with attention-style decode. This module models the attention-side datapath at tiny scale.

### Chisel Concepts
Nested `Vec`, nested loops, score reductions, and matrix-vector weighted sums.

### Verilog Correspondence
The loops elaborate parallel multipliers and adders for every score and output lane.

### Common Pitfalls
- Treating this as full attention; it intentionally omits softmax and scaling.
- Mixing up key rows and value rows.
- Forgetting score-times-value needs enough width.

## TinyMambaBlock

### Function
Combines causal convolution, selective scan, state projection, and residual gating.

### Role in the Accelerator
This is the compact Mamba-like datapath inside the mini accelerator.

### Chisel Concepts
Composing modules, vector wires, registered state, combinational residual output, and explicit narrowing before scan input.

### Verilog Correspondence
This maps to multiple submodules wired together with explicit vector buses.

### Common Pitfalls
- Forgetting convolution and scan state advance on clock edges.
- Assuming numerical equivalence to production Mamba.
- Narrowing convolution output back to input width without considering range.

## TinyJambaBlock

### Function
Combines a Mamba path with an optional tiny attention path.

### Role in the Accelerator
This captures the core Jamba idea at tiny scale: mix SSM-style sequence processing with attention-style decode.

### Chisel Concepts
Parallel child modules, `Mux`, shared vector inputs, and debug outputs for state and attention scores.

### Verilog Correspondence
Both paths exist in hardware. `useAttention` controls the output mux/add path; it does not remove the attention hardware.

### Common Pitfalls
- Thinking `useAttention` prevents attention hardware from being generated.
- Forgetting both paths are combinationally evaluated.
- Not exposing enough debug signals when testing sequential behavior.

## Jamba2MiniAccelerator

### Function
Top-level wrapper around `TinyJambaBlock` with enable, clear, output state, attention scores, and valid.

### Role in the Accelerator
This is the first runnable top-level mini accelerator block in the project.

### Chisel Concepts
Top-level module wrapping, IO forwarding, and a registered valid signal.

### Verilog Correspondence
This becomes a Verilog top module that instantiates the block and registers a one-cycle valid flag.

### Common Pitfalls
- Treating this as a full Jamba 2 implementation rather than a tiny prototype.
- Forgetting `valid` is registered while `y` is driven by current datapath inputs.
- Expecting memory, AXI, softmax, or BF16 support at this learning stage.

## Jamba2MiniCore

### Function
Connects integer RMSNorm, input/gate/SSM projections, the tiny Jamba block, and output projection.

### Role in the Accelerator
This is the more complete mini accelerator datapath: normalize token, project parameters, run Mamba/attention mix, project output.

### Chisel Concepts
Larger module composition, multiple projection submodules, debug outputs, registered `valid`, and fixed-size top-level IO.

### Verilog Correspondence
This becomes a top-level datapath with many submodule instances and explicit buses for weights, activations, state, and scores.

### Common Pitfalls
- Thinking weight input ports are the same as an on-chip SRAM weight system.
- Forgetting recurrent state updates only after a clock edge.
- Ignoring range loss when accumulator outputs are narrowed back to 8-bit lanes.

## Jamba2MiniStream

### Function
Wraps `Jamba2MiniCore` with a one-entry valid/ready output buffer for token-level streaming.

### Role in the Accelerator
This is the first system-boundary wrapper, making the mini core easier to connect to an upstream token source and downstream consumer.

### Chisel Concepts
Handshake signals, output buffering with `RegInit`, fire conditions, backpressure handling, and stable output while stalled.

### Verilog Correspondence
`inValid && inReady` is the input transfer condition. `outValid && outReady` is the output consume condition. The output register holds data while stalled.

### Common Pitfalls
- Forgetting the core state advances when a token is accepted.
- Overwriting output data while `outValid` is high and `outReady` is low.
- Letting `inReady` stay high during clear.

## GenerateVerilog and Scripts

### Function
`GenerateVerilog` emits SystemVerilog for `Jamba2MiniAccelerator`, `Jamba2MiniCore`, and `Jamba2MiniStream`.

### Role in the Accelerator
Generated SystemVerilog is the bridge from Chisel learning code to hardware tool flows.

### Chisel Concepts
`ChiselStage.emitSystemVerilogFile` elaborates Chisel modules and runs FIRRTL/CIRCT lowering.

### Verilog Correspondence
The generated `.sv` files are concrete Verilog/SystemVerilog modules that can be linted by Verilator.

### Common Pitfalls
- Editing generated Verilog instead of editing Chisel source.
- Forgetting generated files are ignored by git and should be regenerated.
- Treating lint success as full functional proof; tests are still needed.
