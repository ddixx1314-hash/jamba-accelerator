# Jamba Accelerator Learning Notes

This project is a small Chisel learning accelerator for Mamba/Jamba-like blocks. It is not a production Jamba 2 implementation. The design intentionally uses small vector sizes, integer arithmetic, and simple IO so each hardware idea is easy to inspect in Chisel, FIRRTL/SystemVerilog, and tests.

## Background: Mamba, SSM, and Jamba

Mamba is a sequence model architecture that uses SSMs (State Space Models) to process tokens through a recurrent state. A simplified state update looks like:

$$
h_{t+1} = A h_t + B x_t
$$

and the output can be written as:

$$
y_t = C h_t
$$

Here $x_t$ is the current token, $h_t$ is the current state, and $y_t$ is the output. This is like a hardware-friendly state machine with memory.

Selective scan is the Mamba-style mechanism that updates this state across a sequence. In real models, some scan parameters depend on the input; in this learning project, `SelectiveScanTiny` keeps only the small shape of state update plus gating.

Jamba combines a Mamba/SSM path with an attention path. In this repository, `TinyJambaBlock` captures that idea at tiny scale: `TinyMambaBlock` handles the state-based path, and `AttentionDecodeTiny` provides a small attention-like decode path.

## Overall Framework: Why These Operators Exist

The modules are not isolated exercises. They form a small token-processing pipeline.

First, the hardware receives one token vector:

$$
x = [x_0, x_1, x_2, x_3]
$$

`RmsNormApprox` normalizes the token magnitude. `Linear4` projections then create different vectors for the block: the main input vector, gate vector, approximate SSM parameters, and final output projection. This is why dot products, GEMM, and linear layers appear early in the project.

The Mamba path uses state to remember sequence history:

```text
token -> CausalConv1D -> MambaStateUpdate / SelectiveScanTiny -> Mamba output
```

`CausalConv1D` captures local history, `MambaStateUpdate` updates the recurrent state, and `SelectiveScanTiny` gates the visible state.

The attention path lets the block directly combine information from keys and values. `AttentionDecodeTiny` keeps only the small hardware shape: dot-product scores and weighted value sums, without softmax, masks, or scaling.

`TinyJambaBlock` mixes the Mamba path and optional attention path. `Jamba2MiniCore` connects normalization, projections, the mixed block, and output projection. `Jamba2MiniStream` wraps the core with valid/ready handshaking.

The project layers can be read like this:

```text
Basic hardware blocks:
  Counter, PE, DotProduct, VectorOps, SmallGemm4x4

Neural-network operators:
  RmsNormStats, RmsNormApprox, Linear4

Mamba/SSM path:
  CausalConv1D, MambaStateUpdate, SelectiveScanTiny, TinyMambaBlock

Attention path:
  AttentionDecodeTiny

Jamba-like mixed blocks:
  TinyJambaBlock, Jamba2MiniCore

System wrapper:
  Jamba2MiniAccelerator, Jamba2MiniStream
```

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
y = a \times b + c
$$

Here $c$ corresponds to `acc_in`, and $y$ corresponds to `acc_out`.

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
S = \sum_i x_i^2
$$

$$
M = \left\lfloor \frac{S}{N} \right\rfloor
$$

Here $S$ is `sumSquares`, $M$ is `meanSquare`, and $N$ is the vector length.

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
y_i = \frac{x_i \gamma_i}{M}
$$

Here $\gamma_i$ corresponds to `weight(i)`, and $M$ corresponds to `meanSquare`.

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
y_r = \beta_r + \sum_c W_{r,c} x_c
$$

Here $W$ is the weight matrix and $\beta$ is the bias vector.

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
h_{t+1} = a h_t + b x_t
$$

Here $h_t$ is the current state and $h_{t+1}$ is the next state.

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
`GenerateVerilog` emits SystemVerilog for `Jamba2MiniTile`, `JambaMiniTile`, `Jamba2MiniAccelerator`, `Jamba2MiniCore`, and `Jamba2MiniStream`.

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

## MacLane

### Function
Computes one signed multiply-accumulate step:

$$
accOut = a \times b + accIn
$$

### Role in the Accelerator
This is the first reusable arithmetic lane for the resource-reuse fabric. Linear layers, attention scores, convolution taps, Mamba state updates, and MLPs all contain this pattern.

### Chisel Concepts
Module-level reuse, explicit signed widths, and helper-based resizing with `SignedMath.resize`.

### Verilog Correspondence
This maps to a signed multiplier followed by a signed adder. There is no register, so it is combinational logic.

### Common Pitfalls
- Making `accWidth` too small for the product.
- Expecting state or latency; this first fabric lane is combinational.
- Forgetting FPGA tools may map the multiplier to DSP resources.

## SharedReduction

### Function
Sums a vector of signed accumulator-width values and resizes the result.

### Role in the Accelerator
Reduction appears in dot products, RMSNorm statistics, attention score accumulation, and weighted value sums.

### Chisel Concepts
Parameterized `Vec`, `reduce(_ +& _)`, and explicit output resizing.

### Verilog Correspondence
This elaborates to an adder chain or tree depending on lowering and synthesis optimization.

### Common Pitfalls
- Thinking `reduce` is a runtime loop; it creates hardware during elaboration.
- Forgetting reduction width can grow beyond the output width.
- Using a zero length, which is illegal.

## SharedDotProduct

### Function
Computes a signed dot product using a chain of `MacLane` modules.

### Role in the Accelerator
This is the first shared-fabric replacement for the baseline `DotProduct`. It makes the MAC resource structure explicit for later resource comparison.

### Chisel Concepts
Submodule instantiation, generated MAC chains, vector ports, and parameterized widths.

### Verilog Correspondence
This becomes several `MacLane` instances wired as an accumulator chain.

### Common Pitfalls
- Assuming the Scala `Seq.fill` creates runtime objects; it instantiates hardware modules at elaboration.
- Forgetting the chain is still combinational in this first version.
- Comparing resources without also checking behavior against the baseline.

## SharedLinear4

### Function
Computes a four-lane linear projection using four `SharedDotProduct` blocks plus bias:

$$
y_r = bias_r + \sum_c weight_{r,c} x_c
$$

### Role in the Accelerator
This is the first shared-fabric version of `Linear4`, which is used by projections in Mamba, attention, and MLP paths.

### Chisel Concepts
Module composition, 2D `Vec` weights, repeated dot-product instances, and explicit signed resizing.

### Verilog Correspondence
This maps to four dot-product submodules plus four bias adders.

### Common Pitfalls
- Mixing up rows and columns in the weight matrix.
- Forgetting bias is accumulator width, not data width.
- Treating this as time-multiplexed; the first version is still parallel and combinational for easier baseline comparison.

## Jamba2MiniTile Loaded Weights

### Function
`Jamba2MiniTile` can select decoded `WeightStoreMini` values with `useLoadedWeights`. The default remains deterministic demo weights.

### Role in the Accelerator
This is the first step from a fixed demo shell toward a parameter-loadable mini accelerator.

### Chisel Concepts
Internal register-file decode, `when`-guarded connection overrides, row-major address mapping, and explicit narrowing from accumulator-width storage to data-width weights.

### Verilog Correspondence
The register file becomes a bank of weight registers. The decode logic wires selected register entries into the typed core ports.

### Common Pitfalls
- Forgetting `clear` resets execution state but preserves weights.
- Assuming one external read port can feed all core weights; the tile uses an internal full decode bus.
- Mixing accumulator-width bias values with data-width matrix weights.

## MacLaneMixed

### Function
Computes a signed multiply-accumulate step where the two operands have different widths.

### Role in the Accelerator
Attention value accumulation multiplies accumulator-width scores by data-width values, so it needs a mixed-width MAC rather than the simpler data-width `MacLane`.

### Chisel Concepts
Independent operand widths, explicit product resizing, and reusable combinational arithmetic.

### Verilog Correspondence
This maps to a signed multiplier with unequal operand widths followed by a signed adder.

### Common Pitfalls
- Reusing an 8-bit-by-8-bit MAC for score-times-value products.
- Forgetting the product can be wider than the accumulator and must be intentionally resized.
- Treating width truncation as accidental instead of documenting it as a fixed-point policy decision.

## SharedAttentionDecodeTiny

### Function
Computes tiny attention decode with shared-fabric blocks:

$$
score_r = q \cdot key_r
$$

$$
y_c = \sum_r score_r \times value_{r,c}
$$

### Role in the Accelerator
This is the first Transformer-like operator mapped onto the reusable MAC/reduction fabric.

### Chisel Concepts
Module reuse across two phases, 2D `Vec` ports, shared dot products, mixed-width MAC chains, and baseline equivalence testing.

### Verilog Correspondence
The score path instantiates shared dot-product modules. The value path instantiates mixed-width MAC chains, one chain per output lane.

### Common Pitfalls
- Confusing this with full attention; softmax, masks, scaling, and multi-head split are still omitted.
- Forgetting scores are accumulator-width while values are data-width.
- Counting generated module lines as synthesis resources without running FPGA synthesis.

## SharedCausalConv1D

### Function
Computes a three-tap per-lane causal convolution using `MacLane` chains:

$$
y_i = x_i k_{0,i} + delay1_i k_{1,i} + delay2_i k_{2,i}
$$

### Role in the Accelerator
This maps the Mamba/Samba local-history convolution path onto the shared MAC fabric while keeping the delay history as operator-specific state.

### Chisel Concepts
Registered delay state, submodule MAC chains, vectorized lane logic, `when`/`elsewhen` state update, and baseline equivalence testing.

### Verilog Correspondence
The delay values map to registers. Each lane maps to three combinational MAC stages wired as an accumulator chain.

### Common Pitfalls
- Forgetting convolution history updates only on a clock edge.
- Clearing execution state but expecting weights or kernels to reset.
- Treating the delay history as a shared arithmetic resource; it is operator-specific state around the shared fabric.

## SharedMambaStateUpdate

### Function
Updates recurrent SSM state with shared MAC-style blocks:

$$
state_i := state_i a_i + x_i b_i
$$

### Role in the Accelerator
This captures the core Mamba recurrence while separating operator-specific state registers from reusable multiply-accumulate arithmetic.

### Chisel Concepts
Registered state, mixed-width recurrent multiply, data-width input multiply, sequential update with `when`, and baseline equivalence testing.

### Verilog Correspondence
The state vector maps to registers. The next-state datapath maps to a mixed-width signed multiplier/add path plus an input multiply-add path.

### Common Pitfalls
- Expecting `stateOut` to show the newly computed value before the clock edge.
- Forgetting `clear` has priority over `en`.
- Treating recurrent state as a shared resource; the arithmetic can be shared, but state ownership is operator-specific.

## SharedSelectiveScanTiny

### Function
Combines shared Mamba state update with an element-wise gate:

$$
y_i = stateOut_i \times gate_i
$$

### Role in the Accelerator
This is the first combined selective-scan operator in the optimized track. It preserves the scan state semantics while reusing shared MAC blocks for update and gate arithmetic.

### Chisel Concepts
Submodule composition, registered state through `SharedMambaStateUpdate`, mixed-width gate multiplication, and baseline equivalence testing.

### Verilog Correspondence
The state update submodule owns the state registers. The gate path is combinational mixed-width multiply-add logic with zero accumulator input.

### Common Pitfalls
- Forgetting `y` is based on the visible current state, not the next state before the clock edge.
- Confusing the gate path with full production Mamba selectivity.
- Assuming shared arithmetic removes the need for operator-specific recurrent state.

## SharedTinyMambaBlock

### Function
Composes shared causal convolution, shared selective scan, state projection, and residual gating.

### Role in the Accelerator
This is the first optimized-track Mamba path, combining multiple shared-fabric operators while preserving the baseline `TinyMambaBlock` behavior.

### Chisel Concepts
Hierarchical module composition, explicit narrowing from convolution accumulator to scan input, mixed-width state projection, residual gate accumulation, and baseline equivalence testing.

### Verilog Correspondence
This elaborates into shared convolution and scan submodules plus MAC chains for state projection and residual gating.

### Common Pitfalls
- Assuming `SelectiveScanTiny.y` is the final block output; the baseline block computes `stateOut * c + x * gate`.
- Forgetting the scan input is narrowed back to data width.
- Comparing only combinational output and missing the one-cycle state update behavior.

## SharedTinyJambaBlock

### Function
Composes the shared Mamba path with shared tiny attention decode. When `useAttention` is true, attention output is added to the Mamba output.

### Role in the Accelerator
This is the first optimized-track hybrid Jamba-like block. It demonstrates shared fabric reuse across both Mamba-style and Transformer-style paths.

### Chisel Concepts
Parallel path composition, `Mux`-selected output behavior, shared submodule reuse, and debug visibility for state and attention scores.

### Verilog Correspondence
Both shared Mamba and shared attention hardware are instantiated. `useAttention` controls the output mux/add path.

### Common Pitfalls
- Thinking `useAttention` removes attention hardware; it only controls whether the attention result contributes to output.
- Forgetting both paths must match the baseline even when one path is not selected.
- Comparing only `y` and missing debug score/state equivalence.

## SharedDenseMLPMini

### Function
Computes the dense MLP path with shared linear projections:

$$
hidden = ReLU(gate(x)) \times up(x)
$$

$$
y = down(hidden)
$$

### Role in the Accelerator
This maps the layer MLP half onto the shared fabric. Together with `SharedTinyJambaBlock`, it gives optimized-track coverage for both `Mixer` and `MLP` layer components.

### Chisel Concepts
Shared projection modules, lane-wise activation, explicit narrowing, local hidden multiplication, and baseline equivalence testing.

### Verilog Correspondence
The three linear projections instantiate shared dot-product structures. The hidden path is lane-local combinational multiply and truncation.

### Common Pitfalls
- Forgetting the ReLU applies to the gate projection before narrowing.
- Missing the baseline's truncation behavior in the hidden multiply.
- Checking only final `y` and missing intermediate gate/up/hidden mismatches.

## SharedMlpPathMini

### Function
Implements the dense-or-MoE MLP path with shared fabric blocks. Dense mode uses `SharedDenseMLPMini`; MoE mode uses shared router dot products plus shared expert MLPs.

### Role in the Accelerator
This completes shared-fabric coverage for the Jamba layer MLP side: dense MLP, router logits, expert MLPs, and final dense/MoE selection now have baseline-equivalent shared versions.

### Chisel Concepts
Hierarchical composition, aggregate `Mux` between vector outputs, shared submodule reuse, top-1 router comparison, and baseline/shared equivalence harnesses.

### Verilog Correspondence
The router elaborates to dot-product score logic. Each expert elaborates to shared dense MLP projection logic. The `enableMoE` signal maps to output muxes and valid-signal gates.

### Common Pitfalls
- Forgetting the dense branch still computes even when MoE is enabled.
- Treating `dispatchReady` and `combineReady` as real backpressure in this first combinational MoE-lite version.
- Comparing only selected expert output while missing router score equivalence.

## SharedJamba2MiniLayer

### Function
Implements the formal mini layer shape `RMSNorm -> Mixer -> residual -> RMSNorm -> MLP -> residual`, with the MLP side using `SharedMlpPathMini`.

### Role in the Accelerator
This is the first layer-level shared-fabric comparison point. It keeps the same Mamba/Attention mixer semantics as `Jamba2MiniLayer` while proving the shared MLP path can sit inside the full layer contract.

### Chisel Concepts
Large IO bundles, submodule composition, residual arithmetic, stateful child modules, vector muxing, and focused mode tests for Mamba, attention, and MoE.

### Verilog Correspondence
RMSNorm and mixers elaborate like the baseline. The MLP submodule elaborates to shared router/dense/expert fabric. The residual adds are combinational add paths.

### Common Pitfalls
- Forgetting the shared mixer modules still preserve their original state/cache timing.
- Forgetting Mamba and attention child modules still own their internal state/cache.
- Missing that `firstResidual` is narrowed back to data width before the second RMSNorm.

## Shared Mixer Projections

### Function
Maps Mamba mixer input/B/C projections and attention mixer Q/K/V/out projections to `SharedLinear4`.

### Role in the Accelerator
This attacks the next layer-level bottleneck after the MLP path. The major linear projection groups in Mixer and MLP now share the same fabric style.

### Chisel Concepts
Baseline/shared comparison harnesses, stateful submodule equivalence, projection narrowing, KV cache timing, and scan/convolution timing.

### Verilog Correspondence
The projection modules elaborate through shared dot-product structures. The surrounding registers for convolution history, scan state, and KV cache remain sequential state owned by each mixer.

### Common Pitfalls
- Assuming shared projections also share cache or recurrent state; only arithmetic fabric is mapped here.
- Missing saturation in attention projections while Mamba projections use truncation.
- Comparing one cycle only and missing stateful behavior after cache/history updates.

## SerialSharedLinear4

### Function
Computes the same 4x4 linear projection as `Linear4`, but reuses one `MacLane` across 16 cycles.

### Role in the Accelerator
This is the first real time-multiplexed fabric block. It turns resource reuse from a structural mapping into an execution schedule: fewer MACs, more cycles.

### Chisel Concepts
Input latching, `RegInit`, `busy/done` control, row/column counters, dynamic `Vec` indexing, and sequential writeback.

### Verilog Correspondence
The row/column counters, accumulator, latched operands, and output vector map to registers. The single `MacLane` is combinational logic reused each cycle by changing mux-selected operands.

### Common Pitfalls
- Expecting the result in the same cycle as `start`; the result appears after the serial MAC schedule finishes.
- Changing inputs while busy and expecting them to affect the current operation; operands are latched on `start`.
- Forgetting `done` is a pulse while `y` remains stored in output registers.

## SerialProjectionScheduler4

### Function
Runs several 4x4 projections through one `SerialSharedLinear4`, producing one output vector per projection.

### Role in the Accelerator
This is the first projection-group scheduler. It models Mamba input/B/C reuse with three projections and attention Q/K/V/out reuse with four projections.

### Chisel Concepts
FSM states, nested `Vec` registers, dynamic projection indexing, submodule start/done handshaking, and multi-result writeback.

### Verilog Correspondence
The scheduler state, projection index, latched weights, and output groups map to registers. The single serial linear submodule is reused by changing the indexed weight/bias source.

### Common Pitfalls
- Forgetting the scheduler adds cycles on top of each serial projection.
- Updating input weights while busy and expecting current results to change; the scheduler latches the projection group on `start`.
- Confusing projection-group scheduling with full layer scheduling; state/cache updates still need separate control.

## Serial Semantic Projection Groups

### Function
Wraps serial projection scheduling with model-level ports: Mamba exposes `projected`, `b`, and `c`; attention exposes `q`, `k`, `v`, and output projection `y`.

### Role in the Accelerator
These modules make the serial fabric usable by future mixer replacements without losing the algorithm dataflow. Attention keeps the output projection input separate from the token input.

### Chisel Concepts
Semantic IO bundles, submodule wrapping, separate input latching, FSM-controlled projection selection, `switch` writeback, and saturation/truncation policy reuse.

### Verilog Correspondence
The Mamba wrapper mostly wires a generic scheduler. The attention wrapper elaborates to one serial linear submodule, control registers, projection-select muxing, and result registers for Q/K/V/out.

### Common Pitfalls
- Treating attention Q/K/V/out as if they all consume the same input vector; out projection consumes the attention value path.
- Forgetting Mamba projection narrowing uses truncation while attention Q/K/V use saturation.
- Replacing mixer projections before accounting for the extra schedule latency.

## SerialMambaMixerMini

### Function
Runs Mamba input/B/C projections through a serial projection group, then runs serial causal convolution and updates selective scan for one token.

### Role in the Accelerator
This is the first token-level serial mixer shell. It demonstrates how projection and convolution schedulers can sit inside a Mamba mixer while preserving post-token state behavior.

### Chisel Concepts
FSM sequencing, submodule handshakes, registered intermediate projection results, one-cycle state update pulses, and valid/done output timing.

### Verilog Correspondence
The projection group, convolution history, scan state, and output registers are separate sequential blocks. The controller runs projection cycles first, then convolution cycles, then pulses the scan update path.

### Common Pitfalls
- Comparing it cycle-for-cycle with the combinational-projection mixer; this module is multi-cycle.
- Reading `conv` after the convolution history updates instead of registering the conv value used for the token.
- Forgetting `done/valid` means post-token output is ready, not that projection alone completed.

## SerialCausalConvMini

### Function
Computes one causal convolution token by reusing a single `MacLane` across every lane and tap.

### Role in the Accelerator
This moves Mamba's local convolution path from parallel tap multipliers to a time-multiplexed schedule, reducing arithmetic replication at the cost of latency.

### Chisel Concepts
Nested lane/tap counters, dynamic `Vec` indexing, history registers, accumulator reset per lane, and start/done handshaking.

### Verilog Correspondence
The history buffer, input/kernel latches, lane/tap counters, accumulator, and output vector map to registers. A single MAC datapath is reused as the counters advance.

### Common Pitfalls
- Updating history before all tap products for the token are computed.
- Forgetting to reset the accumulator between output lanes.
- Comparing against the parallel convolution before waiting for `done`.

## SerialSelectiveScanMini

### Function
Runs one selective-scan token with a single mixed-width MAC lane, sequencing `state * a`, `+ x * b`, and `* c` for each lane.

### Role in the Accelerator
This is the recurrent Mamba state path in the serial fabric. It keeps the complete scan recurrence, but turns each lane update into a scheduled MAC sequence.

### Chisel Concepts
FSM control, lane counters, persistent state registers, temporary recurrence registers, mixed-width arithmetic, and `start/done` handshaking.

### Verilog Correspondence
The scan state, temporary recurrent value, lane counter, and FSM state map to registers. The single `MacLaneMixed` elaborates as combinational multiply-add logic selected by the current micro-op.

### Common Pitfalls
- Comparing before `done`; the post-token state is only valid after the serial lane schedule finishes.
- Forgetting `clear` resets both the FSM and persistent scan state.
- Mixing old-state visible timing with post-token serial output timing.

## SerialAttentionMixerMini

### Function
Runs attention Q/K/V and output projections through one serial projection fabric, then updates the KV cache and decodes a token.

### Role in the Accelerator
This is the attention-side counterpart to the serial Mamba mixer. It proves that the same serial projection structure can support Transformer-style cache behavior.

### Chisel Concepts
Projection scheduling, KV cache registers, cache pointer update, saturation policy, FSM sequencing, and separate output-projection input handling.

### Verilog Correspondence
The serial projection datapath is reused for Q, K, V, and out projection. The KV cache and write pointer are registers, while the current score/value accumulation is still combinational in this mini version.

### Common Pitfalls
- Treating the output projection as another token-input projection; it consumes the decoded attention value vector.
- Forgetting the current token must be written into KV cache before the output projection phase.
- Assuming all attention arithmetic is serial; score/value decode is still a future reuse target.

## SerialJamba2MiniLayer

### Function
Runs a mini Jamba-style layer with serial Mamba or serial attention mixer, residual paths, RMSNorm, and the shared dense/MoE MLP path.

### Role in the Accelerator
This is the first full layer-level semantic-serial execution point. It supports the complete mini algorithm shape while exposing a baseline/shared/serial comparison for resource reuse.

### Chisel Concepts
Layer-level FSM sequencing, mixer mode selection, child-module handshakes, residual narrowing, shared MLP routing, and debug signal propagation.

### Verilog Correspondence
RMSNorm and residual adds elaborate mostly as combinational logic. The serial mixer child modules and their state/cache registers dominate the sequential logic, while the current MLP path remains shared but not fully serial.

### Common Pitfalls
- Expecting same-cycle layer output; this layer waits for the selected serial mixer before asserting `done`.
- Forgetting Mamba and attention modes preserve different internal state: scan/conv history versus KV cache.
- Reading the layer as a final accelerator top; it is a layer primitive that still needs tile-level scheduling.

## UnifiedProjectionScheduler4

### Function
Schedules named Jamba layer projection slots through one `SerialSharedLinear4`, while allowing each slot to use a different input vector.

### Role in the Accelerator
This is the first concrete piece of the `UnifiedJamba2MiniLayer` plan. It can cover Mamba input/B/C, attention Q/K/V/out, and MLP gate/up/down projection slots with one projection MAC schedule.

### Chisel Concepts
Slot enable masks, `PriorityEncoder`, safe dynamic `Vec` indexing, operand latching, reusable submodule scheduling, and sparse projection-slot execution.

### Verilog Correspondence
The slot enables, operands, weights, biases, outputs, and current slot index map to registers. The single `SerialSharedLinear4` elaborates as the reused projection datapath selected by the scheduler.

### Common Pitfalls
- Assuming every projection consumes the same input vector; layer projections use norm1, attention raw output, norm2, or hidden activation.
- Forgetting disabled slots are skipped but their outputs remain zero.
- Indexing a `Vec` with a sentinel slot value without a safe fallback.

## UnifiedJamba2MiniLayer

### Function
Runs a Jamba2 mini layer with one unified projection scheduler for Mamba/Attention projections and dense MLP gate/up/down projections.

### Role in the Accelerator
This is the first layer-level implementation that turns the unified projection idea into an executable token flow. It keeps dedicated units for convolution, scan, and attention score/value accumulation while consolidating projection-heavy work.

### Chisel Concepts
Multi-phase FSM control, repeated submodule reuse, slot-enable scheduling, persistent KV/SSM state, safe narrowing/saturation, and staged data dependency handling.

### Verilog Correspondence
The layer FSM, cached token, intermediate projection outputs, KV cache, scan state, hidden activation, and final output map to registers. One `UnifiedProjectionScheduler4` submodule is relaunched for each projection phase.

### Common Pitfalls
- Trying to run all projection slots in one launch; MLP down depends on hidden activation, and attention out depends on raw attention output.
- Treating this as a global one-MAC design; conv, scan, and attention score/value are still specialized units in this stage.
- Forgetting the MoE path is longer than the dense path because it runs router, selected expert gate/up, and selected expert down.

## UnifiedMoEPathMini

### Function
Runs top-1 MoE-lite with the unified projection scheduler: router scoring first, then only the selected expert's gate/up/down projections.

### Role in the Accelerator
This brings MoE-lite into the unified scheduling path. Instead of computing both experts in parallel, the hardware selects one expert and reuses the same projection fabric for its MLP.

### Chisel Concepts
Projection-slot packing, top-1 selection, staged scheduler launches, selected-expert muxing, and registered router/debug outputs.

### Verilog Correspondence
The router scores, selected expert, gate/up/down intermediates, hidden activation, and output map to registers. One `UnifiedProjectionScheduler4` submodule is relaunched for router, expert gate/up, and expert down phases.

### Common Pitfalls
- Forgetting the router is represented as a 4-output projection where only rows 0 and 1 are real expert scores.
- Computing all experts in parallel; this module intentionally schedules only the selected expert.
- Expecting MoE output immediately; router, gate/up, hidden, and down are separate serial phases.

## UnifiedJamba2MiniAcceleratorTile

### Function
Wraps one `UnifiedJamba2MiniLayer` with token valid/ready, output backpressure, mode selection, status/debug outputs, and a mini weight store.

### Role in the Accelerator
This is the first accelerator-shaped top for the unified execution path. It turns the layer into an end-to-end token engine with loadable weights and a mode-selectable Mamba/Attention demo.

### Chisel Concepts
Multi-cycle top-level FSM, ready/valid handshaking, output buffering, weight-store decode, mode latching, and child-module start/done control.

### Verilog Correspondence
The token buffer, mode registers, output buffer, done flag, and FSM state map to registers. The weight store is a register-file-backed memory, and the unified layer is a multi-cycle child datapath.

### Common Pitfalls
- Treating the unified layer like a combinational core; the top must wait for `layer.io.done`.
- Accepting a second token while an old output is held under backpressure.
- Clearing token execution and accidentally erasing weights; the weight store intentionally preserves contents on `clear`.

## UnifiedJamba2MiniTileScheduler

### Function
Runs multiple `UnifiedJamba2MiniLayer` instances sequentially for one token, with Jamba-style sparse attention placement from `Jamba2MiniConfig`.

### Role in the Accelerator
This proves the unified layer can form a mini Jamba layer stack. Each layer keeps its own SSM/KV state, while the scheduler moves the token output from one layer into the next.

### Chisel Concepts
Layer-index FSMs, sequential child-module launch, dynamic layer progress tracking, per-layer debug vectors, and accumulator-to-data-width narrowing between layers.

### Verilog Correspondence
The active layer index, current token vector, output registers, layer output registers, and done flag map to registers. Each unified layer remains a separate multi-cycle child module with its own state/cache.

### Common Pitfalls
- Sharing one physical layer instance without also virtualizing per-layer SSM/KV state.
- Launching the next layer before the previous layer's `done` pulse.
- Forgetting attention placement is fixed by config, not by the runtime mode field used in the single-layer accelerator shell.

## UnifiedJamba2MiniFullTile

### Function
Wraps the multi-layer unified scheduler with token ready/valid, output backpressure, a field-banked mini weight store, per-layer weight-segment decode, and tile-level debug/status outputs.

### Role in the Accelerator
This is the current full unified accelerator endpoint. It runs a token through the config-defined mini Jamba layer stack, while preserving loadable-weight and FPGA-style top-level control signals.

### Chisel Concepts
Top-level FSM control, ready/valid handshakes, child scheduler launch/wait, output buffering, `Vec` debug fanout, active-layer bank selection, and register-file weight decode.

### Verilog Correspondence
The input token register, output buffer, enable-MoE latch, done flag, and FSM state map to registers. `LayeredWeightStoreMini` maps flat writes into field-specific per-layer registers, then muxes only across layer banks before feeding the scheduler.

### Common Pitfalls
- Expecting runtime mixer mode selection; this full tile follows `Jamba2MiniConfig` attention placement.
- Accepting a new token while an old output is still held under backpressure.
- Forgetting loaded weights are layer-segmented with a 256-address stride, while demo weights remain shared constants.

## LayeredWeightStoreMini

### Function
Accepts flat software weight writes, preserves readback by address, and decodes known mini weight fields into per-layer banks selected by `activeLayer`.

### Role in the Accelerator
This removes the worst `readAll` fanout from the unified full tile. It is the first weight-path optimization that turns the loaded-weight feature from a functional smoke path into a more FPGA-friendly data path.

### Chisel Concepts
Register-file write/read, decoded address writes, nested `Vec` banks, active-layer muxing, and width-safe data resizing.

### Verilog Correspondence
The raw readback memory and each decoded field bank map to registers. Address comparisons update only the matching field, while the active layer index selects the field bank that feeds the scheduler.

### Common Pitfalls
- Confusing the software-visible flat address with the internal banked field storage.
- Forgetting `clear` preserves weights.
- Assuming this is final BRAM inference; it is a banked register-file step before sequential BRAM-style loading.

## WeightAddressGenMini

### Function
Generates the flat weight address for one requested mini field element from `(layer, field, row, col, lane, tap, expert)`.

### Role in the Accelerator
This is the first block for a sequential BRAM-style weight loader. It keeps the address formula centralized so later loader FSMs can request one field element at a time without duplicating layout logic.

### Chisel Concepts
Field-id decoding with `MuxLookup`, parameterized address widths, local-address generation, layer-stride addition, and metadata outputs such as `valid` and `isAcc`.

### Verilog Correspondence
The field decoder maps to combinational mux logic. The layer stride is a multiply/add address calculation, and the output address can feed a memory read port in a later sequential loader.

### Common Pitfalls
- Mixing up `lane` for vectors with `row/col` for matrices.
- Forgetting kernel uses `tap * lanes + lane`.
- Treating all fields as the same width; bias fields remain accumulator-width while weights are data-width.
