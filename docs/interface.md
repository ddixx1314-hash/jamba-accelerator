# Interface

This document describes the public hardware interfaces of the main top-level modules.

## Common Parameters

Most top-level modules use:

```scala
dataWidth = 8
accWidth  = 32
```

Meaning:

- token lanes and weights use `SInt(dataWidth.W)`
- accumulation and output values use `SInt(accWidth.W)`
- vector length is fixed at 4 in the current mini design

## JambaMiniTile

`JambaMiniTile` is the formal first-stage engineering top. In this stage, it wraps `Jamba2MiniStream` and exposes the same simple token/weight/debug interface.

Constructor:

```scala
class JambaMiniTile(config: JambaMiniConfig = JambaMiniConfig()) extends Module
```

Current constraints:

- `config.lanes == 4`
- `config.convTaps == 3`
- default `dataWidth == 8`
- default `accWidth == 32`

Use this top for generated Verilog and future integration work. `Jamba2MiniStream` and `Jamba2MiniCore` stay documented for testing and internal development, but they are not the preferred external integration boundary.

## Jamba2MiniCore

`Jamba2MiniCore` is the main compute datapath.

### Control Inputs

- `en`: advances state and marks output valid for the current token.
- `clear`: synchronously clears state/valid behavior in child stateful blocks.
- `useAttention`: selects whether the attention path contributes to the block output.

### Token Input

```scala
val x = Input(Vec(4, SInt(dataWidth.W)))
```

This is one tiny token vector.

### Weight and Bias Inputs

RMSNorm:

```scala
val rmsWeight = Input(Vec(4, SInt(dataWidth.W)))
```

Projection matrices and biases:

```scala
val inputWeight = Input(Vec(4, Vec(4, SInt(dataWidth.W))))
val inputBias   = Input(Vec(4, SInt(accWidth.W)))
val gateWeight  = Input(Vec(4, Vec(4, SInt(dataWidth.W))))
val gateBias    = Input(Vec(4, SInt(accWidth.W)))
val bWeight     = Input(Vec(4, Vec(4, SInt(dataWidth.W))))
val bBias       = Input(Vec(4, SInt(accWidth.W)))
val cWeight     = Input(Vec(4, Vec(4, SInt(dataWidth.W))))
val cBias       = Input(Vec(4, SInt(accWidth.W)))
val outWeight   = Input(Vec(4, Vec(4, SInt(dataWidth.W))))
val outBias     = Input(Vec(4, SInt(accWidth.W)))
```

Mamba path:

```scala
val kernel = Input(Vec(3, Vec(4, SInt(dataWidth.W))))
val mambaA = Input(Vec(4, SInt(dataWidth.W)))
```

Attention path:

```scala
val attentionKeys   = Input(Vec(4, Vec(4, SInt(dataWidth.W))))
val attentionValues = Input(Vec(4, Vec(4, SInt(dataWidth.W))))
```

### Outputs

Main output:

```scala
val y = Output(Vec(4, SInt(accWidth.W)))
```

Validity:

```scala
val valid = Output(Bool())
```

Debug outputs:

```scala
val normMeanSquare  = Output(SInt(accWidth.W))
val projectedX      = Output(Vec(4, SInt(dataWidth.W)))
val blockY          = Output(Vec(4, SInt(accWidth.W)))
val stateOut        = Output(Vec(4, SInt(accWidth.W)))
val attentionScores = Output(Vec(4, SInt(accWidth.W)))
```

These debug outputs are kept intentionally to make tests and learning easier.

## Jamba2MiniStream

`Jamba2MiniStream` wraps `Jamba2MiniCore` with token-level valid/ready handshaking.

### Input Handshake

```scala
val inValid = Input(Bool())
val inReady = Output(Bool())
val in      = Input(Vec(4, SInt(dataWidth.W)))
```

The input transfer condition is:

$$
f = v_{in} \land r_{in}
$$

In code, this is:

```scala
fire = io.inValid && io.inReady
```

When `fire` is true, the wrapper accepts one token and advances the core.

### Output Handshake

```scala
val outValid = Output(Bool())
val outReady = Input(Bool())
val out      = Output(Vec(4, SInt(accWidth.W)))
```

The output consume condition is:

$$
c = v_{out} \land r_{out}
$$

When `outValid` is true and `outReady` is false, `out` must remain stable.

### Backpressure Behavior

The wrapper has a one-entry output buffer:

- If output buffer is empty, `inReady` is true.
- If output buffer is full and downstream is not ready, `inReady` is false.
- If output buffer is full and downstream consumes it, a new input can be accepted in the same cycle.
- If `clear` is true, buffered output is cleared and `inReady` is false.

## Timing Notes

- `Jamba2MiniCore` has stateful child blocks, so some state effects appear after clock edges.
- `Jamba2MiniStream` registers output data.
- `valid` / `ready` tests should check both data and stall behavior.

## Generated Top Modules

`GenerateVerilog` emits:

```text
JambaMiniTile.sv
Jamba2MiniAccelerator.sv
Jamba2MiniCore.sv
Jamba2MiniStream.sv
```

The generated Verilog is ignored by git and should be regenerated from Chisel source.
