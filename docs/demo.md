# End-to-End Demo

This document describes the first deterministic end-to-end demo for the formal `Jamba2MiniTile` shell.

The demo is still a mini architecture trace. It does not run real Jamba2 checkpoint weights. Its purpose is to prove that the formal top can load weights, clear execution state, accept multiple tokens, advance the hybrid core, expose debug state, and match a Python golden trace.

## What The Demo Covers

The Chisel test is:

```text
jamba.top.Jamba2MiniTileSpec
```

The end-to-end case is:

```text
should run a two-token end-to-end demo trace against Python golden values
```

It covers:

- writing a value through the `WeightStoreMini` shell
- asserting `clear` and checking that loaded weights are preserved
- feeding two tokens through `Jamba2MiniTile`
- checking registered output tokens against Python golden values
- checking layer 0 SSM state debug values
- checking the attention layer KV cache write index and valid count

## Demo Tokens

The current fixture uses the debug config:

```text
numLayers = 4
attentionLayerPeriod = 4
attentionLayerOffset = 3
contextLength = 8
```

The two demo tokens are:

```text
t0 = [1, 0, 0, 0]
t1 = [2, 0, 0, 0]
```

The expected registered tile outputs are:

```text
y0 = [3, 0, 0, 3]
y1 = [6, 0, 0, 3]
```

The expected layer 0 SSM states after each token are:

```text
h0_after_t0 = [2, 0, 0, 0]
h0_after_t1 = [8, 0, 0, 0]
```

The attention layer is layer 3. Its expected KV cache progress is:

```text
after t0: writeIndex = 1, validCount = 1
after t1: writeIndex = 2, validCount = 2
```

## Python Golden

The matching Python helper is:

```text
jamba2_mini_tile_demo_trace
```

It intentionally models the current Chisel-visible timing:

- Mamba mixer output uses the visible old SSM state.
- The next SSM state is observed after the clock edge.
- Attention uses write-through behavior for the token accepted in the current cycle.
- Narrowing points match the current mini Chisel datapath.

This is separate from the architecture-level `jamba2_mini_core_trace`, which is kept as a higher-level behavioral reference.

## Run The Demo

Run only the top-level demo tests:

```bash
sbt "testOnly jamba.top.Jamba2MiniTileSpec"
```

Run Python golden tests:

```bash
.venv/bin/python3 -m pytest python/tests/ -v
```

Run the full project verification:

```bash
./scripts/run_test.sh
```

## Current Limitation

`Jamba2MiniTile` uses deterministic demo weights by default. It can also select decoded `WeightStoreMini` values with `useLoadedWeights`; the first loaded-weight smoke test writes `mlpDownBias(0)` and verifies that the loaded value changes the end-to-end tile output.
