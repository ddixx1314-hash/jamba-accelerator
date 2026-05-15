# Weight Layout

This document defines the first mini weight-storage plan.

The current implementation adds `WeightStoreMini`, a small register-file-backed storage block with:

```text
writeValid
writeReady
writeAddr
writeData
readAddr
readData
```

`clear` does not erase weights. This matches accelerator usage: clearing token/state execution should not reload model weights.

## Address Map Draft

The first address map is a compact draft for future `Jamba2MiniTile` integration.

| Range | Purpose |
| ---: | --- |
| `0-15` | RMSNorm and layer norm weights |
| `16-63` | Mamba mixer projection weights and biases |
| `64-95` | Mamba SSM parameters and convolution kernels |
| `96-143` | Attention Q/K/V/O projection weights and biases |
| `144-191` | Dense MLP weights and biases |
| `192-255` | MoE-lite router and expert weights |

The first hardware block does not yet decode this full map into module ports. It establishes the load/read behavior that the future tile will use.

## Semantics

- Writes occur when `writeValid && writeReady`.
- `writeReady` is always true in the first implementation.
- Reads are combinational by `readAddr`.
- Reset initializes all weights to zero.
- `clear` preserves weights.

## Future Work

Later stages should add:

- typed decode helpers for each weight range
- bulk fixture loading in Chisel tests
- connection from `WeightStoreMini` into `Jamba2MiniTile`
- optional wider packed weight words
