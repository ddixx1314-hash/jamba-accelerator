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
readAll
```

`clear` does not erase weights. This matches accelerator usage: clearing token/state execution should not reload model weights.

`UnifiedJamba2MiniFullTile` uses `LayeredWeightStoreMini` instead of `WeightStoreMini.readAll`. It keeps the same software-visible flat write/read address space, but internally decodes known fields into per-layer banks and selects only the active layer's field values.

`WeightAddressGenMini` uses the same map to generate one flat address per requested field element. This is the first building block for a later BRAM-style sequential field loader.

## Address Map

The first address map is compact and shared by all mini layers in `Jamba2MiniTile`.
`UnifiedJamba2MiniFullTile` extends the same map with a fixed per-layer stride:

```text
layer_base = layer_index * 256
physical_address = layer_base + local_address
```

For example, layer 0 `mlpDownBias[0]` is address `232`, while layer 1 `mlpDownBias[0]` is address `488`.

| Range | Purpose |
| ---: | --- |
| `0-3` | `norm1Weight` |
| `4-7` | `norm2Weight` |
| `8-15` | reserved |
| `16-31` | `mambaInputWeight`, row-major 4x4 |
| `32-35` | `mambaInputBias` |
| `36-51` | `mambaBWeight`, row-major 4x4 |
| `52-55` | `mambaBBias` |
| `56-71` | `mambaCWeight`, row-major 4x4 |
| `72-75` | `mambaCBias` |
| `76-79` | `mambaA` |
| `80-95` | `mambaKernel`, tap-major 4x4 for `convTaps = 4` |
| `96-111` | `qWeight`, row-major 4x4 |
| `112-115` | `qBias` |
| `116-131` | `kWeight`, row-major 4x4 |
| `132-135` | `kBias` |
| `136-151` | `vWeight`, row-major 4x4 |
| `152-155` | `vBias` |
| `156-171` | `attentionOutWeight`, row-major 4x4 |
| `172-175` | `attentionOutBias` |
| `176-191` | `mlpGateWeight`, row-major 4x4 |
| `192-195` | `mlpGateBias` |
| `196-211` | `mlpUpWeight`, row-major 4x4 |
| `212-215` | `mlpUpBias` |
| `216-231` | `mlpDownWeight`, row-major 4x4 |
| `232-235` | `mlpDownBias` |
| `236-243` | `routerWeight`, expert-major 2x4 |
| `244-245` | `routerBias` |
| `246-255` | reserved for MoE expert weights |

`Jamba2MiniTile` uses deterministic demo weights by default. When `useLoadedWeights` is true, this address map drives the typed ports of `Jamba2MiniHybridCore`. `UnifiedJamba2MiniFullTile` uses `LayeredWeightStoreMini` to decode writes into per-layer banks, then uses `scheduler.activeLayer` to select the active layer segment and drive the typed ports of the multi-layer scheduler. Expert MoE weights are still supplied by the deterministic fixture in the first loaded-weight integration.

## Semantics

- Writes occur when `writeValid && writeReady`.
- `writeReady` is always true in the first implementation.
- Reads are combinational by `readAddr`.
- `readAll` is an internal decode bus used by `Jamba2MiniTile`.
- `LayeredWeightStoreMini` does not expose `readAll`; it decodes fields directly to typed active-layer outputs.
- `WeightAddressGenMini` maps `(layer, field, row, col, lane, tap, expert)` into the same flat address space and reports whether the field is accumulator-width or data-width.
- Reset initializes all weights to zero.
- `clear` preserves weights.

## Future Work

Later stages should add:

- bulk fixture loading in Chisel tests
- expert-weight address decode
- optional wider packed weight words
- BRAM-style sequential field loading so every field does not need to be visible as a parallel typed output
