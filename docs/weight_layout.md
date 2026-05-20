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

`SequentialWeightLoaderMini` builds on that address generator and walks every element of one requested field in address order. It emits one valid address at a time and advances only when `outReady` is asserted.

## Address Map

The first address map is compact and shared by all mini layers in `Jamba2MiniTile`.
`UnifiedJamba2MiniFullTile` extends the same map with a fixed per-layer stride:

```text
layer_base = layer_index * 512
physical_address = layer_base + local_address
```

For example, layer 0 `mlpDownBias[0]` is address `232`, while layer 1 `mlpDownBias[0]` is address `744`.

The stride was increased from 256 to **512** to accommodate the six MoE expert weight fields
added in Phase A. Each layer now occupies 366 addresses; addresses 366–511 within each layer
are reserved for future fields.

| Range | Purpose |
| ---: | --- |
| `0-3` | `norm1Weight` |
| `4-7` | `norm2Weight` |
| `8-15` | reserved |
| `16-31` | `mambaInputWeight`, row-major 4×4 |
| `32-35` | `mambaInputBias` |
| `36-51` | `mambaBWeight`, row-major 4×4 |
| `52-55` | `mambaBBias` |
| `56-71` | `mambaCWeight`, row-major 4×4 |
| `72-75` | `mambaCBias` |
| `76-79` | `mambaA` |
| `80-95` | `mambaKernel`, tap-major 4×4 for `convTaps = 4` |
| `96-111` | `qWeight`, row-major 4×4 |
| `112-115` | `qBias` |
| `116-131` | `kWeight`, row-major 4×4 |
| `132-135` | `kBias` |
| `136-151` | `vWeight`, row-major 4×4 |
| `152-155` | `vBias` |
| `156-171` | `attentionOutWeight`, row-major 4×4 |
| `172-175` | `attentionOutBias` |
| `176-191` | `mlpGateWeight`, row-major 4×4 |
| `192-195` | `mlpGateBias` |
| `196-211` | `mlpUpWeight`, row-major 4×4 |
| `212-215` | `mlpUpBias` |
| `216-231` | `mlpDownWeight`, row-major 4×4 |
| `232-235` | `mlpDownBias` |
| `236-243` | `routerWeight`, expert-major 2×4 |
| `244-245` | `routerBias` |
| `246-277` | `expertGateWeight`, expert-major 2×(4×4), row-major within each expert |
| `278-285` | `expertGateBias`, expert-major 2×4 |
| `286-317` | `expertUpWeight`, expert-major 2×(4×4) |
| `318-325` | `expertUpBias`, expert-major 2×4 |
| `326-357` | `expertDownWeight`, expert-major 2×(4×4) |
| `358-365` | `expertDownBias`, expert-major 2×4 |
| `366-511` | reserved |

`UnifiedJamba2MiniFullTile` uses `LayeredWeightStoreMini` to decode writes into per-layer
banks, then uses `scheduler.activeLayer` to select the active layer's field values and
drive the typed ports of the multi-layer scheduler. All six expert weight fields are now
decoded and wired through `connectLoadedWeights`. The `SequentialWeightLoadPathMini`
module provides an alternative BRAM-style element-by-element loading path for any field.

## Semantics

- Writes occur when `writeValid && writeReady`.
- `writeReady` is always true in the first implementation.
- Reads are combinational by `readAddr`.
- `readAll` is an internal decode bus used by `Jamba2MiniTile`.
- `LayeredWeightStoreMini` does not expose `readAll`; it decodes fields directly to typed active-layer outputs.
- `WeightAddressGenMini` maps `(layer, field, row, col, lane, tap, expert)` into the same flat address space and reports whether the field is accumulator-width or data-width.
- `SequentialWeightLoaderMini` maps `(layer, field)` into a ready/valid address stream over all field elements. Matrix fields are row-major, kernels are tap-major, and router fields are expert-major.
- Reset initializes all weights to zero.
- `clear` preserves weights.

## Future Work

- Bulk fixture loading in Chisel tests (currently weights are poked element-by-element in specs)
- Optional wider packed weight words (e.g. 2 elements per write to halve load cycles)
- True single-physical-layer reuse across all logical layers (`SinglePhysicalLayerTile`)
  so that MAC count is independent of `numLayers`
