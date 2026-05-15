# Scale Analysis

This document describes the lightweight scale/resource analysis flow for the Jamba2 Mini Chisel prototype.

This stage does not require FPGA or ASIC synthesis tools. It uses Chisel elaboration plus generated SystemVerilog statistics as an early architecture-level proxy.

## Run

From the repository root:

```bash
./scripts/scale_analysis.sh
```

The script:

1. Generates several `Jamba2MiniTile` variants into `generated/scale/`.
2. Counts generated Verilog size metrics.
3. Writes a markdown report to `generated/reports/scale_analysis.md`.

## Sweep Cases

Current cases:

| Case | Meaning |
| --- | --- |
| `Jamba2MiniTile_Debug4L_Context8` | default debug tile, 4 layers, context 8 |
| `Jamba2MiniTile_Debug4L_Context16` | debug tile with larger KV cache |
| `Jamba2MiniTile_Debug4L_AttnPeriod2` | smoke/debug attention every 2 layers |
| `Jamba2MiniTile_Formal8L_Context16` | default formal mini profile, 8 layers, context 16 |
| `Jamba2MiniTile_Formal8L_Context32` | formal mini profile with larger KV cache |

All cases keep `lanes = 4`, because the current child modules still rely on `Linear4`-based datapaths.

## Report Metrics

The generated report includes:

- byte size of each generated `.sv`
- line count
- module declaration count
- `always_ff` count
- continuous `assign` count
- `reg` declaration count

These metrics are useful for tracking relative growth across configurations. They are not equivalent to LUT, FF, BRAM, SRAM, DSP, timing, or power numbers.

## Example Trend

The first run shows the expected direction:

- increasing `contextLength` increases generated register declarations because the KV cache is larger
- moving from 4 debug layers to 8 formal layers increases generated Verilog size
- changing attention period alone does not necessarily change generated size much, because the current hardware instantiates both mixer paths and selects by control

## Limitations

- No synthesis is run.
- No place-and-route or timing closure is attempted.
- `hiddenSize` is not swept yet because the current datapath is fixed to 4 lanes.
- MoE expert count is not swept yet because the first MoE-lite path supports exactly two experts.

Future work can add synthesis-backed reports if a stable local toolchain is available.
