# Release v0.1

Release tag:

```text
jamba2-mini-accelerator-v0.1
```

## Summary

`v0.1` is the first complete architecture-level Chisel prototype of the Jamba2 Mini style accelerator in this repository.

It is not checkpoint-compatible with real Jamba2 models. Its purpose is to provide a verified, reproducible, readable hardware foundation that captures the key structural ideas:

- `Mixer + MLP` in every formal layer
- Mamba-major sparse attention scheduling
- token-serial SSM scan
- circular KV cache
- dense MLP plus MoE-lite path
- token stream top-level shell
- weight load/read shell
- Python golden and Chisel deterministic trace checks
- generated SystemVerilog and Verilator lint

## Included In This Release

- `Jamba2MiniTile` formal top
- `Jamba2MiniHybridCore`
- `Jamba2MiniLayer`
- `Jamba2MambaMixerMini`
- `AttentionMixerMini`
- `DenseMLPMini`
- `MlpPathMini`
- `RouterMini`, `ExpertMLPMini`, and `MoELiteMini`
- `WeightStoreMini`
- `GenerateVerilog`
- `GenerateScaleSweep`
- full regression script: `./scripts/run_test.sh`
- scale analysis script: `./scripts/scale_analysis.sh`

## Verification

The release acceptance command is:

```bash
./scripts/run_test.sh
```

It runs:

- Chisel tests
- Python golden tests
- SystemVerilog generation
- Verilator lint for generated tops
- lightweight scale analysis

## Documentation

Important documents:

- `docs/architecture.md`
- `docs/interface.md`
- `docs/jamba2_mini_spec.md`
- `docs/fixed_point.md`
- `docs/moe_lite.md`
- `docs/weight_layout.md`
- `docs/demo.md`
- `docs/scale_analysis.md`
- `docs/limitations.md`
- `docs/reproducibility.md`

## Known Limits

See `docs/limitations.md`.
