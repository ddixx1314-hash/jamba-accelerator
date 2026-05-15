# Limitations

This project is a Chisel Jamba2 Mini style hardware accelerator prototype. It is useful for architecture learning, deterministic verification, and generated-Verilog experiments, but it is not a production inference accelerator.

## Current Capability Boundary

The current `v0.1` prototype supports:

- a formal `Jamba2MiniTile` top
- token valid/ready input and output
- `start`, `clear`, and `enableMoE` command controls
- `busy`, `done`, and `error` status outputs
- Jamba2-style `Mixer + MLP` layers
- Mamba and Attention mixer paths
- default sparse attention schedule through `attentionLayerPeriod = 8`
- token-serial SSM scan
- circular sliding-window KV cache
- shift-based approximate attention normalization
- Dense MLP and two-expert MoE-lite path
- a small weight load/read shell
- Python golden tests and Chisel tests
- SystemVerilog generation and Verilator lint
- lightweight generated-Verilog scale analysis

## Not Supported Yet

The current prototype does not support:

- real Jamba2 3B or Jamba2 Mini checkpoint execution
- Hugging Face checkpoint loading
- production model quality
- BF16, FP16, or production quantization formats
- AXI4, AXI-Stream, DMA, DDR, HBM, or a production memory hierarchy
- board bring-up or FPGA timing closure
- ASIC synthesis, place-and-route, timing, power, or area signoff
- true production softmax
- full grouped-query attention implementation
- full production MoE routing and many-expert scheduling
- wide hidden dimensions beyond the current 4-lane datapath
- stored-weight decoding into every typed core port

## Known Architectural Simplifications

- `lanes = 4` is fixed by current `Linear4`-based datapaths.
- `RmsNormApprox` uses integer mean-square division, not production RMSNorm.
- The top-level weight store is loadable and readable, but the core still uses deterministic demo weights.
- Scale analysis is based on generated Verilog size and structural counts, not synthesis resources.
- Some debug outputs are live combinational views, while `out` is a registered tile output.

## Recommended Next Work

The highest-value next steps are:

- decode `WeightStoreMini` into typed layer/core weight ports
- replace ad hoc integer narrowing with the shared fixed-point helpers in more datapaths
- parameterize beyond 4 lanes
- add a closer GQA-style attention interface
- add a larger end-to-end fixture once stored weights drive the core
- add a synthesis-backed resource report when a stable toolchain is available
