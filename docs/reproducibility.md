# Reproducibility

This document describes how to reproduce the current mini accelerator build and verification flow.

## Required Tools

The project expects these tools to be available on `PATH`:

- JDK 17 for Scala/Chisel compilation.
- SBT for running Chisel tests and Verilog generation.
- Python 3 and pip for golden-model tests.
- Verilator for SystemVerilog lint.
- Git for source control checks.

Check the local environment with:

```bash
./scripts/check_env.sh
```

This script only prints tool versions and reports missing commands. It does not install packages, create virtual environments, or generate files.

## First Run

From the repository root:

```bash
./scripts/run_test.sh
```

The full verification script runs:

```text
sbt test
python3 -m pytest python/tests/ -v
sbt "runMain jamba.top.GenerateVerilog"
verilator --lint-only generated/verilog/Jamba2MiniTile.sv
verilator --lint-only generated/verilog/JambaMiniTile.sv
verilator --lint-only generated/verilog/Jamba2MiniAccelerator.sv
verilator --lint-only generated/verilog/Jamba2MiniCore.sv
verilator --lint-only generated/verilog/Jamba2MiniStream.sv
./scripts/scale_analysis.sh
./scripts/resource_reuse_analysis.sh
```

If `.venv/` does not exist, `run_test.sh` creates it and installs `requirements.txt`.

## Generated Files

Generated SystemVerilog is written to:

```text
generated/verilog/
generated/scale/
generated/reports/
generated/resource_reuse/
```

The expected generated Verilog tops are:

```text
Jamba2MiniTile.sv
JambaMiniTile.sv
Jamba2MiniAccelerator.sv
Jamba2MiniCore.sv
Jamba2MiniStream.sv
```

These files are ignored by git because they are reproducible from the Chisel source.

## Common Failures

- `java` missing or wrong version: install JDK 17 and ensure `java -version` reports it.
- `sbt` missing: install SBT and rerun `./scripts/check_env.sh`.
- Python import errors: rerun `./scripts/run_test.sh` so the local `.venv/` is created and dependencies are installed.
- Verilator lint failure: regenerate Verilog with `./scripts/generate_verilog.sh`, then rerun `./scripts/run_test.sh`.
- Stale generated output: remove `generated/verilog/` and rerun `./scripts/generate_verilog.sh`.

## Clean Local Artifacts

The following paths are local build or cache artifacts and should not be committed:

```text
.venv/
target/
generated/
test_run_dir/
.pytest_cache/
```

Before committing, run:

```bash
git status --short --branch
git diff --check
```

The engineering repo currently has no GitHub remote configured by default.
