#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p generated/verilog
sbt "runMain jamba.top.GenerateVerilog"

echo "Generated SystemVerilog in generated/verilog"
