#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p generated/verilog
sbt "runMain basic.GenerateVerilog"

echo "Generated SystemVerilog in generated/verilog"
