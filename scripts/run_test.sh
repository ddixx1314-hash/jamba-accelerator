#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

./scripts/check_env.sh

echo ""
echo "=== Running Chisel tests ==="
sbt test

echo ""
echo "=== Running Python golden model tests ==="
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python3 -m pip install -q -r requirements.txt
python3 -m pytest python/tests/ -v

echo ""
echo "=== Generating SystemVerilog ==="
./scripts/generate_verilog.sh

echo ""
echo "=== Verilator lint ==="
verilator --lint-only generated/verilog/UnifiedJamba2MiniAcceleratorTile.sv
verilator --lint-only generated/verilog/UnifiedJamba2MiniTileScheduler.sv
verilator --lint-only generated/verilog/Jamba2MiniTile.sv
verilator --lint-only generated/verilog/JambaMiniTile.sv
verilator --lint-only generated/verilog/Jamba2MiniAccelerator.sv
verilator --lint-only generated/verilog/Jamba2MiniCore.sv
verilator --lint-only generated/verilog/Jamba2MiniStream.sv

echo ""
echo "=== Scale analysis ==="
./scripts/scale_analysis.sh

echo ""
echo "=== Resource reuse analysis ==="
./scripts/resource_reuse_analysis.sh
