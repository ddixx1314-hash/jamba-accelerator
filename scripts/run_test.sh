#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "=== Running Chisel tests ==="
sbt test

echo ""
echo "=== Running Python golden model tests ==="
source .venv/bin/activate
python3 -m pytest python/tests/ -v
