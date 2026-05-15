#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

missing=0

check_cmd() {
  local cmd="$1"
  local hint="$2"

  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "missing: $cmd ($hint)" >&2
    missing=1
  fi
}

check_cmd java "install JDK 17"
check_cmd sbt "install sbt"
check_cmd python3 "install Python 3"
check_cmd pip "install Python pip"
check_cmd verilator "install Verilator"
check_cmd git "install Git"

if [ "$missing" -ne 0 ]; then
  echo ""
  echo "Environment check failed. Install the missing tools above and rerun ./scripts/check_env.sh." >&2
  exit 1
fi

print_version() {
  local label="$1"
  shift

  printf "%-10s " "$label:"
  "$@" 2>&1 | head -n 1
}

echo "=== Environment check ==="
print_version java java -version
print_version sbt sbt --script-version
print_version python3 python3 --version
print_version pip pip --version
print_version verilator verilator --version
print_version git git --version
echo "Environment check passed."
