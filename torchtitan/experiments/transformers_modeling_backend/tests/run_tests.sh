#!/bin/bash
# Run all MoE tests: unit tests + integration tests.
# Usage: cd torchtitan && bash torchtitan/experiments/transformers_modeling_backend/tests/run_tests.sh [NGPU] [STEPS]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NGPU=${1:-8}
STEPS=${2:-2}

echo "=== Unit tests ==="
../.venv/bin/python -m pytest "$SCRIPT_DIR/test_moe_parallelism.py" -x -v

echo ""
echo "=== Integration tests ==="
bash "$SCRIPT_DIR/run_moe_tests.sh" "$NGPU" "$STEPS"
