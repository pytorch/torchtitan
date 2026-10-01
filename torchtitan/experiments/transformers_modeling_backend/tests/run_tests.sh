#!/bin/bash
# Run all Transformers modeling backend tests.
# Usage: cd torchtitan && bash torchtitan/experiments/transformers_modeling_backend/tests/run_tests.sh [NGPU]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NGPU=${1:-8}
OUTPUT_DIR=$(mktemp -d)

echo "=== Unit tests ==="
../.venv/bin/python -m pytest "$SCRIPT_DIR/test_moe_parallelism.py" -x -v

echo ""
echo "=== Integration tests ==="
python -m torchtitan.experiments.transformers_modeling_backend.tests.integration_tests \
    "$OUTPUT_DIR" --ngpu "$NGPU"

echo ""
echo "=== CP+PP numerical equivalence (logit-level; needs 4 GPUs, self-skips) ==="
python -m torchtitan.experiments.transformers_modeling_backend.tests.cp_pp_numerical
