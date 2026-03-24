#!/bin/bash
# Integration tests for MoE parallelism in the transformers modeling backend.
#
# Usage:
#   cd torchtitan
#   bash torchtitan/experiments/transformers_modeling_backend/tests/run_moe_tests.sh [NGPU]
#
# Default NGPU=8. All configurations use dp_shard=-1 (auto-computed).
# EP borrows from the fsdp*tp product; it does NOT multiply into WORLD_SIZE.

set -euo pipefail

NGPU=${1:-8}
MODULE=transformers_modeling_backend
CONFIG=transformers_modeling_backend_debugmodel_moe
STEPS=${2:-200}
PASSED=0
FAILED=0
FAILURES=""

run_test() {
    local name="$1"
    shift
    echo ""
    echo "================================================================"
    echo "TEST: $name"
    echo "NGPU=$NGPU, args: $*"
    echo "================================================================"

    if NGPU=$NGPU MODULE=$MODULE CONFIG=$CONFIG ./run_train.sh \
        --training.steps $STEPS \
        "$@" 2>&1; then
        echo "PASSED: $name"
        PASSED=$((PASSED + 1))
    else
        echo "FAILED: $name"
        FAILED=$((FAILED + 1))
        FAILURES="$FAILURES\n  - $name"
    fi
}

# ── FSDP-only (no TP, no EP) — baseline ──
run_test "FSDP-only (baseline)" \
    --parallelism.data_parallel_shard_degree -1

# ── EP-only (no TP) ──
run_test "FSDP + EP=2" \
    --parallelism.data_parallel_shard_degree -1 \
    --parallelism.expert_parallel_degree 2

run_test "FSDP + EP=4" \
    --parallelism.data_parallel_shard_degree -1 \
    --parallelism.expert_parallel_degree 4

# ── TP-only MoE (no EP) ──
run_test "FSDP + TP=2 (MoE, no EP)" \
    --parallelism.data_parallel_shard_degree -1 \
    --parallelism.tensor_parallel_degree 2

run_test "FSDP + TP=4 (MoE, no EP)" \
    --parallelism.data_parallel_shard_degree -1 \
    --parallelism.tensor_parallel_degree 4

# ── TP + EP combined ──
run_test "FSDP + TP=2 + EP=2" \
    --parallelism.data_parallel_shard_degree -1 \
    --parallelism.tensor_parallel_degree 2 \
    --parallelism.expert_parallel_degree 2

run_test "FSDP + TP=2 + EP=4" \
    --parallelism.data_parallel_shard_degree -1 \
    --parallelism.tensor_parallel_degree 2 \
    --parallelism.expert_parallel_degree 4

# ── Compile variants ──
run_test "FSDP + EP=4 + compile" \
    --parallelism.data_parallel_shard_degree -1 \
    --parallelism.expert_parallel_degree 4 \
    --compile.enable

run_test "FSDP + TP=2 + EP=2 + compile" \
    --parallelism.data_parallel_shard_degree -1 \
    --parallelism.tensor_parallel_degree 2 \
    --parallelism.expert_parallel_degree 2 \
    --compile.enable

run_test "FSDP + TP=2 (MoE, no EP) + compile" \
    --parallelism.data_parallel_shard_degree -1 \
    --parallelism.tensor_parallel_degree 2 \
    --compile.enable

# ── Pipeline Parallel + MoE ──
run_test "FSDP + PP=2 + EP=2" \
    --parallelism.data_parallel_shard_degree -1 \
    --parallelism.pipeline_parallel_degree 2 \
    --parallelism.pipeline_parallel_schedule 1F1B \
    --parallelism.expert_parallel_degree 2

if [ "$NGPU" -ge 8 ]; then
run_test "FSDP + TP=2 + PP=2 + EP=2" \
    --parallelism.data_parallel_shard_degree -1 \
    --parallelism.tensor_parallel_degree 2 \
    --parallelism.pipeline_parallel_degree 2 \
    --parallelism.pipeline_parallel_schedule 1F1B \
    --parallelism.expert_parallel_degree 2
fi

# ── HSDP (data parallel replicate + shard) + MoE ──
run_test "HSDP + EP=2" \
    --parallelism.data_parallel_replicate_degree 2 \
    --parallelism.data_parallel_shard_degree -1 \
    --parallelism.expert_parallel_degree 2

if [ "$NGPU" -ge 8 ]; then
run_test "HSDP + TP=2 + EP=2" \
    --parallelism.data_parallel_replicate_degree 2 \
    --parallelism.data_parallel_shard_degree -1 \
    --parallelism.tensor_parallel_degree 2 \
    --parallelism.expert_parallel_degree 2
fi

# ── Without SAC (explicit baseline) ──
run_test "FSDP + EP=2 (no SAC)" \
    --parallelism.data_parallel_shard_degree -1 \
    --parallelism.expert_parallel_degree 2 \
    --activation_checkpoint.mode none

run_test "FSDP + TP=2 + EP=2 (no SAC)" \
    --parallelism.data_parallel_shard_degree -1 \
    --parallelism.tensor_parallel_degree 2 \
    --parallelism.expert_parallel_degree 2 \
    --activation_checkpoint.mode none

# ── PP + compile ──
run_test "FSDP + PP=2 + EP=2 + compile" \
    --parallelism.data_parallel_shard_degree -1 \
    --parallelism.pipeline_parallel_degree 2 \
    --parallelism.pipeline_parallel_schedule 1F1B \
    --parallelism.expert_parallel_degree 2 \
    --compile.enable

# ── Summary ──
echo ""
echo "================================================================"
echo "RESULTS: $PASSED passed, $FAILED failed"
if [ $FAILED -gt 0 ]; then
    echo -e "Failed tests:$FAILURES"
    exit 1
else
    echo "All tests passed."
fi
