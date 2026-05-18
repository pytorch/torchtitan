#!/bin/bash
# Benchmark fused kernels: compare baseline vs fused kernel training.
#
# Uses --comm.mode=fake_backend to replace all collectives with no-ops,
# isolating pure compute time. Runs on a single GPU simulating 8.
#
# Usage:
#   FUSED_KERNEL_DIR=/tmp/kernels ./run_benchmark_fused_kernels.sh

STEPS=${STEPS:-20}
NGPU=${NGPU:-8}

if [ -z "${FUSED_KERNEL_DIR:-}" ]; then
    echo "Usage: FUSED_KERNEL_DIR=/tmp/kernels ./run_benchmark_fused_kernels.sh"
    exit 1
fi

export PYTORCH_ALLOC_CONF="expandable_segments:True"
TMPLOG=$(mktemp)

run_bench() {
    local label="$1"
    shift
    echo ""
    echo "=========================================="
    echo "  $label"
    echo "=========================================="
    NGPU=$NGPU LOCAL_RANK=0 python3 -m torchtitan.train \
        --module graph_trainer.llama3 \
        --config graph_trainer_llama3_8b \
        --compile.mode aot_fx_trace \
        --parallelism.data_parallel_shard_degree=4 \
        --parallelism.tensor_parallel_degree=2 \
        --training.steps $STEPS \
        --dataloader.dataset c4_test \
        --compile.disable_passes cudagraph_pass \
        --compile.memory_policy=sac_and_offload \
        --metrics.no-enable_tensorboard \
        --profiler.no-enable_profiling \
        --comm.trace_buf_size=0 \
        --comm.mode=fake_backend \
        "$@" \
        > "$TMPLOG" 2>&1
    grep -E "step:|Fused kernel" "$TMPLOG" || echo "  (no step output found)"
}

run_bench "BASELINE (no fused kernels)"
run_bench "WITH FUSED KERNELS ($FUSED_KERNEL_DIR)" --compile.fused_kernel_dir "$FUSED_KERNEL_DIR"

rm -f "$TMPLOG"
