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
PROFILE_DIR=${PROFILE_DIR:-~/tmp/fused_kernel_profiles}

if [ -z "${FUSED_KERNEL_DIR:-}" ]; then
    echo "Usage: FUSED_KERNEL_DIR=/tmp/kernels ./run_benchmark_fused_kernels.sh"
    exit 1
fi

export PYTORCH_ALLOC_CONF="expandable_segments:True"
mkdir -p "$PROFILE_DIR"
TMPLOG=$(mktemp)

run_bench() {
    local label="$1"
    local trace_subdir="$2"
    shift 2
    echo ""
    echo "=========================================="
    echo "  $label"
    echo "=========================================="
    local dump="$PROFILE_DIR/$trace_subdir"
    rm -rf "$dump"
    NGPU=$NGPU LOCAL_RANK=0 python3 -m torchtitan.train \
        --module graph_trainer.llama3 \
        --config graph_trainer_llama3_8b \
        --compile.mode aot_fx_trace \
        --parallelism.data_parallel_shard_degree=4 \
        --parallelism.tensor_parallel_degree=2 \
        --training.steps $STEPS \
        --dataloader.dataset c4_test \
        \
        --metrics.no-enable_tensorboard \
        --comm.trace_buf_size=0 \
        --comm.mode=fake_backend \
        --profiler.enable_profiling \
        --profiler.profile_freq 10 \
        --dump_folder "$dump" \
        "$@" \
        > "$TMPLOG" 2>&1
    grep -E "step:|Fused kernel|Dumping profiler" "$TMPLOG" || echo "  (no step output found)"
    local trace=$(find "$dump" -name "rank0_*" -type f 2>/dev/null | head -1)
    if [ -n "$trace" ]; then
        echo "  Trace: $trace"
    fi
}

run_bench "BASELINE (no fused kernels)" "baseline"
run_bench "WITH FUSED KERNELS ($FUSED_KERNEL_DIR)" "fused" --compile.fused_kernel_dir "$FUSED_KERNEL_DIR"

echo ""
echo "=========================================="
echo "  Uploading traces to Perfetto"
echo "=========================================="
for subdir in baseline fused; do
    TRACE_FILE=$(find "$PROFILE_DIR/$subdir" -name "rank0_*" -type f 2>/dev/null | head -1)
    if [ -n "$TRACE_FILE" ]; then
        echo "  Uploading $subdir: $TRACE_FILE"
        python3 ~/local/fbsource/arvr/scripts/perfetto/share_trace.py "$TRACE_FILE" 2>&1 | grep -E "Perfetto UI|Manifold"
    else
        echo "  $subdir: no trace found"
    fi
done

rm -f "$TMPLOG"
