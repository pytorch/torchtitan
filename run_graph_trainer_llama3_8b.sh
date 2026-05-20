#!/bin/bash

PROFILE_DIR=~/tmp/profile_traces
rm --preserve-root=all --one-file-system -rf "$PROFILE_DIR"/*

TLPARSE_OUTPUT_DIR=~/tmp/tlparse_output
LOG_DIR=~/tmp/train_logs
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"


tlp ()
{
    rm --preserve-root=all --one-file-system -rf ~/tmp/trace_logs/*;
    TORCH_TRACE=~/tmp/trace_logs "$@";
    tlparse ~/tmp/trace_logs/*rank_0* --overwrite-manifold;
    tlparse parse ~/tmp/trace_logs/*rank_0* -o "$TLPARSE_OUTPUT_DIR" --overwrite;
    "$SCRIPT_DIR/scripts/upload_tlparse_passes.sh" "$TLPARSE_OUTPUT_DIR"
}



# graph_trainer_llama3_debugmodel
    # --compile.disable_passes cudagraph_pass \

# --- Llama3 debugmodel (FSDP only) ---
# NGPU=8 MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_debugmodel tlp ./run_train.sh \
#     --compile.mode aot_fx_trace \
#     --parallelism.data_parallel_shard_degree=4 \
#     --parallelism.tensor_parallel_degree=2 \
#     --training.steps 10 \
#     --dataloader.dataset c4_test \
#     --compile.debug_graph_passes \
#     --profiler.enable_profiling \
#     --profiler.profile_freq 10 \
#     --dump_folder "$PROFILE_DIR" \
#     --debug.print-config \
#     --compile.memory_policy=sac_and_offload \
#     2>&1 | tee "$LOG_FILE"


# --- Llama3 8B (FSDP only, 8 GPUs) ---
# Fused kernel workflow (single config, single command):
#   FUSED_KERNEL_DIR=/tmp/kernels ./run_graph_trainer_llama3_8b.sh
#   - First run: extracts problems + trains with eager fallback
#   - After offline kernel gen: same command auto-picks up kernels
FUSED_KERNEL_FLAGS=""
if [ -n "${FUSED_KERNEL_DIR:-}" ]; then
    FUSED_KERNEL_FLAGS="--compile.fused_kernel_dir $FUSED_KERNEL_DIR"
fi

export PYTORCH_ALLOC_CONF="expandable_segments:True"
COMMON_FLAGS="\
    --compile.mode aot_fx_trace \
    --parallelism.data_parallel_shard_degree=8 \
    --parallelism.tensor_parallel_degree=1 \
    --training.steps 10 \
    --dataloader.dataset c4_test \
    --compile.debug_graph_passes \
    --compile.disable_passes cudagraph_pass \
    --profiler.enable_profiling \
    --profiler.profile_freq 10 \
    --dump_folder $PROFILE_DIR \
    --debug.print-config \
    --compile.memory_policy=default \
    $FUSED_KERNEL_FLAGS"

if [ "${FAKE_BACKEND:-0}" = "1" ]; then
    # Single GPU, no NCCL — collectives are no-ops.
    # Useful for isolating compute time or running without 8 GPUs.
    tlp NGPU=8 LOCAL_RANK=0 python3 -m torchtitan.train \
        --module graph_trainer.llama3 --config graph_trainer_llama3_8b \
        --comm.mode=fake_backend \
        $COMMON_FLAGS \
        2>&1 | tee "$LOG_FILE"
else
    NGPU=8 MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b tlp ./run_train.sh \
        $COMMON_FLAGS \
        2>&1 | tee "$LOG_FILE"
fi

# --- DeepSeek-v3 debugmodel EP (FSDP+TP+EP) ---
# NGPU=8 MODULE=graph_trainer.deepseek_v3 CONFIG=graph_trainer_deepseek_v3_debugmodel_ep tlp ./run_train.sh \
#     --compile.mode aot_fx_trace \
#     --parallelism.data_parallel_shard_degree=4 \
#     --parallelism.tensor_parallel_degree=2 \
#     --parallelism.expert_parallel_degree=4 \
#     --training.steps 10 \
#     --dataloader.dataset c4_test \
#     --compile.debug_graph_passes \
#     --profiler.enable_profiling \
#     --profiler.profile_freq 10 \
#     --dump_folder "$PROFILE_DIR" \
#     --debug.print-config \
#     --debug.enable_structured_logging \
#     --compile.memory_policy=sac_and_offload \
#     2>&1 | tee "$LOG_FILE"

# --- DeepSeek-v3 16B (FSDP+TP+EP) ---
# NGPU=8 MODULE=graph_trainer.deepseek_v3 CONFIG=graph_trainer_deepseek_v3_16b tlp ./run_train.sh \
#     --compile.mode aot_fx_trace \
#     --parallelism.data_parallel_shard_degree=8 \
#     --parallelism.tensor_parallel_degree=1 \
#     --parallelism.expert_parallel_degree=8 \
#     --training.steps 20 \
#     --dataloader.dataset c4_test \
#     --compile.debug_graph_passes \
#     --compile.disable_passes cudagraph_pass \
#     --profiler.enable_profiling \
#     --profiler.profile_freq 10 \
#     --dump_folder "$PROFILE_DIR" \
#     --debug.print-config \
#     --compile.memory_policy=sac_and_offload \
#     2>&1 | tee "$LOG_FILE"

echo "Run log saved to $LOG_FILE"

# Upload rank 0 trace to Perfetto
{
TRACE_FILE=$(find "$PROFILE_DIR" -name "rank0_*" -type f | head -1)
if [ -n "$TRACE_FILE" ]; then
    echo "Uploading $TRACE_FILE"
    python3 ~/local/fbsource/arvr/scripts/perfetto/share_trace.py "$TRACE_FILE"
else
    echo "No rank0 trace found in $PROFILE_DIR"
fi
} 2>&1 | tee -a "$LOG_FILE"

# Upload log to pastry
PASTRY_LINK=$(cat "$LOG_FILE" | pastry)
echo "Pastry link: $PASTRY_LINK"
