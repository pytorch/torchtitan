#!/bin/bash

PROFILE_DIR=~/tmp/profile_traces
rm --preserve-root=all --one-file-system -rf "$PROFILE_DIR"/*

tlp ()
{
    rm --preserve-root=all --one-file-system -rf ~/tmp/trace_logs/*;
    TORCH_TRACE=~/tmp/trace_logs "$@";
    tlparse ~/tmp/trace_logs/*rank_0* --overwrite-manifold
}

# graph_trainer_llama3_debugmodel
# --- Llama3 debugmodel (FSDP only) ---
NGPU=8 MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_debugmodel tlp ./run_train.sh \
    --compile.mode aot_fx_trace \
    --compile.inductor_compilation full \
    --parallelism.data_parallel_shard_degree=4 \
    --parallelism.tensor_parallel_degree=2 \
    --training.steps 10 \
    --dataloader.dataset c4_test \
    --compile.debug_graph_passes \
    --compile.no-enable_cudagraph \
    --profiler.enable_profiling \
    --profiler.profile_freq 10 \
    --dump_folder "$PROFILE_DIR"

# --- DeepSeek-v3 debugmodel EP (FSDP+TP+EP) ---
# NGPU=8 MODULE=graph_trainer.deepseek_v3 CONFIG=graph_trainer_deepseek_v3_debugmodel_ep tlp ./run_train.sh \
#     --compile.mode aot_fx_trace \
#     --parallelism.data_parallel_shard_degree=4 \
#     --parallelism.tensor_parallel_degree=2 \
#     --parallelism.expert_parallel_degree=4 \
#     --parallelism.expert_tensor_parallel_degree=1 \
#     --training.steps 10 \
#     --dataloader.dataset c4_test \
#     --compile.debug_graph_passes \
#     --compile.no-enable_cudagraph \
#     --profiler.enable_profiling \
#     --profiler.profile_freq 10 \
#     --dump_folder "$PROFILE_DIR"

# Upload rank 0 trace to Perfetto
TRACE_FILE=$(find "$PROFILE_DIR/profile_traces" -name "rank0_*" -type f | head -1)
if [ -n "$TRACE_FILE" ]; then
    echo "Uploading $TRACE_FILE"
    python3 ~/local/fbsource/arvr/scripts/perfetto/share_trace.py "$TRACE_FILE"
else
    echo "No rank0 trace found in $PROFILE_DIR/profile_traces"
fi
