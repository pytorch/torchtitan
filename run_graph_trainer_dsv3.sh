#!/bin/bash

PROFILE_DIR=~/tmp/profile_traces
TLPARSE_OUTPUT_DIR=~/tmp/tlparse_output
LOG_DIR=~/tmp/train_logs
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"

rm --preserve-root=all --one-file-system -rf "$PROFILE_DIR"/*

echo "Logging to $LOG_FILE"

# Upload before/after tlparse pass pairs (and standalone artifacts) to pastry.
# Body runs in a subshell ( ) so its `set -euo pipefail` and the `exit 1` error
# path stay contained to this function and don't abort the surrounding run.
upload_tlparse_passes ()
(
    set -euo pipefail
    DIR="${1:?Usage: upload_tlparse_passes <tlparse_output_dir>}"
    ARTIFACT_DIR="$DIR/-_-_-_-"

    if [ ! -d "$ARTIFACT_DIR" ]; then
        echo "Error: $ARTIFACT_DIR not found"
        exit 1
    fi

    echo "=== Uploading tlparse pass pairs from $ARTIFACT_DIR ==="
    echo ""

    # Upload the initial traced graph first
    traced="$ARTIFACT_DIR/make_fx_graph_traced_0.txt"
    if [ -f "$traced" ]; then
        url=$(cat "$traced" | pastry -t "make_fx_graph_traced_0" -l python -q 2>/dev/null)
        echo "make_fx_graph_traced_0: $url"
    fi

    # Find all before/after pairs by extracting pass names from before_ files
    for before_file in "$ARTIFACT_DIR"/before_*_pass_*.txt; do
        [ -f "$before_file" ] || continue

        basename=$(basename "$before_file")
        # Extract pass name: before_<pass_name>_pass_<N>.txt -> <pass_name>
        pass_name=$(echo "$basename" | sed 's/^before_\(.*\)_pass_[0-9]*\.txt$/\1/')

        # Find the matching after file
        after_file=$(ls "$ARTIFACT_DIR"/after_${pass_name}_pass_*.txt 2>/dev/null | head -1)
        if [ -z "$after_file" ]; then
            echo "WARN: no after file for $pass_name, skipping"
            continue
        fi
        after_basename=$(basename "$after_file")

        if diff -q "$before_file" "$after_file" > /dev/null 2>&1; then
            echo ""
            echo "--- $pass_name --- (no changes, skipping)"
            continue
        fi

        before_output=$(cat "$before_file" | pastry -t "$basename" -l python -q 2>/dev/null)
        after_output=$(cat "$after_file" | pastry -t "$after_basename" -l python -q 2>/dev/null)

        before_paste=$(echo "$before_output" | grep -oP 'P\K[0-9]+' | head -1)
        after_paste=$(echo "$after_output" | grep -oP 'P\K[0-9]+' | head -1)

        diff_url="https://www.internalfb.com/intern/diffing/?before_paste_number=${before_paste}&after_paste_number=${after_paste}&regex_remove_pattern=&enable_regex_remove=0&strip_empty_lines=0&line_wrap=0&selected_tab=plain_diff"

        echo "$pass_name : $diff_url"
    done

    # Upload standalone artifacts (not before/after pairs)
    echo ""
    echo "=== Standalone artifacts ==="
    for f in "$ARTIFACT_DIR"/activation_memory_policy_*.txt \
             "$ARTIFACT_DIR"/fx_codegen_*.txt \
             "$ARTIFACT_DIR"/fx_collectives_analytical_estimation_*.txt \
             "$ARTIFACT_DIR"/fx_compute_nodes_runtime_estimation_*.txt; do
        [ -f "$f" ] || continue
        basename=$(basename "$f")
        url=$(cat "$f" | pastry -t "$basename" -l python -q 2>/dev/null)
        echo "$basename: $url"
    done
)

tlp ()
{
    rm --preserve-root=all --one-file-system -rf ~/tmp/trace_logs/*;
    TORCH_TRACE=~/tmp/trace_logs "$@";
    tlparse ~/tmp/trace_logs/*rank_0* --overwrite-manifold;
    tlparse parse ~/tmp/trace_logs/*rank_0* -o "$TLPARSE_OUTPUT_DIR" --overwrite;
    upload_tlparse_passes "$TLPARSE_OUTPUT_DIR"
}

# Everything inside this brace group -- training, tlparse, trace upload -- is
# captured to both the terminal and "$LOG_FILE" by the single redirection on the
# closing brace below. Keep the redirection on the group, NOT glued to the end of
# the long backslash-continued training command: a stray trailing space after a
# "\" silently detaches "| tee" and the whole run goes unlogged (and splits a
# flag into a bogus "command not found"). Whichever block you uncomment here is
# logged automatically.
{

# --- DeepSeek-v3 16B MinimalAsyncEP (FSDP+EP, sync-free MoE) ---
# TORCHINDUCTOR_COMPILE_THREADS caps Inductor compile-worker parallelism:
# MinimalAsyncEP's kernels compile cold, and the default worker pool (this host
# has 368 cores) spikes host RAM enough to OOM-kill regional_inductor without it.
NGPU=8 MODULE=graph_trainer.deepseek_v3 CONFIG=graph_trainer_deepseek_v3_16b_minimal_async_ep TORCHINDUCTOR_COMPILE_THREADS=8 tlp ./run_train.sh \
    --compile.mode aot_fx_trace \
    --parallelism.data_parallel_shard_degree=8 \
    --parallelism.tensor_parallel_degree=1 \
    --parallelism.expert_parallel_degree=4 \
    --training.steps 20 \
    --dataloader.dataset c4_test \
    --compile.debug_graph_passes \
    --compile.disable_passes cudagraph_pass \
    --profiler.enable_profiling \
    --profiler.profile_freq 10 \
    --dump_folder "$PROFILE_DIR" \
    --debug.print-config \
    --compile.memory_policy full

echo "Run log saved to $LOG_FILE"

# Upload rank 0 trace to Perfetto
TRACE_FILE=$(find "$PROFILE_DIR" -name "rank0_*" -type f | head -1)
if [ -n "$TRACE_FILE" ]; then
    echo "Uploading $TRACE_FILE"
    python3 ~/local/fbsource/arvr/scripts/perfetto/share_trace.py "$TRACE_FILE"
else
    echo "No rank0 trace found in $PROFILE_DIR"
fi

} 2>&1 | tee "$LOG_FILE"

# Upload the (now fully-written) log to pastry. Reading happens after the pipe
# above has completed, so there is no flush race with tee.
PASTRY_LINK=$(pastry < "$LOG_FILE")
echo "Pastry link: $PASTRY_LINK"
