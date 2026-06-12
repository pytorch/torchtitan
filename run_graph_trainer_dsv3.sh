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

# Run selectors:
#   MODE = graph (graph_trainer, the fix) | eager (eager Trainer baseline)
#   EP   = regular (all-to-all + _grouped_mm) | minimal (MinimalAsyncEP, sync-free)
# Same model / parallelism / batch / recompute across all four, so graph-vs-eager
# and regular-vs-minimal are directly comparable:
#   Run 7  MODE=graph EP=regular     Run 8  MODE=eager EP=regular
#   Run 9  MODE=graph EP=minimal     Run 10 MODE=eager EP=minimal
# graph+minimal enables cudagraph (MinimalAsyncEP is sync-free / cudagraphable);
# graph+regular disables it (regular EP's _grouped_mm / all-to-all aren't
# cudagraphable on H100). Eager never uses cudagraph.
#   Run 11 MODE=graph EP=minimal INDUCTOR=full  -- full inductor (whole-graph
#          codegen). OOMs at B=16 (memory ~+15 GiB vs regional); use BATCH=8.
# Extra toggles (see their definitions below): INDUCTOR (regional|full),
# CUDAGRAPH (auto|on|off), BATCH (per-rank local batch size).
# TORCHINDUCTOR_COMPILE_THREADS caps Inductor compile workers so the cold MoE
# kernel compile doesn't spike host RAM and OOM-kill regional_inductor.
MODE="${MODE:-graph}"
EP="${EP:-regular}"
if [ "$EP" = "minimal" ]; then SUFFIX="_minimal_async_ep"; else SUFFIX=""; fi

# STEPS    = number of training steps (default 20).
# PROFILE  = 1 (default) runs under `tlp` with the profiler + memory snapshot and
#            uploads all artifacts; 0 is a LEAN throughput run -- no profiler, no
#            memory snapshot, no tlparse, no uploads. Use PROFILE=0 with a larger
#            STEPS for clean steady-state MFU: the profiler perturbs the cudagraph
#            path asymmetrically, and step-1 graph capture pollutes the step-10
#            rolling average, so MFU is only trustworthy from a profiler-free run
#            read past the capture (step 20+).
STEPS="${STEPS:-20}"
PROFILE="${PROFILE:-1}"

# INDUCTOR = regional (default) | full. graph (MODE=graph) only.
#   regional : compile just the FlexAttention regions with inductor; the rest of
#              the traced graph runs interpreted (the historical default for
#              Runs 1-10).
#   full     : compile the ENTIRE traced graph through inductor into Triton
#              kernels (--compile.inductor_compilation full). Faster, but may
#              shift bits vs regional (inductor fuses/reorders + re-enables
#              reorder_for_peak_memory). full inductor still returns a
#              GraphModule, so cudagraph can wrap it -> composes with EP=minimal's
#              forced cudagraph.
INDUCTOR="${INDUCTOR:-regional}"
if [ "$INDUCTOR" = "full" ]; then INDUCTOR_FLAG="--compile.inductor_compilation full"; else INDUCTOR_FLAG=""; fi

# BATCH = per-rank local batch size (default 16). Lower it (e.g. 8) when a config
# does not fit in 95 GiB -- full inductor under forced cudagraph pins the whole
# fused working set into the cudagraph static pool, which OOMs at B=16.
BATCH="${BATCH:-16}"

if [ "$PROFILE" = "1" ]; then
    RUNNER=tlp
    PROFILE_FLAGS="--profiler.enable_profiling --profiler.profile_freq 10 --profiler.enable_memory_snapshot --dump_folder $PROFILE_DIR"
else
    RUNNER=""
    PROFILE_FLAGS=""
fi

if [ "$MODE" = "eager" ]; then
# --- DeepSeek-v3 16B EAGER baseline (FSDP2 + full activation checkpointing) ---
# Eager Trainer reference. No aot_fx_trace graph -> no graph-pass tlparse diffs
# (tlparse covers only the loss torch.compile); the profiler trace and CUDA
# memory snapshot are the meaningful artifacts.
NGPU=8 MODULE=deepseek_v3 CONFIG=deepseek_v3_16b${SUFFIX} TORCHINDUCTOR_COMPILE_THREADS=8 $RUNNER ./run_train.sh \
    --parallelism.data_parallel_shard_degree=8 \
    --parallelism.tensor_parallel_degree=1 \
    --parallelism.expert_parallel_degree=4 \
    --training.steps $STEPS \
    --training.local_batch_size $BATCH \
    --dataloader.dataset c4_test \
    --activation_checkpoint.mode full \
    $PROFILE_FLAGS \
    --debug.print-config
else
# --- DeepSeek-v3 16B graph_trainer (lm_head chunked-loss fix, PR #3636) ---
# CUDAGRAPH = auto (default: ON for EP=minimal which is cudagraphable, OFF for
#             EP=regular whose _grouped_mm/all-to-all aren't) | on | off. The
#             explicit on/off override lets us disable cudagraph on the minimal
#             path (e.g. to fit full inductor at B=16, which OOMs with the
#             cudagraph static pool).
CUDAGRAPH="${CUDAGRAPH:-auto}"
case "$CUDAGRAPH" in
    on)   CUDAGRAPH_FLAG="" ;;
    off)  CUDAGRAPH_FLAG="--compile.disable_passes cudagraph_pass" ;;
    auto) if [ "$EP" = "minimal" ]; then CUDAGRAPH_FLAG=""; else CUDAGRAPH_FLAG="--compile.disable_passes cudagraph_pass"; fi ;;
esac
NGPU=8 MODULE=graph_trainer.deepseek_v3 CONFIG=graph_trainer_deepseek_v3_16b${SUFFIX} TORCHINDUCTOR_COMPILE_THREADS=8 $RUNNER ./run_train.sh \
    --compile.mode aot_fx_trace \
    --parallelism.data_parallel_shard_degree=8 \
    --parallelism.tensor_parallel_degree=1 \
    --parallelism.expert_parallel_degree=4 \
    --training.steps $STEPS \
    --training.local_batch_size $BATCH \
    --dataloader.dataset c4_test \
    --compile.debug_graph_passes \
    $CUDAGRAPH_FLAG \
    $INDUCTOR_FLAG \
    $PROFILE_FLAGS \
    --debug.print-config \
    --compile.memory_policy full
fi

echo "Run log saved to $LOG_FILE"

# Artifact uploads only when profiling (PROFILE=1). A lean throughput run (PROFILE=0)
# produces no trace/snapshot, so there is nothing to upload.
if [ "$PROFILE" = "1" ]; then
# Upload rank 0 trace to Perfetto
TRACE_FILE=$(find "$PROFILE_DIR" -name "rank0_*" -type f | head -1)
if [ -n "$TRACE_FILE" ]; then
    echo "Uploading $TRACE_FILE"
    python3 ~/local/fbsource/arvr/scripts/perfetto/share_trace.py "$TRACE_FILE"
else
    echo "No rank0 trace found in $PROFILE_DIR"
fi

# Upload rank 0 CUDA memory snapshot to the internal PyTorch Memory Visualizer.
# torchtitan names snapshot files "<rank:06d>_step_<N>.pickle"; rank 0 -> 000000_*.
# sort|tail -1 picks the latest step when multiple snapshots are dumped.
SNAPSHOT_FILE=$(find "$PROFILE_DIR" -path "*memory_snapshot*" -name "000000_*.pickle" -type f | sort | tail -1)
if [ -n "$SNAPSHOT_FILE" ]; then
    echo "Uploading $SNAPSHOT_FILE"
    python3 ~/local/fbsource/arvr/scripts/perfetto/share_trace.py --is-memory-snapshot "$SNAPSHOT_FILE"
else
    echo "No rank0 memory snapshot found in $PROFILE_DIR"
fi
fi

} 2>&1 | tee "$LOG_FILE"

# Upload the (now fully-written) log to pastry. Reading happens after the pipe
# above has completed, so there is no flush race with tee.
PASTRY_LINK=$(pastry < "$LOG_FILE")
echo "Pastry link: $PASTRY_LINK"
