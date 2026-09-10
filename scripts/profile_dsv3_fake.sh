#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Select TRAINER=graph (default) or TRAINER=eager.
TRAINER="${TRAINER:-graph}"
NGPU="${NGPU:-256}"
TRAINING_STEPS="${TRAINING_STEPS:-10}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-24}"
SEQ_LEN="${SEQ_LEN:-4096}"
DATASET="${DATASET:-c4_test}"
DP_SHARD_DEGREE="${DP_SHARD_DEGREE:-256}"
EP_DEGREE="${EP_DEGREE:-64}"
PROFILE_FREQ="${PROFILE_FREQ:-10}"
PROFILER_WARMUP="${PROFILER_WARMUP:-3}"
PROFILER_ACTIVE="${PROFILER_ACTIVE:-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"

case "$TRAINER" in
    graph)
        MODULE="${MODULE:-graph_trainer.deepseek_v3}"
        CONFIG="${CONFIG:-graph_trainer_deepseek_v3_671b}"
        ENABLE_TLPARSE="${ENABLE_TLPARSE:-1}"
        ;;
    eager)
        MODULE="${MODULE:-deepseek_v3}"
        CONFIG="${CONFIG:-deepseek_v3_671b}"
        ENABLE_TLPARSE="${ENABLE_TLPARSE:-0}"
        ;;
    *)
        echo "TRAINER must be 'graph' or 'eager', got: $TRAINER" >&2
        exit 2
        ;;
esac

TRACE_SUBDIR="${TRACE_SUBDIR:-profiling/dsv3_fake/$TRAINER/$RUN_ID}"
PROFILE_DIR="$REPO_ROOT/outputs/$TRACE_SUBDIR"
MEMORY_SUBDIR="${MEMORY_SUBDIR:-$TRACE_SUBDIR/memory_snapshot}"
MEMORY_DIR="$REPO_ROOT/outputs/$MEMORY_SUBDIR"
TORCH_TRACE_DIR="${TORCH_TRACE_DIR:-$PROFILE_DIR/torch_trace}"
TLPARSE_OUTPUT_DIR="${TLPARSE_OUTPUT_DIR:-$PROFILE_DIR/tlparse}"
LOG_FILE="${LOG_FILE:-$PROFILE_DIR/profile.log}"
PROFILER_UPLOAD_LOG="$PROFILE_DIR/profiler_upload.log"
MEMORY_UPLOAD_LOG="$PROFILE_DIR/memory_upload.log"
TLPARSE_MANIFOLD_LOG="$PROFILE_DIR/tlparse_manifold_upload.log"
TLPARSE_UPLOAD_LOG="$PROFILE_DIR/tlparse_artifact_uploads.log"
PERFETTO_UPLOADER="${PERFETTO_UPLOADER:-$REPO_ROOT/scripts/share_trace.py}"
UPLOAD_TRACE="${UPLOAD_TRACE:-1}"
UPLOAD_MEMORY_SNAPSHOT="${UPLOAD_MEMORY_SNAPSHOT:-$UPLOAD_TRACE}"
UPLOAD_TLPARSE="${UPLOAD_TLPARSE:-1}"
UPLOAD_LOG="${UPLOAD_LOG:-1}"

if (( PROFILE_FREQ < PROFILER_WARMUP + PROFILER_ACTIVE )); then
    echo "PROFILE_FREQ must be at least PROFILER_WARMUP + PROFILER_ACTIVE" >&2
    exit 2
fi

if (( TRAINING_STEPS < PROFILE_FREQ )); then
    echo "TRAINING_STEPS must be at least PROFILE_FREQ to produce a trace" >&2
    exit 2
fi

if { [[ "$UPLOAD_TRACE" == "1" ]] || [[ "$UPLOAD_MEMORY_SNAPSHOT" == "1" ]]; } \
    && [[ ! -f "$PERFETTO_UPLOADER" ]]; then
    echo "Trace uploader not found: $PERFETTO_UPLOADER" >&2
    exit 2
fi

if [[ "$ENABLE_TLPARSE" == "1" ]] && ! command -v tlparse >/dev/null 2>&1; then
    echo "ENABLE_TLPARSE=1 requires tlparse on PATH" >&2
    exit 2
fi

mkdir -p "$PROFILE_DIR" "$MEMORY_DIR" "$(dirname "$LOG_FILE")"
if [[ "$ENABLE_TLPARSE" == "1" ]]; then
    mkdir -p "$TORCH_TRACE_DIR" "$TLPARSE_OUTPUT_DIR"
fi

upload_tlparse_passes()
(
    set -u
    local dir="${1:?Usage: upload_tlparse_passes <tlparse_output_dir>}"
    local artifact_dir="$dir/-_-_-_-"
    local upload_status=0

    if [[ ! -d "$artifact_dir" ]]; then
        echo "No tlparse artifact directory found at $artifact_dir" >&2
        exit 2
    fi

    echo "=== Tlparse pass artifacts ==="

    local traced traced_name traced_output
    local -a traced_files=("$artifact_dir"/make_fx_graph_traced_*.txt)
    traced="${traced_files[0]}"
    if [[ -f "$traced" ]]; then
        traced_name="${traced##*/}"
        if traced_output="$(pastry -t "${traced_name%.txt}" -l python -q \
            < "$traced" 2>/dev/null)"; then
            echo "${traced_name%.txt}: $traced_output"
        else
            echo "Failed to upload $traced" >&2
            upload_status=1
        fi
    fi

    local before_file basename pass_name after_file after_basename
    local before_output after_output before_paste after_paste diff_url
    local -a after_files
    for before_file in "$artifact_dir"/before_*_pass_*.txt; do
        [[ -f "$before_file" ]] || continue
        basename="${before_file##*/}"
        pass_name="${basename#before_}"
        pass_name="${pass_name%_pass_*.txt}"
        after_files=("$artifact_dir"/after_"${pass_name}"_pass_*.txt)
        after_file="${after_files[0]}"

        if [[ ! -f "$after_file" ]]; then
            echo "WARN: no after file for $pass_name, skipping" >&2
            continue
        fi

        if diff -q "$before_file" "$after_file" >/dev/null 2>&1; then
            echo "$pass_name: no changes, skipping"
            continue
        fi

        after_basename="${after_file##*/}"
        if ! before_output="$(pastry -t "$basename" -l python -q \
            < "$before_file" 2>/dev/null)"; then
            echo "Failed to upload $before_file" >&2
            upload_status=1
            continue
        fi
        if ! after_output="$(pastry -t "$after_basename" -l python -q \
            < "$after_file" 2>/dev/null)"; then
            echo "Failed to upload $after_file" >&2
            upload_status=1
            continue
        fi

        if [[ "$before_output" =~ P([0-9]+) ]]; then
            before_paste="${BASH_REMATCH[1]}"
        else
            echo "Could not extract a paste number from: $before_output" >&2
            upload_status=1
            continue
        fi
        if [[ "$after_output" =~ P([0-9]+) ]]; then
            after_paste="${BASH_REMATCH[1]}"
        else
            echo "Could not extract a paste number from: $after_output" >&2
            upload_status=1
            continue
        fi

        diff_url="https://www.internalfb.com/intern/diffing/?before_paste_number=${before_paste}&after_paste_number=${after_paste}&regex_remove_pattern=&enable_regex_remove=0&strip_empty_lines=0&line_wrap=0&selected_tab=plain_diff"
        echo "$pass_name: $diff_url"
    done

    echo "=== Standalone tlparse artifacts ==="
    local artifact artifact_name artifact_output
    for artifact in \
        "$artifact_dir"/activation_memory_policy_*.txt \
        "$artifact_dir"/fx_codegen_*.txt \
        "$artifact_dir"/fx_collectives_analytical_estimation_*.txt \
        "$artifact_dir"/fx_compute_nodes_runtime_estimation_*.txt; do
        [[ -f "$artifact" ]] || continue
        artifact_name="${artifact##*/}"
        if artifact_output="$(pastry -t "$artifact_name" -l python -q \
            < "$artifact" 2>/dev/null)"; then
            echo "$artifact_name: $artifact_output"
        else
            echo "Failed to upload $artifact" >&2
            upload_status=1
        fi
    done

    exit "$upload_status"
)

train_args=(
    --module "$MODULE"
    --config "$CONFIG"
    --comm.mode=fake_backend
    --parallelism.data_parallel_shard_degree "$DP_SHARD_DEGREE"
    --parallelism.expert_parallel_degree "$EP_DEGREE"
    --compile.enable
    --training.local_batch_size "$LOCAL_BATCH_SIZE"
    --training.seq_len "$SEQ_LEN"
    --training.steps "$TRAINING_STEPS"
    --dataloader.dataset "$DATASET"
    --debug.seed 42
    --debug.deterministic
    --debug.moe_force_load_balance
    --profiler.enable_profiling
    --profiler.enable_memory_snapshot
    --profiler.save_traces_folder "$TRACE_SUBDIR"
    --profiler.save_memory_snapshot_folder "$MEMORY_SUBDIR"
    --profiler.profile_freq "$PROFILE_FREQ"
    --profiler.profiler_warmup "$PROFILER_WARMUP"
    --profiler.profiler_active "$PROFILER_ACTIVE"
    --profiler.profiler_repeat 1
)

if [[ "$TRAINER" == "graph" ]]; then
    train_args+=(
        --compile.mode aot_fx_trace
        --compile.memory_policy full
        --compile.debug_graph_passes
    )
fi
train_args+=("$@")
if [[ "$TRAINER" == "eager" ]]; then
    train_args+=(activation-checkpoint:full)
fi

run_env=(
    NGPU="$NGPU"
    LOCAL_RANK=0
    LOG_RANK=0
    PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
)
if [[ "$ENABLE_TLPARSE" == "1" ]]; then
    run_env+=(TORCH_TRACE="$TORCH_TRACE_DIR")
fi

set +e
{
    echo "Trainer: $TRAINER"
    echo "Model: $MODULE / $CONFIG"
    echo "Parallelism: FSDP=$DP_SHARD_DEGREE EP=$EP_DEGREE virtual GPUs=$NGPU"
    echo "Batch/sequence: local batch=$LOCAL_BATCH_SIZE sequence=$SEQ_LEN"
    echo "Dataset: $DATASET"
    echo "Profile directory: $PROFILE_DIR"
    echo "Training log: $LOG_FILE"
    printf "Command:"
    printf " %q" env "${run_env[@]}" python -m torchtitan.train "${train_args[@]}"
    printf "\n"

    run_status=0
    env "${run_env[@]}" python -m torchtitan.train "${train_args[@]}" \
        || run_status=$?
    training_status=$run_status

    if [[ "$ENABLE_TLPARSE" == "1" ]]; then
        mapfile -t torch_trace_files < <(
            rg --files "$TORCH_TRACE_DIR" -g '*rank_0*' | sort
        )
        if (( ${#torch_trace_files[@]} == 0 )); then
            echo "No rank-0 Torch trace found in $TORCH_TRACE_DIR" >&2
            (( training_status != 0 )) || run_status=2
        else
            torch_trace_file="${torch_trace_files[0]}"
            echo "Parsing Torch trace: $torch_trace_file"

            if [[ "$UPLOAD_TLPARSE" == "1" ]]; then
                tlparse "$torch_trace_file" --overwrite-manifold \
                    2>&1 | tee "$TLPARSE_MANIFOLD_LOG"
                tlparse_status=${PIPESTATUS[0]}
                (( tlparse_status == 0 )) || run_status=$tlparse_status
            fi

            tlparse parse "$torch_trace_file" -o "$TLPARSE_OUTPUT_DIR" --overwrite
            tlparse_status=$?
            (( tlparse_status == 0 )) || run_status=$tlparse_status

            if (( tlparse_status == 0 )) && [[ "$UPLOAD_TLPARSE" == "1" ]]; then
                if command -v pastry >/dev/null 2>&1; then
                    upload_tlparse_passes "$TLPARSE_OUTPUT_DIR" \
                        2>&1 | tee "$TLPARSE_UPLOAD_LOG"
                    upload_status=${PIPESTATUS[0]}
                    (( upload_status == 0 )) || run_status=$upload_status
                else
                    echo "pastry not found; skipping tlparse artifact uploads" >&2
                    run_status=2
                fi
            fi
        fi
    fi

    if (( training_status == 0 )) && [[ "$UPLOAD_TRACE" == "1" ]]; then
        mapfile -t trace_files < <(rg --files "$PROFILE_DIR" -g 'rank0_*' | sort)
        if (( ${#trace_files[@]} > 0 )); then
            echo "Uploading ${trace_files[0]}"
            python3 "$PERFETTO_UPLOADER" "${trace_files[0]}" \
                >"$PROFILER_UPLOAD_LOG" 2>&1
            upload_status=$?
            cat "$PROFILER_UPLOAD_LOG"
            (( upload_status == 0 )) || run_status=$upload_status
        else
            echo "No rank-0 profiler trace found in $PROFILE_DIR" >&2
            run_status=2
        fi
    fi

    if (( training_status == 0 )) && [[ "$UPLOAD_MEMORY_SNAPSHOT" == "1" ]]; then
        mapfile -t memory_files < <(rg --files "$MEMORY_DIR" -g '*.pickle' | sort)
        if (( ${#memory_files[@]} > 0 )); then
            echo "Uploading ${memory_files[0]}"
            python3 "$PERFETTO_UPLOADER" --is-memory-snapshot \
                "${memory_files[0]}" >"$MEMORY_UPLOAD_LOG" 2>&1
            upload_status=$?
            cat "$MEMORY_UPLOAD_LOG"
            (( upload_status == 0 )) || run_status=$upload_status
        else
            echo "No memory snapshot found in $MEMORY_DIR" >&2
            run_status=2
        fi
    fi

    exit "$run_status"
} 2>&1 | tee "$LOG_FILE"
run_status=${PIPESTATUS[0]}
set -e

if [[ "$UPLOAD_LOG" == "1" ]]; then
    if command -v pastry >/dev/null 2>&1; then
        PASTRY_LINK="$(pastry < "$LOG_FILE")"
        echo "Pastry link: $PASTRY_LINK"
    else
        echo "pastry not found; log remains at $LOG_FILE" >&2
    fi
fi

profiler_link=""
memory_link=""
tlparse_link=""
if [[ -f "$PROFILER_UPLOAD_LOG" ]]; then
    profiler_link="$(awk '/^Perfetto UI:$/ { getline; print; exit }' \
        "$PROFILER_UPLOAD_LOG")"
fi
if [[ -f "$MEMORY_UPLOAD_LOG" ]]; then
    memory_link="$(awk '/^Memory snapshot:$/ { getline; print; exit }' \
        "$MEMORY_UPLOAD_LOG")"
fi
if [[ -f "$TLPARSE_MANIFOLD_LOG" ]]; then
    tlparse_link="$(awk '
        {
            for (i = 1; i <= NF; i++) {
                if ($i ~ /^https:\/\//) {
                    gsub(/[),]$/, "", $i)
                    print $i
                    exit
                }
            }
        }
    ' "$TLPARSE_MANIFOLD_LOG")"
fi

if [[ -n "$profiler_link" || -n "$memory_link" || -n "$tlparse_link" ]]; then
    echo "Artifact links:"
    [[ -z "$profiler_link" ]] || echo "Profiler trace: $profiler_link"
    [[ -z "$memory_link" ]] || echo "Memory snapshot: $memory_link"
    [[ -z "$tlparse_link" ]] || echo "Tlparse report: $tlparse_link"
fi
if [[ -f "$TLPARSE_UPLOAD_LOG" ]]; then
    echo "Tlparse artifact links: $TLPARSE_UPLOAD_LOG"
fi

exit "$run_status"
