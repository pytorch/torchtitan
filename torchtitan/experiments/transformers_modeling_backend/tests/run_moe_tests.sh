#!/bin/bash
# Integration tests for MoE parallelism in the transformers modeling backend.
#
# Usage:
#   cd torchtitan
#   bash torchtitan/experiments/transformers_modeling_backend/tests/run_moe_tests.sh [NGPU] [STEPS]
#
# Default NGPU=8, STEPS=200. All configurations use dp_shard=-1 (auto-computed).
# EP borrows from the fsdp*tp product; it does NOT multiply into WORLD_SIZE.
#
# When NGPU >= 8, tests that fit on 4 GPUs are run in parallel pairs (2 tests
# at a time on GPUs 0-3 and 4-7) to halve wall-clock time. Tests requiring
# 8 GPUs run sequentially on all GPUs.

set -euo pipefail

TOTAL_GPUS=${1:-8}
MODULE=transformers_modeling_backend
CONFIG=transformers_modeling_backend_debugmodel_moe
STEPS=${2:-200}

RESULTS_DIR=${RESULTS_DIR:-$(mktemp -d)}
mkdir -p "$RESULTS_DIR"

run_test() {
    local name="$1"
    local gpus="$2"
    local ngpu="$3"
    shift 3

    echo "  START: $name (GPUs: $gpus, NGPU=$ngpu)"

    local result_file="$RESULTS_DIR/$(echo "$name" | tr ' /+()' '_____')"

    if CUDA_VISIBLE_DEVICES=$gpus NGPU=$ngpu MODULE=$MODULE CONFIG=$CONFIG \
        ./run_train.sh --training.steps $STEPS "$@" > "$result_file.log" 2>&1; then
        echo "PASSED" > "$result_file.status"
        echo "  PASSED: $name"
    else
        echo "FAILED" > "$result_file.status"
        echo "$name" > "$result_file.fail"
        echo "  FAILED: $name"
    fi
}

# Run two 4-GPU tests in parallel on GPUs 0-3 and 4-7
run_pair() {
    local name_a="$1"; shift
    local args_a=()
    while [ "$1" != "--" ]; do args_a+=("$1"); shift; done
    shift  # skip --
    local name_b="$1"; shift
    local args_b=("$@")

    echo ""
    echo "================================================================"
    echo "PARALLEL: $name_a  |  $name_b"
    echo "================================================================"

    run_test "$name_a" "0,1,2,3" 4 "${args_a[@]}" &
    local pid_a=$!
    run_test "$name_b" "4,5,6,7" 4 "${args_b[@]}" &
    local pid_b=$!
    wait $pid_a $pid_b
}

# Run a single test on all GPUs
run_single() {
    local name="$1"; shift
    echo ""
    echo "================================================================"
    echo "SINGLE: $name (all $TOTAL_GPUS GPUs)"
    echo "================================================================"
    run_test "$name" "$(seq -s, 0 $((TOTAL_GPUS - 1)))" "$TOTAL_GPUS" "$@"
}

# Run a single test on 4 GPUs (when we can't pair it)
run_half() {
    local name="$1"; shift
    echo ""
    echo "================================================================"
    echo "SINGLE: $name (4 GPUs)"
    echo "================================================================"
    run_test "$name" "0,1,2,3" 4 "$@"
}

if [ "$TOTAL_GPUS" -ge 8 ]; then
    # ── Parallel pairs (4 GPUs each) ──

    run_pair \
        "FSDP-only (baseline)" \
            --parallelism.data_parallel_shard_degree -1 \
        -- \
        "FSDP + EP=2" \
            --parallelism.data_parallel_shard_degree -1 \
            --parallelism.expert_parallel_degree 2

    run_pair \
        "FSDP + EP=4" \
            --parallelism.data_parallel_shard_degree -1 \
            --parallelism.expert_parallel_degree 4 \
        -- \
        "FSDP + TP=2 (MoE, no EP)" \
            --parallelism.data_parallel_shard_degree -1 \
            --parallelism.tensor_parallel_degree 2

    run_pair \
        "FSDP + TP=4 (MoE, no EP)" \
            --parallelism.data_parallel_shard_degree -1 \
            --parallelism.tensor_parallel_degree 4 \
        -- \
        "FSDP + TP=2 + EP=2" \
            --parallelism.data_parallel_shard_degree -1 \
            --parallelism.tensor_parallel_degree 2 \
            --parallelism.expert_parallel_degree 2

    run_pair \
        "FSDP + TP=2 + EP=4" \
            --parallelism.data_parallel_shard_degree -1 \
            --parallelism.tensor_parallel_degree 2 \
            --parallelism.expert_parallel_degree 4 \
        -- \
        "FSDP + EP=4 + compile" \
            --parallelism.data_parallel_shard_degree -1 \
            --parallelism.expert_parallel_degree 4 \
            --compile.enable

    run_pair \
        "FSDP + TP=2 + EP=2 + compile" \
            --parallelism.data_parallel_shard_degree -1 \
            --parallelism.tensor_parallel_degree 2 \
            --parallelism.expert_parallel_degree 2 \
            --compile.enable \
        -- \
        "FSDP + TP=2 (MoE, no EP) + compile" \
            --parallelism.data_parallel_shard_degree -1 \
            --parallelism.tensor_parallel_degree 2 \
            --compile.enable

    run_pair \
        "FSDP + PP=2 + EP=2" \
            --parallelism.data_parallel_shard_degree -1 \
            --parallelism.pipeline_parallel_degree 2 \
            --parallelism.pipeline_parallel_schedule 1F1B \
            --parallelism.expert_parallel_degree 2 \
        -- \
        "HSDP + EP=2" \
            --parallelism.data_parallel_replicate_degree 2 \
            --parallelism.data_parallel_shard_degree -1 \
            --parallelism.expert_parallel_degree 2

    run_pair \
        "FSDP + EP=2 (no SAC)" \
            --parallelism.data_parallel_shard_degree -1 \
            --parallelism.expert_parallel_degree 2 \
            --activation_checkpoint.mode none \
        -- \
        "FSDP + TP=2 + EP=2 (no SAC)" \
            --parallelism.data_parallel_shard_degree -1 \
            --parallelism.tensor_parallel_degree 2 \
            --parallelism.expert_parallel_degree 2 \
            --activation_checkpoint.mode none

    run_pair \
        "FSDP + CP=2" \
            --parallelism.data_parallel_shard_degree -1 \
            --parallelism.context_parallel_degree 2 \
        -- \
        "FSDP + CP=2 + EP=2" \
            --parallelism.data_parallel_shard_degree -1 \
            --parallelism.context_parallel_degree 2 \
            --parallelism.expert_parallel_degree 2

    run_half \
        "FSDP + PP=2 + EP=2 + compile" \
            --parallelism.data_parallel_shard_degree -1 \
            --parallelism.pipeline_parallel_degree 2 \
            --parallelism.pipeline_parallel_schedule 1F1B \
            --parallelism.expert_parallel_degree 2 \
            --compile.enable

    run_half \
        "FSDP + CP=2 + TP=2" \
            --parallelism.data_parallel_shard_degree -1 \
            --parallelism.context_parallel_degree 2 \
            --parallelism.tensor_parallel_degree 2

    # ── 8-GPU-only tests (sequential) ──

    run_single \
        "FSDP + TP=2 + PP=2 + EP=2" \
            --parallelism.data_parallel_shard_degree -1 \
            --parallelism.tensor_parallel_degree 2 \
            --parallelism.pipeline_parallel_degree 2 \
            --parallelism.pipeline_parallel_schedule 1F1B \
            --parallelism.expert_parallel_degree 2

    run_single \
        "HSDP + TP=2 + EP=2" \
            --parallelism.data_parallel_replicate_degree 2 \
            --parallelism.data_parallel_shard_degree -1 \
            --parallelism.tensor_parallel_degree 2 \
            --parallelism.expert_parallel_degree 2

    run_single \
        "FSDP + CP=2 + TP=2 + EP=2" \
            --parallelism.data_parallel_shard_degree -1 \
            --parallelism.context_parallel_degree 2 \
            --parallelism.tensor_parallel_degree 2 \
            --parallelism.expert_parallel_degree 2

else
    # ── Sequential mode (< 8 GPUs) ──
    GPU_LIST=$(seq -s, 0 $((TOTAL_GPUS - 1)))

    for test_args in \
        "FSDP-only (baseline)|--parallelism.data_parallel_shard_degree -1" \
        "FSDP + EP=2|--parallelism.data_parallel_shard_degree -1 --parallelism.expert_parallel_degree 2" \
        "FSDP + EP=4|--parallelism.data_parallel_shard_degree -1 --parallelism.expert_parallel_degree 4" \
        "FSDP + TP=2 (MoE, no EP)|--parallelism.data_parallel_shard_degree -1 --parallelism.tensor_parallel_degree 2" \
        "FSDP + TP=4 (MoE, no EP)|--parallelism.data_parallel_shard_degree -1 --parallelism.tensor_parallel_degree 4" \
        "FSDP + TP=2 + EP=2|--parallelism.data_parallel_shard_degree -1 --parallelism.tensor_parallel_degree 2 --parallelism.expert_parallel_degree 2" \
        "FSDP + TP=2 + EP=4|--parallelism.data_parallel_shard_degree -1 --parallelism.tensor_parallel_degree 2 --parallelism.expert_parallel_degree 4" \
        "FSDP + EP=4 + compile|--parallelism.data_parallel_shard_degree -1 --parallelism.expert_parallel_degree 4 --compile.enable" \
        "FSDP + TP=2 + EP=2 + compile|--parallelism.data_parallel_shard_degree -1 --parallelism.tensor_parallel_degree 2 --parallelism.expert_parallel_degree 2 --compile.enable" \
        "FSDP + TP=2 (MoE, no EP) + compile|--parallelism.data_parallel_shard_degree -1 --parallelism.tensor_parallel_degree 2 --compile.enable" \
        "FSDP + PP=2 + EP=2|--parallelism.data_parallel_shard_degree -1 --parallelism.pipeline_parallel_degree 2 --parallelism.pipeline_parallel_schedule 1F1B --parallelism.expert_parallel_degree 2" \
        "HSDP + EP=2|--parallelism.data_parallel_replicate_degree 2 --parallelism.data_parallel_shard_degree -1 --parallelism.expert_parallel_degree 2" \
        "FSDP + EP=2 (no SAC)|--parallelism.data_parallel_shard_degree -1 --parallelism.expert_parallel_degree 2 --activation_checkpoint.mode none" \
        "FSDP + TP=2 + EP=2 (no SAC)|--parallelism.data_parallel_shard_degree -1 --parallelism.tensor_parallel_degree 2 --parallelism.expert_parallel_degree 2 --activation_checkpoint.mode none" \
        "FSDP + PP=2 + EP=2 + compile|--parallelism.data_parallel_shard_degree -1 --parallelism.pipeline_parallel_degree 2 --parallelism.pipeline_parallel_schedule 1F1B --parallelism.expert_parallel_degree 2 --compile.enable" \
        "FSDP + CP=2|--parallelism.data_parallel_shard_degree -1 --parallelism.context_parallel_degree 2" \
        "FSDP + CP=2 + EP=2|--parallelism.data_parallel_shard_degree -1 --parallelism.context_parallel_degree 2 --parallelism.expert_parallel_degree 2" \
        "FSDP + CP=2 + TP=2|--parallelism.data_parallel_shard_degree -1 --parallelism.context_parallel_degree 2 --parallelism.tensor_parallel_degree 2" \
    ; do
        name="${test_args%%|*}"
        args="${test_args#*|}"
        echo ""
        echo "================================================================"
        echo "TEST: $name"
        echo "================================================================"
        # shellcheck disable=SC2086
        run_test "$name" "$GPU_LIST" "$TOTAL_GPUS" $args
    done
fi

# ── Full model × parallelism matrix ──
# Runs all 17 parallelism configs for each supported MoE model.
# Requires network access to download HF model configs.
# Set SKIP_MODEL_SWEEP=1 to skip in offline environments.
if [ "${SKIP_MODEL_SWEEP:-0}" != "1" ]; then
    SWEEP_STEPS=${3:-2}
    SWEEP_MODELS=(
        "Qwen/Qwen3-30B-A3B"
        "mistralai/Mixtral-8x7B-Instruct-v0.1"
        "Qwen/Qwen2-57B-A14B"
        "zai-org/GLM-4.7"
        "deepseek-ai/DeepSeek-V3"
        "zai-org/GLM-5"
        "microsoft/Phi-3.5-MoE-instruct"
    )

    # 4-GPU configs (run as parallel pairs or single halves)
    CONFIGS_4GPU=(
        "FSDP|--parallelism.data_parallel_shard_degree -1"
        "EP=2|--parallelism.data_parallel_shard_degree -1 --parallelism.expert_parallel_degree 2"
        "EP=4|--parallelism.data_parallel_shard_degree -1 --parallelism.expert_parallel_degree 4"
        "TP=2|--parallelism.data_parallel_shard_degree -1 --parallelism.tensor_parallel_degree 2"
        "TP=4|--parallelism.data_parallel_shard_degree -1 --parallelism.tensor_parallel_degree 4"
        "TP=2+EP=2|--parallelism.data_parallel_shard_degree -1 --parallelism.tensor_parallel_degree 2 --parallelism.expert_parallel_degree 2"
        "TP=2+EP=4|--parallelism.data_parallel_shard_degree -1 --parallelism.tensor_parallel_degree 2 --parallelism.expert_parallel_degree 4"
        "EP=4+compile|--parallelism.data_parallel_shard_degree -1 --parallelism.expert_parallel_degree 4 --compile.enable"
        "TP=2+EP=2+compile|--parallelism.data_parallel_shard_degree -1 --parallelism.tensor_parallel_degree 2 --parallelism.expert_parallel_degree 2 --compile.enable"
        "TP=2+compile|--parallelism.data_parallel_shard_degree -1 --parallelism.tensor_parallel_degree 2 --compile.enable"
        "PP=2+EP=2|--parallelism.data_parallel_shard_degree -1 --parallelism.pipeline_parallel_degree 2 --parallelism.pipeline_parallel_schedule 1F1B --parallelism.expert_parallel_degree 2"
        "HSDP+EP=2|--parallelism.data_parallel_replicate_degree 2 --parallelism.data_parallel_shard_degree -1 --parallelism.expert_parallel_degree 2"
        "EP=2(noSAC)|--parallelism.data_parallel_shard_degree -1 --parallelism.expert_parallel_degree 2 --activation_checkpoint.mode none"
        "TP=2+EP=2(noSAC)|--parallelism.data_parallel_shard_degree -1 --parallelism.tensor_parallel_degree 2 --parallelism.expert_parallel_degree 2 --activation_checkpoint.mode none"
        "PP=2+EP=2+compile|--parallelism.data_parallel_shard_degree -1 --parallelism.pipeline_parallel_degree 2 --parallelism.pipeline_parallel_schedule 1F1B --parallelism.expert_parallel_degree 2 --compile.enable"
        "CP=2|--parallelism.data_parallel_shard_degree -1 --parallelism.context_parallel_degree 2"
        "CP=2+EP=2|--parallelism.data_parallel_shard_degree -1 --parallelism.context_parallel_degree 2 --parallelism.expert_parallel_degree 2"
        "CP=2+TP=2|--parallelism.data_parallel_shard_degree -1 --parallelism.context_parallel_degree 2 --parallelism.tensor_parallel_degree 2"
    )

    # 8-GPU configs (run sequentially on all GPUs)
    CONFIGS_8GPU=(
        "TP=2+PP=2+EP=2|--parallelism.data_parallel_shard_degree -1 --parallelism.tensor_parallel_degree 2 --parallelism.pipeline_parallel_degree 2 --parallelism.pipeline_parallel_schedule 1F1B --parallelism.expert_parallel_degree 2"
        "HSDP+TP=2+EP=2|--parallelism.data_parallel_replicate_degree 2 --parallelism.data_parallel_shard_degree -1 --parallelism.tensor_parallel_degree 2 --parallelism.expert_parallel_degree 2"
        "CP=2+TP=2+EP=2|--parallelism.data_parallel_shard_degree -1 --parallelism.context_parallel_degree 2 --parallelism.tensor_parallel_degree 2 --parallelism.expert_parallel_degree 2"
    )

    for hf_model in "${SWEEP_MODELS[@]}"; do
        model_short="${hf_model##*/}"
        echo ""
        echo "################################################################"
        echo "MODEL: $hf_model"
        echo "################################################################"

        # Run 4-GPU configs in parallel pairs
        num_4gpu=${#CONFIGS_4GPU[@]}
        for ((i=0; i<num_4gpu; i+=2)); do
            cfg_a="${CONFIGS_4GPU[$i]}"
            name_a="${cfg_a%%|*}"
            args_a="${cfg_a#*|}"

            if [ $((i+1)) -lt "$num_4gpu" ]; then
                cfg_b="${CONFIGS_4GPU[$((i+1))]}"
                name_b="${cfg_b%%|*}"
                args_b="${cfg_b#*|}"

                echo ""
                echo "================================================================"
                echo "PARALLEL: ${name_a} ${model_short}  |  ${name_b} ${model_short}"
                echo "================================================================"

                # shellcheck disable=SC2086
                run_test "${name_a} ${model_short}" "0,1,2,3" 4 \
                    --training.steps "$SWEEP_STEPS" --hf_model "$hf_model" $args_a &
                pid_a=$!
                # shellcheck disable=SC2086
                run_test "${name_b} ${model_short}" "4,5,6,7" 4 \
                    --training.steps "$SWEEP_STEPS" --hf_model "$hf_model" $args_b &
                pid_b=$!
                wait $pid_a $pid_b
            else
                # Odd config out — run solo
                # shellcheck disable=SC2086
                run_half "${name_a} ${model_short}" \
                    --training.steps "$SWEEP_STEPS" --hf_model "$hf_model" $args_a
            fi
        done

        # Run 8-GPU configs sequentially
        for cfg in "${CONFIGS_8GPU[@]}"; do
            name="${cfg%%|*}"
            args="${cfg#*|}"
            # shellcheck disable=SC2086
            run_single "${name} ${model_short}" \
                --training.steps "$SWEEP_STEPS" --hf_model "$hf_model" $args
        done
    done
fi

# ── Summary ──
PASSED=$(find "$RESULTS_DIR" -name "*.status" -exec grep -l PASSED {} \; | wc -l)
FAILED=$(find "$RESULTS_DIR" -name "*.status" -exec grep -l FAILED {} \; | wc -l)
TOTAL=$((PASSED + FAILED))

echo ""
echo "================================================================"
echo "RESULTS: $PASSED/$TOTAL passed, $FAILED failed"
if [ "$FAILED" -gt 0 ]; then
    echo "Failed tests:"
    for f in "$RESULTS_DIR"/*.fail; do
        [ -f "$f" ] && echo "  - $(cat "$f")"
    done
    exit 1
else
    echo "All tests passed."
fi
