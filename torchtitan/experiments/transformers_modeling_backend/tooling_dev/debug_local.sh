#!/usr/bin/bash

# Check if flavor is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <flavor> [--compile]"
    echo "Example: $0 debugperf_large"
    echo "Example: $0 debugperf_large --compile"
    exit 1
fi

FLAVOR=$1
COMPILE_FLAG=""
FULL_AC_FLAG=""

# Check for flags
for arg in "$@"; do
    if [ "$arg" = "--compile" ]; then
        COMPILE_FLAG="--enable_compile"
    fi
    if [ "$arg" = "--full_ac" ]; then
        FULL_AC_FLAG="--enable_full_ac"
    fi
done

# Shared model configuration for fair comparison
VOCAB_SIZE=2048
N_LAYERS=6
N_HEADS=16
N_KV_HEADS=16
DIM=256
ROPE_THETA=500000

tt_model_names=(
    "llama3"
)

model_names=(
    "meta-llama/Llama-3.2-1B"  # ✅
)

# TorchTitan models - pass same model args
for model_name in "${tt_model_names[@]}"; do
    rm -rf debug_local_results/${model_name}

    python ./tooling_dev/test_hf_integration.py create_configs --model_name "$model_name" --out_dir debug_local_results --flavor "$FLAVOR" --model_type torchtitan --enable_profiling --profile_freq 5 $COMPILE_FLAG $FULL_AC_FLAG
    python ./tooling_dev/test_hf_integration.py submit_jobs --inp_dir debug_local_results/${model_name}/${FLAVOR}/seed_checkpoint --qos high
    while [ ! -f debug_local_results/${model_name}/${FLAVOR}/seed_checkpoint/status.txt ] || [ "$(cat debug_local_results/${model_name}/${FLAVOR}/seed_checkpoint/status.txt)" != "completed" ]; do
        echo "Waiting for seed checkpoint from ${model_name} to complete ..."
        sleep 1
    done
    python ./tooling_dev/test_hf_integration.py submit_jobs --inp_dir debug_local_results/${model_name}/${FLAVOR} --qos high
    echo "================"
done

for model_name in "${model_names[@]}"; do
    rm -rf debug_local_results/${model_name}

    python ./tooling_dev/test_hf_integration.py create_configs --model_name "$model_name" --out_dir debug_local_results --flavor "$FLAVOR" --model_type transformers_modeling_backend --hf_assets_path "/fsx/ferdinandmom/ferdinand-hf/huggingface/torchtitan/tests/assets/tokenizer" --enable_profiling --profile_freq 5 $COMPILE_FLAG $FULL_AC_FLAG
    python ./tooling_dev/test_hf_integration.py submit_jobs --inp_dir debug_local_results/${model_name}/${FLAVOR}/seed_checkpoint --qos high
    while [ ! -f debug_local_results/${model_name}/${FLAVOR}/seed_checkpoint/status.txt ] || [ "$(cat debug_local_results/${model_name}/${FLAVOR}/seed_checkpoint/status.txt)" != "completed" ]; do
        echo "Waiting for seed checkpoint from ${model_name} to complete ..."
        sleep 1
    done
    python ./tooling_dev/test_hf_integration.py submit_jobs --inp_dir debug_local_results/${model_name}/${FLAVOR} --qos high
    echo "================"
done

# for model_name in "${moe_model_names[@]}"; do
#     rm -rf debug_local_results/${model_name}

    #     USE_MOE=1 python ./tooling_dev/test_hf_integration.py create_configs --model_name "$model_name" --out_dir debug_local_results --flavor $FLAVOR
#     USE_MOE=1 python ./tooling_dev/test_hf_integration.py submit_jobs --inp_dir debug_local_results/${model_name}/${FLAVOR}/seed_checkpoint --qos high
#     while [ ! -f debug_local_results/${model_name}/${FLAVOR}/seed_checkpoint/status.txt ] || [ "$(cat debug_local_results/${model_name}/${FLAVOR}/seed_checkpoint/status.txt)" != "
#         echo "Waiting for seed checkpoint from ${model_name} to complete ..."
#         sleep 1
#     done
#     USE_MOE=1 python ./tooling_dev/test_hf_integration.py submit_jobs --inp_dir debug_local_results/${model_name}/debugperf_large --qos high
#     echo "================"
# done