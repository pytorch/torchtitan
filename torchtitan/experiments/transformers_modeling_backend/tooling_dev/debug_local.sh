#!/usr/bin/bash

# Parse command line arguments
COMPILE_FLAG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --compile)
            COMPILE_FLAG="--enable_compile"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--compile]"
            exit 1
            ;;
    esac
done

# create a list of model_name

model_names=(
    # "meta-llama/Llama-3.2-1B"  # ✅
    # "microsoft/phi-2" # ✅/
    "Qwen/Qwen2.5-7B" # ✅
    # "mistralai/Mistral-7B-v0.1" # ✅
    # "google/gemma-3-270m" # ❌ new layers to handle 
    # "ByteDance-Seed/Seed-Coder-8B-Instruct" # ✅
    # "Qwen/Qwen3-4B-Instruct-2507" # ✅
)

# moe_model_names=(
#     # "deepseek_v3"
#     # "deepseek-ai/DeepSeek-V3"
#     # "moonshotai/Moonlight-16B-A3B"
#     # "openai/gpt-oss-20b"
#     # "moonshotai/Kimi-K2-Instruct"
# )


for model_name in "${model_names[@]}"; do
    rm -rf debug_local_results/${model_name}

    python ./tooling_dev/test_hf_integration.py create_configs --model_name "$model_name" --out_dir debug_local_results --flavor debugmodel $COMPILE_FLAG
    python ./tooling_dev/test_hf_integration.py submit_jobs --inp_dir debug_local_results/${model_name}/debugmodel/seed_checkpoint --qos high
    while [ ! -f debug_local_results/${model_name}/debugmodel/seed_checkpoint/status.txt ] || [ "$(cat debug_local_results/${model_name}/debugmodel/seed_checkpoint/status.txt)" != "completed" ]; do
        echo "Waiting for seed checkpoint from ${model_name} to complete ..."
        sleep 1
    done
    python ./tooling_dev/test_hf_integration.py submit_jobs --inp_dir debug_local_results/${model_name}/debugmodel --qos high
    echo "================"
done

# for model_name in "${moe_model_names[@]}"; do
#     rm -rf debug_local_results/${model_name}

#     USE_MOE=1 python ./tooling_dev/test_hf_integration.py create_configs --model_name "$model_name" --out_dir debug_local_results --flavor debugmodel $COMPILE_FLAG
#     USE_MOE=1 python ./tooling_dev/test_hf_integration.py submit_jobs --inp_dir debug_local_results/${model_name}/debugmodel/seed_checkpoint --qos high
#     while [ ! -f debug_local_results/${model_name}/debugmodel/seed_checkpoint/status.txt ] || [ "$(cat debug_local_results/${model_name}/debugmodel/seed_checkpoint/status.txt)" != "completed" ]; do
#         echo "Waiting for seed checkpoint from ${model_name} to complete ..."
#         sleep 1
#     done
#     USE_MOE=1 python ./tooling_dev/test_hf_integration.py submit_jobs --inp_dir debug_local_results/${model_name}/debugmodel --qos high
#     echo "================"
# done