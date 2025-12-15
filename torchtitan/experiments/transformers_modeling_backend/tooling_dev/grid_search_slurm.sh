#!/usr/bin/bash

# Parse command line arguments
COMPILE_FLAG=""
FLAVOR="debugmodel"
while [[ $# -gt 0 ]]; do
    case $1 in
        --compile)
            COMPILE_FLAG="--enable_compile"
            shift
            ;;
        --flavor)
            FLAVOR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--compile] [--flavor FLAVOR]"
            exit 1
            ;;
    esac
done

# create a list of model_name

model_names=(
    "meta-llama/Llama-3.2-1B" # ✅
    "microsoft/phi-2" # ✅
    "Qwen/Qwen2.5-7B" # ✅
    "mistralai/Mistral-7B-v0.1" # ✅
    "ByteDance-Seed/Seed-Coder-8B-Instruct" # ✅
    "Qwen/Qwen3-4B-Instruct-2507" # ✅
    "arcee-ai/AFM-4.5B" # ✅
    "ibm-granite/granite-3b-code-base-2k"  # ✅
    "baidu/ERNIE-4.5-0.3B-Base-PT" # ✅
    "kyutai/helium-1-preview-2b" # ✅
    "allenai/OLMo-7B-hf" # ✅
    "mistralai/Ministral-8B-Instruct-2410"  # ✅
)

# moe_model_names=(
#     "deepseek-ai/DeepSeek-V3"
#     "moonshotai/Moonlight-16B-A3B"
#     "openai/gpt-oss-20b"
#     "moonshotai/Kimi-K2-Instruct"
#     "zai-org/GLM-4.5"
# )


for model_name in "${model_names[@]}"; do
    rm -rf slurm_results/${model_name}

    python tooling_dev/test_hf_integration.py create_configs --model_name "$model_name" --out_dir slurm_results --flavor $FLAVOR $COMPILE_FLAG
    python tooling_dev/test_hf_integration.py submit_jobs --inp_dir slurm_results/${model_name}/${FLAVOR}/seed_checkpoint --qos high
    while [ ! -f slurm_results/${model_name}/${FLAVOR}/seed_checkpoint/status.txt ] || [ "$(cat slurm_results/${model_name}/${FLAVOR}/seed_checkpoint/status.txt)" != "completed" ]; do
        echo "Waiting for seed checkpoint from ${model_name} to complete ..."
        sleep 1
    done
    python tooling_dev/test_hf_integration.py submit_jobs --inp_dir slurm_results/${model_name}/${FLAVOR} --qos high
    echo "================"
done

for model_name in "${moe_model_names[@]}"; do
    rm -rf slurm_results/${model_name}

    USE_MOE=1 python tooling_dev/test_hf_integration.py create_configs --model_name "$model_name" --out_dir slurm_results --flavor $FLAVOR $COMPILE_FLAG
    USE_MOE=1 python tooling_dev/test_hf_integration.py submit_jobs --inp_dir slurm_results/${model_name}/${FLAVOR}/seed_checkpoint --qos high
    while [ ! -f slurm_results/${model_name}/${FLAVOR}/seed_checkpoint/status.txt ] || [ "$(cat slurm_results/${model_name}/${FLAVOR}/seed_checkpoint/status.txt)" != "completed" ]; do
        echo "Waiting for seed checkpoint from ${model_name} to complete ..."
        sleep 1
    done
    USE_MOE=1 python tooling_dev/test_hf_integration.py submit_jobs --inp_dir slurm_results/${model_name}/${FLAVOR} --qos high
    echo "================"
done
