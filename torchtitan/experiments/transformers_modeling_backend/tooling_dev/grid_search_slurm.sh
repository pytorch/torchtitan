#!/usr/bin/bash

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

    python test_hf_integration.py create_configs --model_name "$model_name" --out_dir slurm_results --flavor debugmodel
    python test_hf_integration.py submit_jobs --inp_dir slurm_results/${model_name}/debugmodel/seed_checkpoint --qos high
    while [ ! -f slurm_results/${model_name}/debugmodel/seed_checkpoint/status.txt ] || [ "$(cat slurm_results/${model_name}/debugmodel/seed_checkpoint/status.txt)" != "completed" ]; do
        echo "Waiting for seed checkpoint from ${model_name} to complete ..."
        sleep 1
    done
    python test_hf_integration.py submit_jobs --inp_dir slurm_results/${model_name}/debugmodel --qos high
    echo "================"
done

for model_name in "${moe_model_names[@]}"; do
    rm -rf slurm_results/${model_name}

    USE_MOE=1 python test_hf_integration.py create_configs --model_name "$model_name" --out_dir slurm_results --flavor debugmodel
    USE_MOE=1 python test_hf_integration.py submit_jobs --inp_dir slurm_results/${model_name}/debugmodel/seed_checkpoint --qos high
    while [ ! -f slurm_results/${model_name}/debugmodel/seed_checkpoint/status.txt ] || [ "$(cat slurm_results/${model_name}/debugmodel/seed_checkpoint/status.txt)" != "completed" ]; do
        echo "Waiting for seed checkpoint from ${model_name} to complete ..."
        sleep 1
    done
    USE_MOE=1 python test_hf_integration.py submit_jobs --inp_dir slurm_results/${model_name}/debugmodel --qos high
    echo "================"
done