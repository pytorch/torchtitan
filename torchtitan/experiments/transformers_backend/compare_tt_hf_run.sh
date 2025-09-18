#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex
set -o pipefail

# Common settings
NGPU=${NGPU:-"1"}
export LOG_RANK=${LOG_RANK:-0}

# Parse command line arguments for model selection
MODEL_TYPE=${1:-"llama"}
export MODEL_TYPE

# Set model names based on argument
case $MODEL_TYPE in
    "llama")
        TT_MODEL_NAME="llama3"
        HF_MODEL_NAME="meta-llama/Llama-3.2-1B"
        ;;
    "deepseek")
        TT_MODEL_NAME="deepseek_v3"
        HF_MODEL_NAME="deepseek-ai/DeepSeek-V3"
        ;;
    *)
        echo "Error: Unsupported model type '$MODEL_TYPE'"
        echo "Usage: $0 [llama|deepseek] [additional_args...]"
        echo "  llama   - Uses llama3 for TT and meta-llama/Llama-3.2-1B for HF"
        echo "  deepseek - Uses deepseek_v3 for TT and deepseek-ai/DeepSeek-V3 for HF"
        exit 1
        ;;
esac

echo "Using model type: $MODEL_TYPE"
echo "  TT model: $TT_MODEL_NAME"
echo "  HF model: $HF_MODEL_NAME"

# Shift to remove the model type argument, pass remaining args to training
shift

run_tt() {
    echo "##############################################"
    echo "### Running TorchTitan (native) training ###"
    echo "##############################################"
    TT_CONFIG="/fsx/ferdinandmom/ferdinand-hf/huggingface/torchtitan/torchtitan/experiments/transformers_backend/configs/debug_1_gpu_tt.toml"

    # Use CUDA_VISIBLE_DEVICES=0 for TT run
    CUDA_VISIBLE_DEVICES=0 \
    torchrun --nproc_per_node=${NGPU} --master_port 1234 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    -m torchtitan.train --job.config_file ${TT_CONFIG} --training.seed 42 --training.deterministic --model.name ${TT_MODEL_NAME} "$@"
}

run_hf() {
    echo "#######################################################"
    echo "### Running TorchTitan with HF backend training ###"
    echo "#######################################################"
    HF_CONFIG="/fsx/ferdinandmom/ferdinand-hf/huggingface/torchtitan/torchtitan/experiments/transformers_backend/configs/debug_1_gpu_hf.toml"

    # Use CUDA_VISIBLE_DEVICES=1 for HF run
    CUDA_VISIBLE_DEVICES=1 \
    torchrun --nproc_per_node=${NGPU} --master_port 1235 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    -m torchtitan.train --job.config_file ${HF_CONFIG} --training.seed 42 --training.deterministic --model.name ${HF_MODEL_NAME} "$@"
}

TT_LOG="tt_run.log"
HF_LOG="hf_run.log"
DIFF_LOG="run_diff.log"

export DEBUG_JSON_PATH="/fsx/ferdinandmom/ferdinand-hf/huggingface/torchtitan/torchtitan/experiments/transformers_backend/debug_mode_hf"
run_hf "$@" 2>&1 | tee ${HF_LOG} || true
export DEBUG_JSON_PATH="/fsx/ferdinandmom/ferdinand-hf/huggingface/torchtitan/torchtitan/experiments/transformers_backend/debug_mode_tt"
run_tt "$@" 2>&1 | tee ${TT_LOG} || true
# run_tt "$@" 2>&1 | tee ${HF_LOG}


# Filter logs to remove noisy differences
TT_LOG_FILTERED="${TT_LOG}.filtered"
HF_LOG_FILTERED="${HF_LOG}.filtered"

# This sed command removes timestamps, PIDs, master ports, and other
# volatile details that change between runs.
# Feel free to adjust the regex patterns to better suit your log format.
sed -E \
    -e 's/([0-9]{4}-[0-9]{2}-[0-9]{2} )?[0-9]{2}:[0-9]{2}:[0-9]{2}(,[0-9]+)?/TIMESTAMP/g' \
    -e 's/torchrun.*--master_port[= ]([0-9]+)/torchrun ... --master_port=XXXX/g' \
    -e 's/PID [0-9]+/PID XXXX/g' \
    -e 's/localhost:[0-9]+/localhost:XXXX/g' \
    < "${TT_LOG}" > "${TT_LOG_FILTERED}"

sed -E \
    -e 's/([0-9]{4}-[0-9]{2}-[0-9]{2} )?[0-9]{2}:[0-9]{2}:[0-9]{2}(,[0-9]+)?/TIMESTAMP/g' \
    -e 's/torchrun.*--master_port[= ]([0-9]+)/torchrun ... --master_port=XXXX/g' \
    -e 's/PID [0-9]+/PID XXXX/g' \
    -e 's/localhost:[0-9]+/localhost:XXXX/g' \
    < "${HF_LOG}" > "${HF_LOG_FILTERED}"

echo "############################################"
echo "### Diff between TT and HF run logs      ###"
echo "############################################"
echo "### Log diff is being saved to ${DIFF_LOG}"
echo "############################################"
git diff --no-index --color=always --word-diff=color "${TT_LOG_FILTERED}" "${HF_LOG_FILTERED}" | tee "${DIFF_LOG}" || true
