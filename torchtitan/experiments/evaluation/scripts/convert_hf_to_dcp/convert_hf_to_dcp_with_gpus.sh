#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Refer to original script:
# https://github.com/pytorch/torchtitan/blob/main/torchtitan/experiments/llama4/scripts/convert_hf_to_dcp_with_gpus.sh.


set -ex

# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./convert_hf_to_dcp_with_gpus.sh
NGPU=${NGPU:-"2"}
LOG_RANK=${LOG_RANK:-0}
CONFIG_FILE=${CONFIG_FILE:-"../../llama3/train_configs/llama3.2_1b.toml"}

CHECKPOINT=""  # TODO: Set the default checkpoint path if needed

# Set the path where you downloaded the HF model
# through `download_hf_checkpoint.py`.
CONVERT_PATH=${CONVERT_PATH:-${CHECKPOINT}/"llama_3.2_1b"} 
# Set the folder where you want to save the converted model
FOLDER=${FOLDER:-${CHECKPOINT}/"llama_3.2_1b_dcp"}
HF_TOKEN=${HF_TOKEN:-""}  # Set your Hugging Face token if needed

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
convert_hf_to_dcp_with_gpus.py --job.config_file ${CONFIG_FILE} \
--checkpoint.enable_checkpoint --checkpoint.convert_path ${CONVERT_PATH} \
--checkpoint.convert_load_every_n_ranks 2 --checkpoint.folder ${FOLDER} \
$overrides  # --checkpoint.convert_hf_token ${HF_TOKEN}