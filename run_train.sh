#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex
# export CUDA_VISIBLE_DEVICES=1,2,3,5
# NGPU=4 CONFIG_FILE="./torchtitan/models/qwen3/train_configs/qwen3_0.6b.toml" ./run_train.sh

# NGPU=4 CONFIG_FILE="./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b.toml" ./run_train.sh
# use envs as local overwrites for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_train.sh
# COMM_MODE="fake_backend" ./run_train.sh  # for config validation without GPU
# COMM_MODE="local_tensor" ./run_train.sh  # for local tensor debugging mode

# DEBUG=1 ./run_train.sh  # enable debugpy for attaching debugger (ports 5678+)
# DEBUG=1 DEBUG_WAIT_RANKS="0,1" DEBUG_TIMEOUT=60 ./run_train.sh  # wait for specific ranks with custom timeout
# DEBUG=1 DEBUG_WAIT_RANKS="0" NGPU=8 CONFIG_FILE="./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b.toml" ./run_train.sh
# NGPU=4 DEBUG=1 DEBUG_WAIT_RANKS="0" CONFIG_FILE="./torchtitan/models/qwen3/train_configs/qwen3_0.6b.toml" ./run_train.sh

# DEBUG=1 DEBUG_WAIT_RANKS="0" NGPU=8 CONFIG_FILE="./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b.toml" ./run_train.sh

NGPU=${NGPU:-"8"}
export LOG_RANK=${LOG_RANK:-0}
CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/llama3/train_configs/debug_model.toml"}
TRAIN_FILE=${TRAIN_FILE:-"torchtitan.train"}
# COMM_MODE options: "fake_backend" (dry run), "local_tensor" (debug mode), or empty for normal training
COMM_MODE=${COMM_MODE:-""}
# DEBUG/DEBUGPY: set to 1/true/yes to enable debugpy servers on ports 5678+ (one per rank)
# DEBUG_WAIT_RANKS: comma-separated list of ranks to wait for, or "all" for all ranks (default: "all")
# DEBUG_TIMEOUT: timeout in seconds to wait for debugger attachment (default: 30)
export DEBUG=${DEBUG:-${DEBUGPY:-""}}
export DEBUG_WAIT_RANKS=${DEBUG_WAIT_RANKS:-""}
export DEBUG_TIMEOUT=${DEBUG_TIMEOUT:-""}

# NGPU=8 CONFIG_FILE="./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b.toml" ./run_train.sh
# DEBUG=1 DEBUG_WAIT_RANKS="all" NGPU=8 CONFIG_FILE="./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b.toml" ./run_train.sh

# NGPU=8 DEBUG=1 DEBUG_WAIT_RANKS="0" CONFIG_FILE="./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b.toml" ./run_train.sh


# NGPU=8 DEBUG=1 DEBUG_WAIT_RANKS="0" CONFIG_FILE="./torchtitan/models/qwen3/train_configs/qwen3_0.6b.toml" ./run_train.sh
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

# python scripts/download_hf_assets.py \
#     --repo_id deepseek-ai/DeepSeek-V3.1-Base \
#     --local_dir ./assets/hf \
#     --assets tokenizer config index


# python scripts/download_hf_assets.py \
#     --repo_id deepseek-ai/DeepSeek-V3.1-Base \
#     --local_dir /root/.cache/team_artifacts/ \
#     --all


if [ -n "$COMM_MODE" ]; then
    # Communication mode specified: validate configuration or run in debug mode
    echo "Running with comm_mode=${COMM_MODE}"
    NGPU="${NGPU}" LOCAL_RANK=0 python3 -m "${TRAIN_FILE}" --job.config_file "${CONFIG_FILE}" "$@" --comm.mode=${COMM_MODE} --training.steps=1
else
    # Normal training with torchrun
    PYTORCH_ALLOC_CONF="expandable_segments:True" \
    TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
    torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    -m ${TRAIN_FILE} --job.config_file ${CONFIG_FILE} "$@"
fi
