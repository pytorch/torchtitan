#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# use envs as local overwrites for convenience
# e.g.
# BACKEND=tt LOG_RANK=0,1 NGPU=4 ./run_train.sh
NGPU=${NGPU:-"8"}
export LOG_RANK=${LOG_RANK:-0}

DEBUG_PORT=${DEBUG_PORT:-5678}
# Option to switch between debug and train
MODE=${MODE:-"train"}  # Set MODE=debug or MODE=train

# Option to switch between hf and tt backend
BACKEND=${BACKEND:-"hf"}

if [ "$BACKEND" = "tt" ]; then
    CONFIG_FILE=${CONFIG_FILE:-"/fsx/ferdinandmom/ferdinand-hf/huggingface/torchtitan/torchtitan/models/llama3/train_configs/my_debug_model.toml"}
elif [ "$BACKEND" = "hf" ]; then
    CONFIG_FILE=${CONFIG_FILE:-"configs/debug_1_gpu_hf.toml"}
else
    echo "Invalid BACKEND set: ${BACKEND}"
    exit 1
fi

if [ "$MODE" = "debug" ]; then
    PYTHON_CMD="debugpy-run -p ${DEBUG_PORT} -m torch.distributed.run --"
else
    PYTHON_CMD="torchrun"
fi

TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

PYTORCH_ALLOC_CONF="expandable_segments:True" \
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
$PYTHON_CMD --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
-m torchtitan.train --job.config_file ${CONFIG_FILE} "$@"