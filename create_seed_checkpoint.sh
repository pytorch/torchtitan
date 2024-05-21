#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# create_seed_checkpoint.sh
#
# Run this script to create a seed checkpoint used to initialize a model from step-0.
# Seed checkpoints are used to initialize pipeline-parallel models since the model initializer
# functions don't cleanly run on chunked model parts after meta-initialization.
#
# Use the same model config to generate your seed checkpoint as you use for training.
# e.g.
# CONFIG=<path to model_config> ./create_seed_checkpoint.sh

set -ex

export USE_LIBUV=1
TRAINER_DIR=${1:-/home/$USER/local/torchtitan}
NGPU=1
LOG_RANK=0
CONFIG_FILE=${CONFIG_FILE:-"./train_configs/debug_model.toml"}

seed_checkpoint="--checkpoint.enable_checkpoint --checkpoint.create_seed_checkpoint"
force_1d="--training.data_parallel_degree 1 --training.tensor_parallel_degree 1 --experimental.pipeline_parallel_degree 1"
overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
train.py --job.config_file ${CONFIG_FILE} $seed_checkpoint $force_1d $overrides
