#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_llama_train.sh
NGPU=${NGPU:-"2"}
LOG_RANK=${LOG_RANK:-0,1}
CONFIG_FILE=${CONFIG_FILE:-"./train_configs/debug_model.toml"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"./outputs/checkpoint/"}
PROMPT=${PROMPT:-"Hello!"}

overrides=""
if [ $# -ne 0 ]; then
	overrides="$*"
fi

# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_BLOCKING_WAIT=1
# export NCCL_ASYNC_ERROR_HANDLING=1

torchrun --standalone \
         --nproc_per_node="${NGPU}" \
         --local-ranks-filter="${LOG_RANK}" \
         test/generate/test_generate_dist.py \
         --config="${CONFIG_FILE}" \
         --checkpoint="${CHECKPOINT_DIR}" \
         --prompt="${PROMPT}" \
         ${overrides}
