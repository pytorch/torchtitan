#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# libUV is a scalable backend for TCPStore which is used in processGroup
# rendezvous. This is the recommended backend for distributed training.
export USE_LIBUV=1
TRAINER_DIR=${1:-/home/$USER/local/torchtitan}

# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_llama_train.sh

NGPU=${NGPU:-"2"}

# by default log just rank 0 output,
LOG_RANK=${LOG_RANK:-0}


CONFIG_FILE=${CONFIG_FILE:-"./train_configs/debug_model.toml"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
train.py --job.config_file ${CONFIG_FILE} $overrides
