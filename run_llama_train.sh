#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# libUV is a scalable backend for TCPStore which is used in processGroup
# rendezvous. This is the recommended backend for distributed training.
export USE_LIBUV=1
TRAINER_DIR=${TRAINER_DIR:-/home/$USER/local/torchtitan}

# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_llama_train.sh

NGPU=${NGPU:-"8"}
NNODES=${NNODES:-"1"}

# by default log just rank 0 output,
LOG_RANK=${LOG_RANK:-0}


CONFIG_FILE=${CONFIG_FILE:-"./train_configs/debug_model.toml"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

# Check if --estimate.memory=True is in the arguments
if echo "$overrides" | grep -q -- "--estimate.memory=True"; then
    # Calculate WORLD_SIZE as the product of NGPU and NNODES
    # Export WORLD_SIZE and LOCAL_RANK
    export WORLD_SIZE=$((NGPU * NNODES))
    export LOCAL_RANK=0
    python estimation.py --job.config_file ${CONFIG_FILE} $overrides
else
    # Call train.py if not in estimation mode
    torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    train.py --job.config_file ${CONFIG_FILE} $overrides
fi
