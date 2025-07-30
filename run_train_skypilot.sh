#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_train.sh
NGPU=${NGPU:-"8"}
CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/motif/train_configs/motif_10b.toml"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

MASTER_ADDR=$(echo "$SKYPILOT_NODE_IPS" | head -n1)
echo "Starting distributed training, head node: $MASTER_ADDR"

if [ "${SKYPILOT_NODE_RANK}" == "0" ]; then
    # TODO (jeesoo): set proper rank for logging
    LOG_RANK=0
else
    LOG_RANK=-1
fi

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun    --nnodes=${SKYPILOT_NUM_NODES} \
            --nproc_per_node=${SKYPILOT_NUM_GPUS_PER_NODE} \
            --node_rank=${SKYPILOT_NODE_RANK} \
            --local-ranks-filter ${LOG_RANK} \
            --master-addr ${MASTER_ADDR} \
            --master-port 33551 \
            --role rank \
            --tee 3 \
            -m torchtitan.train --job.config_file ${CONFIG_FILE} $overrides