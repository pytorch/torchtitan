#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./convert_meta_to_dcp_with_gpus.sh
NGPU=${NGPU:-"8"}
LOG_RANK=${LOG_RANK:-0,1,2,3,4,5,6,7}
CONFIG_FILE=${CONFIG_FILE:-"../train_configs/llama4_17bx16e.toml"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
convert_meta_to_dcp_with_gpus_meta.py --job.config_file ${CONFIG_FILE} $overrides
