#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# use envs as local overrides for convenience
# e.g.
# NGPU=4 ./run_runtime_estimation.sh --runtime_estimation.estimate_mode_type operator-level-cost-model
# NGPU=4 ./run_runtime_estimation.sh --runtime_estimation.estimate_mode_type operator-level-benchmark
NGPU=${NGPU:-"8"}
NNODES=${NNODES:-"1"}
CONFIG_FILE=${CONFIG_FILE:-"./train_configs/llama3_8b.toml"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

# Calculate WORLD_SIZE as the product of NGPU and NNODES
# Export WORLD_SIZE and LOCAL_RANK
export WORLD_SIZE=$((NGPU * NNODES))
export LOCAL_RANK=0
python runtime_estimation.py --job.config_file ${CONFIG_FILE} $overrides
