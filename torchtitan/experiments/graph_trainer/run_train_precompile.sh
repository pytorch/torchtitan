#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Dedicated training script for graph_trainer with CooR precompile support.
#
# Each worker runs on its real device (cuda:{local_rank}) with all GPUs
# visible. A CooR artifact is compiled on one rank (cuda:0) but is device
# hoisted: inductor regions resolve the device at runtime
# (config.runtime_device_index) and the eager FX graph's baked constants are
# remapped to the current device at load. This replaces the old
# --virtual-local-rank hack (which masqueraded every rank as cuda:0 via
# CUDA_VISIBLE_DEVICES) and is required for features like symmetric memory
# that need peer GPUs visible.

set -ex

NGPU=${NGPU:-"8"}
export LOG_RANK=${LOG_RANK:-0}
MODULE=${MODULE:-"graph_trainer.llama3"}
CONFIG=${CONFIG:-"graph_trainer_llama3_debugmodel"}

PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
-m torchtitan.train --module ${MODULE} --config ${CONFIG} "$@"
