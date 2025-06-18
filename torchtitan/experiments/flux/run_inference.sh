#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./torchtitan/experiments/flux/run_inference.sh
NGPU=${NGPU:-"8"}
export LOG_RANK=${LOG_RANK:-0}
OUTPUT_DIR=${OUTPUT_DIR:-"inference_results"}
CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/experiments/flux/train_configs/debug_model.toml"}
LOGDIR=${LOGDIR:-"/logdir"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi


PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
-m torchtitan.experiments.flux.infer --job.config_file ${CONFIG_FILE} \
--job.dump_folder=${LOGDIR} --checkpoint.exclude_from_loading lr_scheduler,dataloader,optimizer \
--inference.save_path ${OUTPUT_DIR} --checkpoint.enable_checkpoint $overrides
