#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./torchtitan/experiments/flux/run_preprocess.sh
NGPU=${NGPU:-"8"}
export LOG_RANK=${LOG_RANK:-0}
CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/experiments/flux/train_configs/flux_schnell_model.toml"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

dataset_path="/home/jianiw/tmp/mffuse/cc12m-wds"

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--role rank --tee 3 \
-m torchtitan.experiments.flux.scripts.preprocess.preprocess_dataset \
--job.config_file "${CONFIG_FILE}" \
# --job.dump_folder "outputs/" \
# --training.dataset_path "${dataset_path}" \
# --metrics.disable_color_printing \
# --encoder.t5_encoder "/home/jianiw/tmp/mffuse/flux/t5-v1_1-xxl/" \
# --encoder.clip_encoder "/home/jianiw/tmp/mffuse/flux/clip-vit-large-patch14/" \
# --encoder.autoencoder_path "/home/jianiw/tmp/mffuse/flux/autoencoder/ae.safetensors" \
--parallelism.data_parallel_replicate_degree 8 \
--parallelism.data_parallel_shard_degree 1 \
# --training.classifer_free_guidance_prob 0.0 \
# --job.print_args \
$overrides
