#!/usr/bin/bash
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


set -ex

# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./torchtitan/experiments/flux/run_inference.sh
NGPU=${NGPU:-"8"}
export LOG_RANK=${LOG_RANK:-0}
CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/experiments/flux/train_configs/flux_schnell_mlperf.toml"}
export HF_HUB_CACHE=/root/.cache/huggingface/hub/
overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi


PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
-m torchtitan.experiments.flux.scripts.preprocess_flux_dataset --job.config_file ${CONFIG_FILE} \
--experimental.custom_args_module torchtitan.experiments.flux.preprocessing_config \
--eval.dataset= \
--checkpoint.no_enable_checkpoint --training.batch_size 256 --training.dataset=cc12m-disk --training.dataset_path=/dataset/cc12m_disk \
--parallelism.data_parallel_replicate_degree=${NGPU} \
--training.classifer_free_guidance_prob=0.0 \
--model.flavor=flux-debug $overrides

mkdir -p /dataset/empty_encodings
torchrun --nproc_per_node=1 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
-m torchtitan.experiments.flux.scripts.save_empty_encodings --job.config_file ${CONFIG_FILE} \
--experimental.custom_args_module torchtitan.experiments.flux.preprocessing_config \
--eval.dataset= \
--preprocessing.output_dataset_path=/dataset/empty_encodings \
--training.classifer_free_guidance_prob=0.0 \
--checkpoint.no_enable_checkpoint --training.batch_size 256 --training.dataset=dummy \
--model.flavor=flux-debug --encoder.empty_encodings_path=
