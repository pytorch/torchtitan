#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# PTQ calibration + NVFP4 export script.
# Uses torchtitan's distributed infrastructure for memory-efficient calibration.
#
# Usage:
#   NGPU=4 CONFIG_FILE=path/to/ptq.toml OUTPUT_DIR=./outputs/nvfp4 bash run_ptq.sh
#
# Optional:
#   --no_pack_nvfp4    Save BF16 weights without NVFP4 packing

set -ex

NGPU=${NGPU:-"4"}
export LOG_RANK=${LOG_RANK:-0}
CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/gpt_oss/train_configs/persona_kappa_ptq.toml"}
OUTPUT_DIR=${OUTPUT_DIR:-"./outputs/nvfp4_export"}

PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
scripts/ptq_export.py --output_dir ${OUTPUT_DIR} --job.config_file ${CONFIG_FILE} "$@"
