#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./torchtitan/models/flux/run_train.sh
NGPU=${NGPU:-"8"}
export LOG_RANK=${LOG_RANK:-0}
MODEL=${MODEL:-"flux"}
CONFIG=${CONFIG:-"flux_debugmodel"}

# Inference loads model weights from this run's checkpoint folder
# ({job.dump_folder}/{checkpoint.folder}, defaults: outputs/checkpoint).
# initial_load_path requires an absolute path, so resolve it here.
DUMP_FOLDER=${DUMP_FOLDER:-"outputs"}
CHECKPOINT_FOLDER=${CHECKPOINT_FOLDER:-"checkpoint"}
INITIAL_LOAD_PATH="$(realpath -m "${DUMP_FOLDER}/${CHECKPOINT_FOLDER}")"

PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
-m torchtitan.models.flux.inference.infer --module ${MODEL} --config ${CONFIG} \
--checkpoint.enable \
--checkpoint.initial_load_path "${INITIAL_LOAD_PATH}" "$@"
