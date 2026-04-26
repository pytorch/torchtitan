#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# Usage:
#   NGPU=8 HARDWARE=h100-sm90 ./torchtitan/experiments/llm_trainer/run_benchmarker.sh \
#       --module graph_trainer.llama3 \
#       --config graph_trainer_llama3_debugmodel
#
#   # With promote:
#   NGPU=8 HARDWARE=h100-sm90 ./torchtitan/experiments/llm_trainer/run_benchmarker.sh \
#       --promote \
#       --module graph_trainer.llama3 \
#       --config graph_trainer_llama3_debugmodel

NGPU=${NGPU:-"1"}
HARDWARE=${HARDWARE:?'Set HARDWARE (e.g. HARDWARE=h100-sm90)'}
export LOG_RANK=${LOG_RANK:-0}

torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    -m torchtitan.experiments.llm_trainer.benchmarker \
    --hardware "${HARDWARE}" \
    "$@"
