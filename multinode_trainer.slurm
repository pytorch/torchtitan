#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Example allocation. Override these with sbatch command-line options for your
# cluster, including partition, account, and QoS when they are required.
#SBATCH --job-name=torchtitan_multi_node
#SBATCH --nodes=4
# Run one Slurm task per node; torchrun launches one worker per GPU.
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96

set -euo pipefail

# Use the first allocated node for rendezvous. Slurm hostnames should resolve on
# every compute node.
mapfile -t nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
head_node=${nodes[0]}

RUN_ID=${RUN_ID:-"${SLURM_JOB_ID}"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/${RUN_ID}"}
NPROC_PER_NODE=${NPROC_PER_NODE:-"gpu"}

mkdir -p "${OUTPUT_DIR}"

# debugging flags (optional)
export NCCL_DEBUG=${NCCL_DEBUG:-"WARN"}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-"1"}
# optional debug settings
# export NCCL_DEBUG=INFO
# NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV

# Select any registered TorchTitan model and configuration at submission time,
# for example: MODULE=llama3 CONFIG=llama3_8b sbatch ...
MODULE=${MODULE:-"llama3"}
CONFIG=${CONFIG:-"llama3_debugmodel"}

if [[ "${TORCHTITAN_COLLECT_ENVIRONMENT:-0}" == "1" ]]; then
    python3 scripts/collect_environment.py \
        --output "${OUTPUT_DIR}/environment.json"
fi

# "gpu" launches one worker for each GPU that Slurm makes visible to the task.
# Set NPROC_PER_NODE to an integer when a job intentionally uses fewer workers.
srun --kill-on-bad-exit=1 \
    python3 -m torch.distributed.run \
    --nnodes "${SLURM_NNODES}" \
    --nproc-per-node "${NPROC_PER_NODE}" \
    --rdzv-id "${SLURM_JOB_ID}" \
    --rdzv-backend c10d \
    --rdzv-endpoint "${head_node}:29500" \
    -m torchtitan.train \
    --module "${MODULE}" \
    --config "${CONFIG}" \
    "$@" 2>&1 | tee "${OUTPUT_DIR}/training.log"
