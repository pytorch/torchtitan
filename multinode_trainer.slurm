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

# debugging flags (optional)
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
# optional debug settings
# export NCCL_DEBUG=INFO
# NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV

# Select any registered TorchTitan model and configuration at submission time,
# for example: MODULE=llama3 CONFIG=llama3_8b sbatch ...
MODULE=${MODULE:-"llama3"}
CONFIG=${CONFIG:-"llama3_debugmodel"}

# "gpu" launches one worker for each GPU that Slurm makes visible to the task.
srun --kill-on-bad-exit=1 \
    torchrun \
    --nnodes "${SLURM_NNODES}" \
    --nproc-per-node gpu \
    --rdzv-id "${SLURM_JOB_ID}" \
    --rdzv-backend c10d \
    --rdzv-endpoint "${head_node}:29500" \
    -m torchtitan.train \
    --module "${MODULE}" \
    --config "${CONFIG}" \
    "$@"
