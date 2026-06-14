#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

# use envs as local overwrites for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_train.sh
#
# COMM_MODE options for debugging:
#
# 1. "fake_backend" - Dry-run mode for config validation without GPU execution
#    - Uses fake process groups (no actual communication)
#    - Runs on a single GPU without torchrun or NCCL initialization
#    - Useful for validating configuration and model setup
#    Example: NGPU=32 COMM_MODE="fake_backend" ./run_train.sh
#
# 2. "local_tensor" - Single-GPU debugging mode with simulated multi-GPU behavior
#    - All communication and computation execute on a single shared GPU
#    - Simulates the full training workflow without actual distributed communication
#    - Useful for debugging distributed training logic locally
#    Example: NGPU=32 COMM_MODE="local_tensor" ./run_train.sh

NGPU=${NGPU:-"$(nvidia-smi -L | wc -l)"}
export LOG_RANK=${LOG_RANK:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_SHM_LOCALITY=${NCCL_SHM_LOCALITY:-1}
MODULE=${MODULE:-"llama3"}
CONFIG=${CONFIG:-"llama3_debugmodel"}
COMM_MODE=${COMM_MODE:-""}
NNODES=${NNODES:-${SLURM_JOB_NUM_NODES:-1}}
NODE_RANK=${NODE_RANK:-${SLURM_NODEID:-0}}

# xx injects these
TORCHTITAN_ARGS=()
for arg in "$@"; do
  case "$arg" in
    codedir=*) TORCHTITAN_ARGS+=("--codedir=${arg#codedir=}") ;;
    master_addr=*) MASTER_ADDR="${arg#master_addr=}" ;;
    master_port=*) MASTER_PORT="${arg#master_port=}" ;;
    --metrics.enable_reporterv2=) TORCHTITAN_ARGS+=("--metrics.enable_reporterv2") ;;
    *) TORCHTITAN_ARGS+=("$arg") ;;
  esac
done
set -- "${TORCHTITAN_ARGS[@]}"

MASTER_PORT=${MASTER_PORT:-12355}

TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}

generate_uuid() {
  if [[ -z "${1:-}" ]]; then
    cat /proc/sys/kernel/random/uuid
    return
  fi

  local hex
  hex=$(printf '%s' "$1" | sha1sum)
  hex=${hex%% *}
  printf '%s-%s-%s-%s-%s\n' "${hex:0:8}" "${hex:8:4}" "${hex:12:4}" "${hex:16:4}" "${hex:20:12}"
}

export REPORTERV2_HOST="${REPORTERV2_HOST:-mkv://data-gen.comma.life:3080/reporterv2}"

if [[ -z "${REPORTERV2_TRAINING_ID:-}" ]]; then
  reporterv2_seed=""
  if [[ -n "${SLURM_ARRAY_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    reporterv2_seed="slurm:${SLURM_ARRAY_JOB_ID}:${SLURM_ARRAY_TASK_ID}"
  elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
    reporterv2_seed="slurm:${SLURM_JOB_ID}"
  elif [[ -n "${RDZV_ID:-}" ]]; then
    reporterv2_seed="rdzv:${RDZV_ID}"
  fi
  REPORTERV2_TRAINING_ID="$(generate_uuid "$reporterv2_seed")"
fi
export REPORTERV2_TRAINING_ID

if [ -n "$COMM_MODE" ]; then
    # Communication mode specified: validate configuration or run in debug mode
    echo "Running with comm_mode=${COMM_MODE}"
    NGPU="${NGPU}" LOCAL_RANK=0 python3 -m torchtitan.train --module ${MODULE} --config ${CONFIG} "$@" --comm.mode=${COMM_MODE} --training.steps 1
else
    if [[ -n "${MASTER_ADDR:-}" ]]; then
        RDZV_ENDPOINT="${MASTER_ADDR}:${MASTER_PORT}"
    else
        RDZV_ENDPOINT="localhost:0"
    fi

    # Normal training with torchrun
    PYTORCH_ALLOC_CONF="expandable_segments:True" \
    TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
    torchrun --nnodes=${NNODES} --node_rank=${NODE_RANK} --nproc_per_node=${NGPU} \
    --rdzv_id=${RDZV_ID:-${SLURM_JOB_ID:-$(generate_uuid)}} --rdzv_backend c10d \
    --rdzv_endpoint="${RDZV_ENDPOINT}" \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    -m torchtitan.train --module ${MODULE} --config ${CONFIG} "$@"
fi
