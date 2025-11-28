#!/bin/bash

# Shared SFT launch script for multinode runs
set -euo pipefail

ulimit -n 32000

NUM_TRAINING_NODES=${NUM_TRAINING_NODES:-${SLURM_JOB_NUM_NODES:-1}}
TRAIN_GPUS_PER_NODE=${TRAIN_GPUS_PER_NODE:-8}
RDZV_ID=${SFT_RDZV_ID:-101}

if [[ "${SLURM_NODEID:-0}" -ge "${NUM_TRAINING_NODES}" ]]; then
    echo "Skipping node ${SLURM_NODEID} (reserved for non-training work)."
    exit 0
fi

if [[ -n "${TRAIN_ENV_COMMAND:-}" ]]; then
    eval "${TRAIN_ENV_COMMAND}"
elif [[ -n "${TRAIN_ENV:-}" && -f "${TRAIN_ENV}/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "${TRAIN_ENV}/bin/activate"
else
    echo "WARNING: No training environment activation configured; using system Python." >&2
fi

cd "${TRAIN_PATH}"

echo "Node ${SLURM_NODEID} starting SFT training."
nodes=( $( scontrol show hostnames "$SLURM_JOB_NODELIST" ) )
nodes_array=("${nodes[@]}")
head_node=${nodes_array[0]}

echo "Head node: ${head_node}, rendezvous IP: ${head_node_ip}"
export LOGLEVEL=INFO
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONFAULTHANDLER=1
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=0

# WandB configuration (from submit_sft.py or defaults)
export WANDB_TEAM="${WANDB_TEAM:-nous_research}"
export WANDB_PROJECT="${WANDB_PROJECT:-hillclimb}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen3-30b-a3b-sft-$(date +%Y-%m-%d)}"

torchrun \
    --nnodes "${NUM_TRAINING_NODES}" \
    --node_rank "${SLURM_NODEID:-0}" \
    --nproc_per_node "${TRAIN_GPUS_PER_NODE}" \
    --rdzv_id "${RDZV_ID}" \
    --rdzv_backend c10d \
    --rdzv_endpoint "${head_node_ip}:29500" \
    --role rank \
    --tee 3 \
    -m torchtitan.train \
    --job.config_file "${CONFIG_FILE}" \
    ${TRAINING_ARGS:-}

if [[ -n "${TRAIN_ENV_DEACTIVATE_CMD:-}" ]]; then
    eval "${TRAIN_ENV_DEACTIVATE_CMD}" || true
elif declare -F deactivate >/dev/null 2>&1; then
    deactivate
fi
