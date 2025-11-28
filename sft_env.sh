#!/bin/bash

# Shared environment defaults for SFT SLURM jobs.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd)"
: "${TRAIN_PATH:=${SCRIPT_DIR}}"

: "${CONFIG_FILE:=/path/to/default/config.yaml}"
: "${MODEL_NAME:=default_model}"
: "${PYTHON_SCRIPT:=/path/to/default/script.py}"
: "${PYTHON_ARGS:=}"
: "${TRAINING_ARGS:=}"

if [[ -z "${NUM_TRAINING_NODES:-}" ]]; then
    if [[ -n "${SLURM_JOB_NUM_NODES:-}" ]]; then
        export NUM_TRAINING_NODES="${SLURM_JOB_NUM_NODES}"
    else
        export NUM_TRAINING_NODES=1
    fi
fi

: "${NUM_INFERENCE_NODES:=0}"

export CONFIG_FILE MODEL_NAME PYTHON_SCRIPT PYTHON_ARGS TRAINING_ARGS NUM_TRAINING_NODES NUM_INFERENCE_NODES

export NCCL_BUFFSIZE=33554432
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_IB_AR_THRESHOLD=0
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_IB_QPS_PER_CONNECTION=2
export NCCL_IB_SPLIT_DATA_ON_QPS=0
export NCCL_IGNORE_CPU_AFFINITY=1
export NCCL_IB_HCA=mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1,mlx5_10:1,mlx5_13:1,mlx5_14:1,mlx5_15:1
export NCCL_SOCKET_IFNAME=bond0
export UCX_NET_DEVICES=bond0

: "${TRAIN_ENV:=${TRAIN_PATH}/.venv}"
: "${VENV_ENV:=}"
: "${TRAIN_CONDA_ENV:=}"
: "${CONDA_ENV:=${TRAIN_CONDA_ENV:-}}"
: "${CONDA_BASE:=}"
: "${CONDA_PROFILE_SCRIPT:=}"
: "${TRAIN_ENV_COMMAND:=}"
: "${TRAIN_ENV_DEACTIVATE_CMD:=}"
: "${TRAIN_GPUS_PER_NODE:=8}"

if [[ -z "${TRAIN_ENV_COMMAND}" && -n "${CONDA_ENV}" ]]; then
    if [[ -n "${CONDA_PROFILE_SCRIPT}" && -f "${CONDA_PROFILE_SCRIPT}" ]]; then
        TRAIN_ENV_COMMAND="source ${CONDA_PROFILE_SCRIPT} && conda activate ${CONDA_ENV}"
        TRAIN_ENV_DEACTIVATE_CMD="conda deactivate"
    elif [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
        TRAIN_ENV_COMMAND="source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV}"
        TRAIN_ENV_DEACTIVATE_CMD="conda deactivate"
    else
        echo "WARNING: CONDA_ENV='${CONDA_ENV}' requested but neither CONDA_BASE nor CONDA_PROFILE_SCRIPT is set." >&2
    fi
fi

if [[ -z "${TRAIN_ENV_COMMAND}" && -n "${VENV_ENV}" && -f "${VENV_ENV}/bin/activate" ]]; then
    TRAIN_ENV_COMMAND="source ${VENV_ENV}/bin/activate"
    TRAIN_ENV_DEACTIVATE_CMD="deactivate"
fi

if [[ -z "${TRAIN_ENV_COMMAND}" && -n "${TRAIN_ENV}" && -f "${TRAIN_ENV}/bin/activate" ]]; then
    TRAIN_ENV_COMMAND="source ${TRAIN_ENV}/bin/activate"
    TRAIN_ENV_DEACTIVATE_CMD="deactivate"
fi

if [[ -n "${TRAIN_ENV_COMMAND}" ]]; then
    export TRAIN_ENV_COMMAND TRAIN_ENV_DEACTIVATE_CMD
else
    echo "WARNING: TRAIN_ENV_COMMAND is not set; proceeding without env activation." >&2
fi

export TRAIN_PATH TRAIN_ENV VENV_ENV CONDA_ENV CONDA_BASE CONDA_PROFILE_SCRIPT TRAIN_GPUS_PER_NODE
