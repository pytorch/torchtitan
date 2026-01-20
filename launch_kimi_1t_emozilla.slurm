#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# --- This script is optimized for AWS with EFA
# --- adjust NCCL_BUFFSIZE if you encounter memory
# --- constraint issues or to tune for improved performance.
# ---

#SBATCH --job-name=kimi_1t_emozilla

#SBATCH --ntasks=40

#SBATCH --nodes=40

#SBATCH --gpus-per-task=8

#SBATCH --cpus-per-task=64

#SBATCH --partition=batch


nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
# Enable for A100
export FI_PROVIDER="efa"
# Ensure that P2P is available
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

# debugging flags (optional)
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
# optional debug settings
# export NCCL_DEBUG=INFO
# NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV

export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=0

# on your cluster you might need these:
# set the network interface
export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"
export NCCL_BUFFSIZE=2097152
#export TORCH_DIST_INIT_BARRIER=1
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0

export TRITON_HOME=/tmp/emotritoncache_$$_$SLURM_PROCID
export NUMBA_CACHE_DIR=/tmp/numbacache_$$_$SLURM_PROCID
mkdir -p $TRITON_HOME $NUMBA_CACHE_DIR

export WANDB_ENTITY="nous_research"
#export WANDB_PROJECT="torchtune"
export WANDB_PROJECT="moe"

export HF_HOME="/home/phuc/.cache/huggingface"
mkdir -p $HF_HOME
#export NVSHMEM_DISABLE_NIC_LOCKING=1
#export NVSHMEM_VERBOSE=3
#export PYTORCH_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.95"

# Activate conda environment on all nodes
export PATH="/home/phuc/kimi_1t/env/bin:$PATH"
export CONDA_PREFIX="/home/phuc/kimi_1t/env"

CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/deepseek_v3/train_configs/kimi_1t_emozilla.toml"}
#CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/llama3/train_configs/llama3_8b.toml"}
#CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/llama3/train_configs/llama3_405b.toml"}


#dcgmi profile --pause
# adjust sbatch --ntasks and sbatch --nodes above and --nnodes below
# to your specific node count, and update target launch file.
srun --export=ALL torchrun --nnodes 40 --nproc_per_node 8 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500" -m torchtitan.train --job.config_file ${CONFIG_FILE} "$@"
#dcgmi profile --resume
