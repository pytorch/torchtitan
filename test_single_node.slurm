#!/bin/bash
#SBATCH --job-name=kimi_1t_test_1node
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=64
#SBATCH --partition=batch
#SBATCH --time=00:30:00

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "Node IP: $head_node_ip"
export LOGLEVEL=INFO
export FI_PROVIDER="efa"
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=0
export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"
export NCCL_BUFFSIZE=2097152
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0
export TRITON_HOME=/tmp/emotritoncache
export NUMBA_CACHE_DIR=/tmp/numbacache
export WANDB_ENTITY="nous_research"
export WANDB_PROJECT="moe"
export HF_HOME="/home/phuc/.cache/huggingface"
mkdir -p $HF_HOME

# Activate conda environment
export PATH="/home/phuc/kimi_1t/env/bin:$PATH"
export CONDA_PREFIX="/home/phuc/kimi_1t/env"

CONFIG_FILE="./torchtitan/models/deepseek_v3/train_configs/kimi_1t_emozilla.toml"

echo "Testing single node with 8 GPUs"
srun torchrun --nnodes 1 --nproc_per_node 8 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500" -m torchtitan.train --job.config_file ${CONFIG_FILE} "$@"
