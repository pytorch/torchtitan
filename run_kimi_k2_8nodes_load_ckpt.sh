#!/bin/bash
#SBATCH --job-name=kimi_k2_ckpt
#SBATCH --ntasks=8
#SBATCH --nodes=8
#SBATCH --gpus-per-task=8
#SBATCH --partition=batch
#SBATCH --time=02:00:00
#SBATCH --output=/home/phuc/workspace/moe/small_prs/pr008_saleforce_lbs/torchtitan/slurm_logs/kimi_k2_ckpt_%j.out
#SBATCH --error=/home/phuc/workspace/moe/small_prs/pr008_saleforce_lbs/torchtitan/slurm_logs/kimi_k2_ckpt_%j.err

set -euo pipefail

source /home/phuc/miniconda3/bin/activate torchtitan

CONFIG_FILE="torchtitan/models/deepseek_v3/train_configs/kimi_k2_ep64_cp1_seq24k_lbs1_16n_load_ckpt.toml"

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES ($SLURM_NODELIST)"
echo "Master: $head_node_ip:29500"
echo "Config: $CONFIG_FILE"
echo "Checkpoint: /home/emozilla/dcp/hermes3-1t-ep64-cp8-upstream/step-1"
echo "GPUs total: $(($SLURM_NNODES * 8))"
echo "================"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
export NCCL_SOCKET_IFNAME="bond0"
export NCCL_BUFFSIZE=2097152
export TORCH_DIST_INIT_BARRIER=1
export HF_HOME="/tmp/hf_cache_${SLURM_JOB_ID}"
export HF_DATASETS_CACHE="/tmp/hf_datasets_${SLURM_JOB_ID}"

cd /home/phuc/workspace/moe/small_prs/pr008_saleforce_lbs/torchtitan

srun torchrun \
    --nnodes $SLURM_NNODES \
    --nproc_per_node 8 \
    --rdzv_id $SLURM_JOB_ID \
    --rdzv_backend c10d \
    --rdzv_endpoint "$head_node_ip:29500" \
    -m torchtitan.train \
    --job.config_file $CONFIG_FILE
