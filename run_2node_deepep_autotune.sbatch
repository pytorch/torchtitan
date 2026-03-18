#!/bin/bash
#SBATCH --job-name=deepep_autotune_2node
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --partition=batch
#SBATCH --time=00:30:00
#SBATCH --output=logs/deepep_autotune_2node-%j.out
#SBATCH --error=logs/deepep_autotune_2node-%j.err

set -ex

TORCHTITAN_DIR="$(pwd)"
VENV_PATH="${TORCHTITAN_DIR}/.venv"
CONFIG_FILE=${CONFIG_FILE:-"torchtitan/models/qwen3/train_configs/qwen3_moe_debug_deepep_2node.toml"}

NODES=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
MASTER_ADDR=${NODES[0]}
MASTER_PORT=29500
NNODES=${#NODES[@]}

echo "Job ID: $SLURM_JOB_ID"
echo "DeepEP Autotune 2-node test: EP=16, autotune=true"
echo "Config: ${CONFIG_FILE}"
echo "Nodes ($NNODES): ${NODES[@]}"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

TORCHRUN_CMD="torchrun --nnodes ${NNODES} --nproc_per_node 8 --rdzv_id ${SLURM_JOB_ID} --rdzv_backend c10d --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} --local-ranks-filter 0 --role rank --tee 3 -m torchtitan.train --job.config_file ${CONFIG_FILE}"

PIDS=()
for node in "${NODES[@]}"; do
    ssh "$node" "source ${VENV_PATH}/bin/activate && cd ${TORCHTITAN_DIR} && \
export LOGLEVEL=INFO && \
export NCCL_DEBUG=WARN && \
export NCCL_SOCKET_IFNAME=bond0 && \
export NCCL_BUFFSIZE=2097152 && \
export TORCH_DIST_INIT_BARRIER=1 && \
export HF_HOME=/tmp/hf_cache && \
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache && \
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0 && \
export NVSHMEM_HCA_LIST=mlx5_4,mlx5_7,mlx5_8,mlx5_9,mlx5_10,mlx5_13,mlx5_14,mlx5_15 && \
export PYTORCH_ALLOC_CONF=expandable_segments:True && \
${TORCHRUN_CMD}" &
    PIDS+=($!)
done

for pid in "${PIDS[@]}"; do wait "$pid"; done
