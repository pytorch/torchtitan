#!/bin/bash
#SBATCH --job-name=sft_runner
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=224
#SBATCH --export=ALL

# Usage:
#   # Single-node (defaults shown above)
#   sbatch --export=ALL,CONDA_ENV=torchtitan,CONDA_BASE=/path/to/miniconda3 \
#          sft.slurm
#
#   # Multi-node example (override nodes/gpus/cpus as needed)
#   sbatch --nodes=2 --export=ALL,CONDA_ENV=torchtitan,CONDA_BASE=/path/to/miniconda3 \
#          sft.slurm

# Create logs directory if it doesn't exist
mkdir -p logs/$SLURM_JOB_ID

# set ulimit higher... dang async io
ulimit -n 32000
export LOGDIR="$(pwd)/logs/${SLURM_JOB_ID}"

# echo slurm nodes
echo "SLURM nodes: $SLURM_JOB_NODELIST"

: "${TRAIN_PATH:=${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "${TRAIN_PATH}/sft_env.sh"

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
export head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

srun -l --export=ALL "${TRAIN_PATH}/launch_sft.sh"
