#!/bin/bash
#SBATCH --job-name=multinode_online_runner
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=64

# Create logs directory if it doesn't exist
mkdir -p logs/$SLURM_JOB_ID

# set ulimit higher... dang async io
ulimit -n 32000
export LOGDIR="$(pwd)/logs/${SLURM_JOB_ID}"


# echo slurm nodes
echo "SLURM nodes: $SLURM_JOB_NODELIST"

# basic config stuff
export CONFIG_FILE=${CONFIG_FILE:-"/path/to/default/config.yaml"}
export MODEL_NAME=${MODEL_NAME:-"default_model"}
export PYTHON_SCRIPT=${PYTHON_SCRIPT:-"/path/to/default/script.py"}
export PYTHON_ARGS=${PYTHON_ARGS:-""}
export TRAINING_ARGS=${TRAINING_ARGS:-""}
export NUM_TRAINING_NODES=${NUM_TRAINING_NODES:-1}
export NUM_INFERENCE_NODES=${NUM_INFERENCE_NODES:-1}

# Weights & Biases
export WANDB_PROJECT=${WANDB_PROJECT:-"torchtitan-grpo"}

# PEFT base directory for LoRA adapters
export PEFT_BASE_DIR=${PEFT_BASE_DIR:-"/home/shared/peft_bases"}

# Define environment paths (needed early for PEFT setup)
export TRAIN_PATH="/home/dakota/github/torchtitan"
export TRAIN_ENV="${TRAIN_PATH}/.venv"
export VLLM_ENV="${TRAIN_PATH}/.venv"
export API_ENV="/home/dakota/github/trajectory-handler/.venv"

# Setup PEFT base adapter (creates if doesn't exist)
# Reads LoRA config (rank, alpha, dropout) from the train config file
echo "Setting up PEFT base adapter..."
source ${VLLM_ENV}/bin/activate
eval $(python scripts/setup_peft_base.py \
    --config-file "$CONFIG_FILE" \
    --base-dir "$PEFT_BASE_DIR")
deactivate
echo "PEFT_PATH: $PEFT_PATH"
echo "BASE_MODEL: $BASE_MODEL"
echo "LORA_RANK: $LORA_RANK"
echo "LORA_ALPHA: $LORA_ALPHA"
echo "LORA_DROPOUT: $LORA_DROPOUT"
export PEFT_PATH
export BASE_MODEL
export LORA_RANK
export LORA_ALPHA
export LORA_DROPOUT

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

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
export head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

srun -l --export=ALL ./vllm_lora_launch.sh
