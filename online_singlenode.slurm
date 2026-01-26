#!/bin/bash
#SBATCH --job-name=singlenode_online_runner
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gpus=8
#SBATCH --cpus-per-task=64

# Create logs directory if it doesn't exist
mkdir -p logs/$SLURM_JOB_ID

# set ulimit higher... dang async io
ulimit -n 32000

LOGDIR="$(pwd)/logs/${SLURM_JOB_ID}"

# Configuration variables (can be overridden when submitting the job)
# Example: sbatch --export=CONFIG_FILE=/path/to/config.yaml,MODEL_NAME=llama2-7b,PYTHON_SCRIPT=/path/to/script.py online_singlenode.slurm
CONFIG_FILE=${CONFIG_FILE:-"/path/to/default/config.yaml"}
MODEL_NAME=${MODEL_NAME:-"default_model"}
PYTHON_SCRIPT=${PYTHON_SCRIPT:-"/path/to/default/script.py"}
PYTHON_ARGS=${PYTHON_ARGS:-""}
TRAINING_ARGS=${TRAINING_ARGS:-""}

# Define environment paths
TRAIN_PATH="/home/dakota/github/torchtitan"
TRAIN_ENV="${TRAIN_PATH}/.venv"
SGLANG_ENV="/home/dakota/github/sglang/.venv"
API_ENV="/home/dakota/github/atropos/.venv"

echo "Starting job at $(date)"
# Setup 4 sglang instances with model in sglang venv
echo "Starting sglang instances..."
source ${SGLANG_ENV}/bin/activate
PORT_BASE=9000
export NUM_INFERENCE_NODES=0

# Start 4 sglang instances on GPUs 0-3
# this assumes you can run it with tp=1
# if not, well, good luck with single node training, I'll pray for you
for i in {4..7}; do
    GPU_ID=$i
    PORT=$((PORT_BASE + i))
    echo "Starting sglang instance on GPU $GPU_ID, port $PORT"
    CUDA_VISIBLE_DEVICES=$GPU_ID nohup python -m sglang.launch_server \
        --model-path $MODEL_NAME \
        --host 0.0.0.0 \
        --mem-fraction-static 0.80 \
        --log-level="error" \
        --attention-backend triton \
        --port $PORT > ${LOGDIR}/sglang_${GPU_ID}.log 2>&1 &
    # Add endpoint to the list
    if [ -z "$ENDPOINTS" ]; then
        ENDPOINTS="localhost:$PORT"
    else
        ENDPOINTS="$ENDPOINTS,localhost:$PORT"
    fi
done
deactivate

source ${TRAIN_ENV}/bin/activate

# Run llama training with 4 GPUs in first venv
cd $TRAIN_PATH
NGPU=4 nohup bash run_llama_train.sh --grpo.sglang_urls=$ENDPOINTS --grpo.sglang_slurm_num_nodes=-1 $TRAINING_ARGS > ${LOGDIR}/train.log 2>&1 &
deactivate


# API and gym setup
source ${API_ENV}/bin/activate
run-api > ${LOGDIR}/api.log 2>&1 &
python /home/dakota/github/atropos/environments/math_server.py serve --slurm=True > ${LOGDIR}/env_server_math.log 2>&1 &
python $PYTHON_SCRIPT serve --slurm=True $PYTHON_ARGS > ${LOGDIR}/env_server.log 2>&1
