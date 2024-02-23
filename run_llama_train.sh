#!/usr/bin/bash

set -ex

TRAINER_DIR=${1:-/home/$USER/local/torchtrain}

# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 SP=2 ./run_llama_train.sh

MODEL=${MODEL:-"debugmodel"}
NGPU=${NGPU:-"2"}
PP=${PP:-"2"}
SP=${SP:-"1"}
DP=${DP:-"-1"}

# by default log just rank 0 output,
LOG_RANK=${LOG_RANK:-0}

# Change this string to a meaningful one to enable checkpoint
CHECKPOINT_FOLDER=${CHECKPOINT_FOLDER:-""}
# Please adjust this to a longer interval period. The unit of measurement is in steps.
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-5}

export TORCH_NCCL_TRACE_BUFFER_SIZE=20000
export TORCH_NCCL_DEBUG_INFO_PIPE_FILE="/tmp/nccl_dump_pipe_"
export TORCH_CPP_LOG_LEVEL=INFO
torchrun --nproc_per_node=${NGPU} \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
train.py --steps 10 --compile \
--pp_degree ${PP} --sp_degree ${SP} --dp_degree ${DP}
--checkpoint-folder=${CHECKPOINT_FOLDER} --checkpoint-interval=${CHECKPOINT_INTERVAL}
