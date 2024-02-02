#!/usr/bin/bash

set -ex

# libUV is a scalable backend for TCPStore which is used in processGroup
# rendezvous. This is the recommended backend for distributed training.
export USE_LIBUV=1
TRAINER_DIR=${1:-/home/$USER/local/torchtrain}

# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_llama_train.sh

NGPU=${NGPU:-"2"}

# by default log just rank 0 output,
LOG_RANK=${LOG_RANK:-0}


CONFIG_FILE=${CONFIG_FILE:-"./train_configs/debug_model.toml"}
export TORCH_NCCL_TRACE_BUFFER_SIZE=20000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="${HOME}/dumps/nccl_trace_rank_"

torchrun --nproc_per_node=${NGPU} --rdzv_endpoint="localhost:5972" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
train.py --job.config_file ${CONFIG_FILE}
