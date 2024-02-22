#!/usr/bin/bash

set -ex

# libUV is a scalable backend for TCPStore which is used in processGroup
# rendezvous. This is the recommended backend for distributed training.
export USE_LIBUV=1
TRAINER_DIR=${1:-/home/$USER/local/torchtrain}

# use envs as local overrides for convenience
# e.g.
# LOG_RANK=0,1 NGPU=4 ./run_llama_train.sh

NGPU=${NGPU:-"8"}

# by default log just rank 0 output,
LOG_RANK=${LOG_RANK:-0}


CONFIG_FILE=${CONFIG_FILE:-"./train_configs/llama_7b.toml"}

torchrun --nproc_per_node=${NGPU} --rdzv-backend=c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
train.py --job.config_file ${CONFIG_FILE}
