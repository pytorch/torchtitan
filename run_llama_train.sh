#!/usr/bin/bash

set -ex

TRAINER_DIR=${1:-/home/$USER/local/torchtrain}

MODEL="debugmodel"
NGPU=8
MP=4
# Change this string to a meaningful one to enable checkpoint
CHECKPOINT_FOLDER=""
# Please adjust this to a longer interval period. The unit of measurement is in steps.
CHECKPOINT_INTERVAL=5

torchrun --nproc_per_node=${NGPU} \
train.py --steps 10 --compile \
--checkpoint-folder=${CHECKPOINT_FOLDER} --checkpoint-interval=${CHECKPOINT_INTERVAL}
