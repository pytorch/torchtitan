#!/usr/bin/bash

set -ex

TRAINER_DIR=${1:-/home/$USER/local/torchtrain}

MODEL="debugmodel"
NGPU=8
MP=4

torchrun --nproc_per_node=${NGPU} \
train.py --steps 10 --compile --meta_init
