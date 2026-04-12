#!/bin/bash
# Smoke test: run tiny MoE model for 100 steps on 8 GPUs
set -ex

CONFIG="${CONFIG:-configs/moe_tiny.yaml}"
NGPU="${NGPU:-4}"

PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node="${NGPU}" --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter 0 --role rank --tee 3 \
    train.py --config "${CONFIG}" "$@"
