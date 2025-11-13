#!/bin/bash
# Run DeepEP tuning for single-node setup

EP_SIZE=${1:-4}
MODE=${2:-full}

echo "========================================="
echo "DeepEP Single-Node Tuner"
echo "========================================="
echo "EP Size: $EP_SIZE"
echo "Mode: $MODE"
echo "========================================="

# Activate venv
if [ -z "$VIRTUAL_ENV" ]; then
    source ../../../.venv/bin/activate
fi

# Run tuning
torchrun \
    --nproc_per_node=${EP_SIZE} \
    --rdzv_backend c10d \
    --rdzv_endpoint="localhost:0" \
    tune_singlenode.py \
    --ep-size ${EP_SIZE} \
    --mode ${MODE} \
    --output results_ep${EP_SIZE}.json

echo "========================================="
echo "Complete!"
echo "========================================="
