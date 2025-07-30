#!/bin/bash
# Script to run the permute test with 4 GPUs

# Set environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=4
export PYTHONPATH=/data/users/jianiw/torchtitan:$PYTHONPATH
export LOG_RANK=0

# Run the test with torchrun
torchrun --nnodes=1 --nproc_per_node=4 --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT /data/users/jianiw/torchtitan/scripts/test_permute.py
