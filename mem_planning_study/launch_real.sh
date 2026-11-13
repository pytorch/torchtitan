#!/bin/bash
# Usage: launch_real.sh <run_name> <config> <gpu_list e.g. 0,1> [extra env...]
# Real NCCL torchrun on len(gpu_list) GPUs.
set -e
RUN=$1; CONFIG=$2; GPUS=$3; shift 3
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
OUT=/data/users/bahuang/local/torchtitan_2/outputs/memplan/$RUN
mkdir -p "$OUT"; rm -rf "$OUT"/rank*/memstat.jsonl
cd /data/users/bahuang/local/torchtitan_2
env TORCH_SYMMMEM_IMPLICIT_POOL=0 "$@" MEMPLAN_OUT="$OUT" CUDA_VISIBLE_DEVICES=$GPUS \
  /home/bahuang/.conda/envs/pt_dev/bin/torchrun --nproc_per_node=$NPROC \
  --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
  -m mem_planning_study.run --module mem_planning_study.recipes --config "$CONFIG" \
  > "$OUT/log.txt" 2>&1
