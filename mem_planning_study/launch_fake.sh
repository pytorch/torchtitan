#!/bin/bash
# Usage: launch_fake.sh <run_name> <config> <gpu> [extra env...]
# Pure fake PG, one process = logical rank 0 of NGPU=256.
set -e
RUN=$1; CONFIG=$2; GPU=$3; shift 3
OUT=/data/users/bahuang/local/torchtitan_2/outputs/memplan/$RUN
mkdir -p "$OUT"; rm -f "$OUT/memstat.jsonl"
cd /data/users/bahuang/local/torchtitan_2
env TORCH_SYMMMEM_IMPLICIT_POOL=0 NGPU=256 "$@" MEMPLAN_OUT="$OUT" LOCAL_RANK=0 CUDA_VISIBLE_DEVICES=$GPU \
  /home/bahuang/.conda/envs/pt_dev/bin/python -m mem_planning_study.run \
  --module mem_planning_study.recipes --config "$CONFIG" --comm-backend fake \
  > "$OUT/log.txt" 2>&1
