#!/bin/bash
# Usage: launch_pp.sh <run_name> <config> <gpu_list e.g. 0,1> [extra env...]
# real_pp_fake_spmd: one physical process per PP rank (real PP send/recv),
# every SPMD axis fake at NGPU=256 logical ranks.
set -e
RUN=$1; CONFIG=$2; GPUS=$3; shift 3
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
OUT=/data/users/bahuang/local/torchtitan_2/outputs/memplan/$RUN
mkdir -p "$OUT"; rm -rf "$OUT"/rank*/memstat.jsonl
cd /data/users/bahuang/local/torchtitan_2
env TORCH_SYMMMEM_IMPLICIT_POOL=0 TORCH_DIST_USE_NCCL2=0 "$@" MEMPLAN_OUT="$OUT" NGPU=256 CUDA_VISIBLE_DEVICES=$GPUS \
  /home/bahuang/.conda/envs/pt_dev/bin/torchrun --nproc_per_node=$NPROC \
  --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
  -m mem_planning_study.run --module mem_planning_study.recipes --config "$CONFIG" \
  --comm-backend real_pp_fake_spmd > "$OUT/log.txt" 2>&1
