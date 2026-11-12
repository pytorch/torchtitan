#!/bin/bash
# Capture one model's eager and graph artifacts back to back on one GPU.
#   outputs/finder_substrate/capture/run.sh <dsv3|qwen3_30b|llama3_8b> <gpu>
set -uo pipefail
MODEL=$1
GPU=$2
REPO=/home/bahuang/local/torchtitan
HERE=$REPO/outputs/finder_substrate
S=$REPO/torchtitan/experiments/graph_trainer/.claude/skills/fusion_finder/scripts
case $MODEL in
  dsv3) NGPU=256 ;;
  *) NGPU=8 ;;
esac
cd $REPO
for ARM in eager graph; do
  OUT=$HERE/captures/${MODEL}_${ARM}
  mkdir -p $OUT
  echo "$(date -Is) start $MODEL $ARM on GPU $GPU" | tee -a $HERE/captures/log.txt
  env OUT=$OUT PYTHONPATH=$HERE/capture:$S:. NGPU=$NGPU LOCAL_RANK=0 LOG_RANK=0 \
    CUDA_VISIBLE_DEVICES=$GPU PYTORCH_ALLOC_CONF=expandable_segments:True \
    python -m torchtitan.train --module recipes --config ${MODEL}_${ARM} --comm-backend fake \
    > $OUT/train.log 2>&1
  echo "$(date -Is) end $MODEL $ARM exit $?" | tee -a $HERE/captures/log.txt
done
