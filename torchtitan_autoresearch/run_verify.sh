#!/usr/bin/bash
# Tier-1 single-batch gradient probe launcher. Mirrors run_train.sh but runs the
# verify entrypoint instead of the training loop. Pass the SAME TorchTitan flags
# as the candidate command, plus --verify.* flags (which this script forwards).
#
# Capture a golden (high-precision reference) once per shape:
#   NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./torchtitan_autoresearch/run_verify.sh \
#     --training.seq_len=128 --training.local_batch_size=176 \
#     --debug.seed=42 --debug.deterministic \
#     --verify.mode=capture --verify.snapshot=torchtitan_autoresearch/goldens/seq128_bf16.pt
#
# Compare a candidate:
#   NGPU=8 MODULE=qwen3 CONFIG=qwen3_14b ./torchtitan_autoresearch/run_verify.sh \
#     <candidate flags> --debug.seed=42 --debug.deterministic \
#     --verify.mode=compare --verify.cls=precision \
#     --verify.snapshot=torchtitan_autoresearch/goldens/seq128_bf16.pt \
#     --verify.champion=torchtitan_autoresearch/goldens/champion.pt

set -ex

NGPU=${NGPU:-"8"}
export LOG_RANK=${LOG_RANK:-0}
MODULE=${MODULE:-"qwen3"}
CONFIG=${CONFIG:-"qwen3_14b"}

PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
  --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
  -m torchtitan_autoresearch.verify_main --module ${MODULE} --config ${CONFIG} "$@"
