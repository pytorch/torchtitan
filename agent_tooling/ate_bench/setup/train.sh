#!/usr/bin/bash
# ATE-Bench fixed-config training launcher for TorchTitan.
#
# This is the "provided training script" the operate-and-profile tasks refer to
# (paper Appendix B.2). It encodes the paper's fixed evaluation config:
#   PP=4, EP=2, DP=1, seq_len 2048, global batch 1024, BF16, 8 GPUs
# mapped onto TorchTitan's run_train.sh. Treat it as READ-ONLY documentation of
# best practice — the agent's job (OP1) is to make the *environment* able to run
# it, not to edit it.
#
# Overridable via env for smoke runs (e.g. a 1-GPU dry run):
#   STEPS=5 NGPU=8 MODULE=deepseek_v3 CONFIG=deepseek_v3_debugmodel \
#   PP=4 EP=2 DP=1 bash agent_tooling/ate_bench/setup/train.sh
#
#   # single-GPU config-validation (no NCCL, no real GPUs):
#   COMM_MODE=fake_backend PP=1 EP=1 DP=1 NGPU=8 bash .../setup/train.sh
set -euo pipefail

STEPS="${STEPS:-5}"
MODULE="${MODULE:-deepseek_v3}"
CONFIG="${CONFIG:-deepseek_v3_debugmodel}"
NGPU="${NGPU:-8}"
PP="${PP:-4}"
EP="${EP:-2}"
# dp_shard: -1 = auto-fill the mesh (world_size/(dp_replicate*cp*tp*pp)). EP is
# carved from this data axis, not multiplied on top, so PP=4 on 8 GPUs -> dp_shard=2.
DP="${DP:--1}"
SEQ_LEN="${SEQ_LEN:-2048}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1024}"

# repo root = three levels up from setup/ (agent_tooling/ate_bench/setup)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

echo "[ate_bench] module=$MODULE config=$CONFIG ngpu=$NGPU mesh(PP=$PP,EP=$EP,DP=$DP) steps=$STEPS"

# run_train.sh reads these from the environment.
export MODULE CONFIG NGPU
[ -n "${COMM_MODE:-}" ] && export COMM_MODE

# With pipeline parallelism the loss is computed on the LAST pipeline stage; rank 0
# (first stage) only prints a sentinel value. Log the last stage's rank so the
# captured loss is the real one (and the loss-curve check sees a true curve).
if [ "${PP}" -gt 1 ] && [ -z "${LOG_RANK:-}" ]; then
  export LOG_RANK="$(( NGPU - NGPU / PP ))"
fi

./run_train.sh \
    --parallelism.pipeline_parallel_degree="$PP" \
    --parallelism.expert_parallel_degree="$EP" \
    --parallelism.data_parallel_shard_degree="$DP" \
    --training.seq_len="$SEQ_LEN" \
    --training.global_batch_size="$GLOBAL_BATCH_SIZE" \
    --training.steps="$STEPS" \
    "$@"
