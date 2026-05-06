#!/usr/bin/env bash
# 20-step GRPO smoke on 2 GPUs (1 trainer + 1 generator) with W&B.
# Override WANDB_PROJECT / WANDB_RUN_NAME to retag the run.
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
MONARCH_RDMA_DISABLE_IBVERBS=1 \
MONARCH_RDMA_ALLOW_TCP_FALLBACK=1 \
python torchtitan/experiments/rl/grpo.py \
  --module rl --config rl_grpo_qwen3_0_6b \
  --num_steps 20 \
  --trainer.parallelism.tensor-parallel-degree 1 \
  --generator.parallelism.tensor-parallel-degree 1 \
  --metrics.enable-wandb \
  "$@"
