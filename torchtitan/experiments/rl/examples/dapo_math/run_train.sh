#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exo pipefail

# Environment variables override launcher defaults, for example:
# TRAINING_STEPS=10 DUMP_FOLDER=outputs/rl/dapo_smoke ./torchtitan/experiments/rl/examples/dapo_math/run_train.sh
MODULE=${MODULE:-"dapo_math"}
CONFIG=${CONFIG:-"rl_dapo_qwen3_4b_math_8k_tp2"}
TRAINING_STEPS=${TRAINING_STEPS:-150}
DUMP_FOLDER=${DUMP_FOLDER:-"outputs/rl/qwen3_4b_dapo_math_8k_tp2_150"}

# Resolve the worktree that contains this script, regardless of the caller's cwd.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)
cd "${REPO_ROOT}"

# Monarch-spawned trainer and generator processes import this checkout.
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"

# Limit CPU thread pools because every actor process inherits these settings.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
# Same-host weight sync resolves to shared memory before the cross-host transport.
export USE_TORCHCOMMS=${USE_TORCHCOMMS:-0}

# Same-host GET returns the shared-memory tensor directly instead of cloning each shard.
export TORCHSTORE_MUTABLE_SHM=${TORCHSTORE_MUTABLE_SHM:-1}

# This single-node recipe uses shared memory and does not require InfiniBand.
export MONARCH_RDMA_DISABLE_IBVERBS=${MONARCH_RDMA_DISABLE_IBVERBS:-1}
export MONARCH_RDMA_ALLOW_TCP_FALLBACK=${MONARCH_RDMA_ALLOW_TCP_FALLBACK:-1}

mkdir -p "${DUMP_FOLDER}"

python3 -m torchtitan.experiments.rl.train \
    --module "${MODULE}" \
    --config "${CONFIG}" \
    --async-loop.num-training-steps "${TRAINING_STEPS}" \
    --trainer.lr-scheduler.total-steps "${TRAINING_STEPS}" \
    --dump-folder "${DUMP_FOLDER}" \
    "$@" \
    2>&1 | tee "${DUMP_FOLDER}/train.log"
