#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Benchmark script: runs FlexShard vs SimpleFSDP back-to-back.
# Outputs TensorBoard logs for comparison.
#
# Usage:
#   ./benchmark_flex_shard.sh [output_dir] [--ngpu N] [--steps N] [--config CONFIG]
#
# Example:
#   ./benchmark_flex_shard.sh /tmp/flex_shard_bench --ngpu 4 --steps 50

set -ex

OUTPUT_DIR=${1:-"/tmp/flex_shard_benchmark"}
shift || true

NGPU=4
STEPS=50
CONFIG="8b"

while [[ $# -gt 0 ]]; do
    case $1 in
        --ngpu) NGPU="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

SIMPLE_FSDP_DIR="${OUTPUT_DIR}/simple_fsdp"
FLEX_SHARD_DIR="${OUTPUT_DIR}/flex_shard"

mkdir -p "${SIMPLE_FSDP_DIR}" "${FLEX_SHARD_DIR}"

echo "=== SimpleFSDP baseline (${CONFIG}) ==="
PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --virtual-local-rank \
    --local-ranks-filter 0 --role rank --tee 3 \
    -m torchtitan.train \
    --module graph_trainer.llama3 \
    --config "graph_trainer_llama3_${CONFIG}" \
    --compile.mode jit \
    --training.steps "${STEPS}" \
    --job.dump_folder "${SIMPLE_FSDP_DIR}"

echo ""
echo "=== FlexShard (${CONFIG}) ==="
PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --virtual-local-rank \
    --local-ranks-filter 0 --role rank --tee 3 \
    -m torchtitan.train \
    --module graph_trainer.flex_shard_llama3 \
    --config "graph_trainer_flex_shard_llama3_${CONFIG}" \
    --compile.mode jit \
    --training.steps "${STEPS}" \
    --job.dump_folder "${FLEX_SHARD_DIR}"

echo ""
echo "=== Done ==="
echo "SimpleFSDP logs: ${SIMPLE_FSDP_DIR}"
echo "FlexShard logs:  ${FLEX_SHARD_DIR}"
echo "Compare with: tensorboard --logdir ${OUTPUT_DIR}"
