#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# =============================================================================
# Deterministic vs Non-Deterministic Benchmark for TorchTitan
# =============================================================================
#
# This script runs the same training configuration twice — once with default
# (non-deterministic) settings and once with full determinism enabled — then
# compares throughput (tokens/sec), TFLOPs, and MFU from the training logs.
#
# It also optionally captures PyTorch profiler traces for kernel-level analysis
# of which CUDA kernels are slower under deterministic mode.
#
# Usage:
#   # Quick benchmark with gpt_oss debugmodel (fast, for validation)
#   ./scripts/determinism_benchmark.sh
#
#   # Benchmark gpt_oss_20b MoE (production-like)
#   MODULE=gpt_oss CONFIG=gpt_oss_20b ./scripts/determinism_benchmark.sh
#
#   # Benchmark with profiling enabled (captures Chrome traces)
#   ENABLE_PROFILING=1 MODULE=gpt_oss CONFIG=gpt_oss_20b ./scripts/determinism_benchmark.sh
#
#   # Custom steps, GPUs, and output directory
#   MODULE=gpt_oss CONFIG=gpt_oss_20b STEPS=50 NGPU=8 OUTPUT_DIR=my_benchmark \
#     ./scripts/determinism_benchmark.sh
#
#   # Skip warmup steps in throughput calculation
#   WARMUP_STEPS=5 MODULE=gpt_oss CONFIG=gpt_oss_20b ./scripts/determinism_benchmark.sh
#
#   # With memory snapshots for memory analysis
#   ENABLE_MEMORY_SNAPSHOT=1 MODULE=gpt_oss CONFIG=gpt_oss_20b ./scripts/determinism_benchmark.sh
#
#   # Additional CLI overrides (e.g., parallelism settings for multi-node)
#   MODULE=gpt_oss CONFIG=gpt_oss_20b EXTRA_OPTS="--parallelism.expert_parallel_degree 4" \
#     ./scripts/determinism_benchmark.sh
#
# Environment Variables:
#   MODULE              Model module (default: gpt_oss)
#   CONFIG              Config name (default: gpt_oss_debugmodel)
#   NGPU                Number of GPUs (default: 8)
#   STEPS               Training steps per run (default: 20)
#   WARMUP_STEPS        Steps to skip for throughput avg (default: 3)
#   SEED                Random seed for reproducibility (default: 42)
#   OUTPUT_DIR           Output directory (default: determinism_benchmark_output)
#   ENABLE_PROFILING    Set to 1 to capture profiler traces (default: 0)
#   PROFILE_FREQ        Profiler frequency in steps (default: 5)
#   ENABLE_MEMORY_SNAPSHOT  Set to 1 to capture memory snapshots (default: 0)
#   EXTRA_OPTS          Additional CLI flags passed to both runs
#
# Output:
#   $OUTPUT_DIR/
#     nondet_training.log    — full training log (non-deterministic)
#     det_training.log       — full training log (deterministic)
#     profile_traces/        — Chrome traces (if profiling enabled)
#       nondet/              — traces from non-deterministic run
#       det/                 — traces from deterministic run
#     memory_snapshot/       — memory snapshots (if enabled)
#       nondet/
#       det/
#     benchmark_summary.txt  — side-by-side comparison of metrics
# =============================================================================

set -euo pipefail

# --------------- Configuration ---------------
MODULE=${MODULE:-"gpt_oss"}
CONFIG=${CONFIG:-"gpt_oss_debugmodel"}
NGPU=${NGPU:-"8"}
STEPS=${STEPS:-"20"}
WARMUP_STEPS=${WARMUP_STEPS:-"3"}
SEED=${SEED:-"42"}
OUTPUT_DIR=${OUTPUT_DIR:-"determinism_benchmark_output"}
ENABLE_PROFILING=${ENABLE_PROFILING:-"0"}
PROFILE_FREQ=${PROFILE_FREQ:-"5"}
ENABLE_MEMORY_SNAPSHOT=${ENABLE_MEMORY_SNAPSHOT:-"0"}
EXTRA_OPTS=${EXTRA_OPTS:-""}

# Derived
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SUMMARY_FILE="${OUTPUT_DIR}/benchmark_summary.txt"

# --------------- Pre-flight checks ---------------
echo "============================================================"
echo " Deterministic vs Non-Deterministic Benchmark"
echo "============================================================"
echo "Module:           ${MODULE}"
echo "Config:           ${CONFIG}"
echo "GPUs:             ${NGPU}"
echo "Steps:            ${STEPS}"
echo "Warmup steps:     ${WARMUP_STEPS}"
echo "Seed:             ${SEED}"
echo "Output dir:       ${OUTPUT_DIR}"
echo "Profiling:        ${ENABLE_PROFILING}"
echo "Memory snapshot:  ${ENABLE_MEMORY_SNAPSHOT}"
echo "Extra options:    ${EXTRA_OPTS}"
echo "============================================================"
echo ""

if [ -d "${OUTPUT_DIR}" ]; then
    echo "ERROR: Output directory '${OUTPUT_DIR}' already exists."
    echo "       Remove it first: rm -rf ${OUTPUT_DIR}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

cd "${SCRIPT_DIR}"

# --------------- Build commands ---------------
# Common options for both runs
COMMON_OPTS="--debug.seed=${SEED} --training.steps=${STEPS} --metrics.log_freq=1"
COMMON_OPTS="${COMMON_OPTS} --metrics.enable_tensorboard --dump_folder=${OUTPUT_DIR}"

if [ "${ENABLE_PROFILING}" = "1" ]; then
    COMMON_OPTS="${COMMON_OPTS} --profiling.enable_profiling --profiling.profile_freq=${PROFILE_FREQ}"
fi

if [ "${ENABLE_MEMORY_SNAPSHOT}" = "1" ]; then
    COMMON_OPTS="${COMMON_OPTS} --profiling.enable_memory_snapshot"
fi

if [ -n "${EXTRA_OPTS}" ]; then
    COMMON_OPTS="${COMMON_OPTS} ${EXTRA_OPTS}"
fi

# Non-deterministic run: just the seed, no deterministic flags
NONDET_OPTS="${COMMON_OPTS}"
NONDET_OPTS="${NONDET_OPTS} --metrics.save_tb_folder=tb_nondet"
if [ "${ENABLE_PROFILING}" = "1" ]; then
    NONDET_OPTS="${NONDET_OPTS} --profiling.save_traces_folder=profile_traces/nondet"
fi
if [ "${ENABLE_MEMORY_SNAPSHOT}" = "1" ]; then
    NONDET_OPTS="${NONDET_OPTS} --profiling.save_memory_snapshot_folder=memory_snapshot/nondet"
fi

# Deterministic run: enable deterministic mode
DET_OPTS="${COMMON_OPTS} --debug.deterministic"
DET_OPTS="${DET_OPTS} --metrics.save_tb_folder=tb_det"
if [ "${ENABLE_PROFILING}" = "1" ]; then
    DET_OPTS="${DET_OPTS} --profiling.save_traces_folder=profile_traces/det"
fi
if [ "${ENABLE_MEMORY_SNAPSHOT}" = "1" ]; then
    DET_OPTS="${DET_OPTS} --profiling.save_memory_snapshot_folder=memory_snapshot/det"
fi

# --------------- Run 1: Non-Deterministic ---------------
echo ""
echo "============================================================"
echo " Run 1/2: Non-Deterministic"
echo "============================================================"
NONDET_LOG="${OUTPUT_DIR}/nondet_training.log"
echo "Command: MODULE=${MODULE} CONFIG=${CONFIG} NGPU=${NGPU} ./run_train.sh ${NONDET_OPTS}"
echo "Logging to: ${NONDET_LOG}"
echo ""

PYTORCH_ALLOC_CONF="expandable_segments:True" \
PYTHONUNBUFFERED=1 \
NGPU=${NGPU} MODULE=${MODULE} CONFIG=${CONFIG} \
  ./run_train.sh ${NONDET_OPTS} 2>&1 | tee "${NONDET_LOG}"

echo ""
echo "Non-deterministic run completed."
echo ""

# --------------- Run 2: Deterministic ---------------
echo "============================================================"
echo " Run 2/2: Deterministic"
echo "============================================================"
DET_LOG="${OUTPUT_DIR}/det_training.log"
echo "Command: MODULE=${MODULE} CONFIG=${CONFIG} NGPU=${NGPU} ./run_train.sh ${DET_OPTS}"
echo "Logging to: ${DET_LOG}"
echo ""

PYTORCH_ALLOC_CONF="expandable_segments:True" \
PYTHONUNBUFFERED=1 \
NGPU=${NGPU} MODULE=${MODULE} CONFIG=${CONFIG} \
  ./run_train.sh ${DET_OPTS} 2>&1 | tee "${DET_LOG}"

echo ""
echo "Deterministic run completed."
echo ""

# --------------- Extract metrics and summarize ---------------
echo "============================================================"
echo " Extracting Metrics & Summary"
echo "============================================================"

python3 "${SCRIPT_DIR}/scripts/determinism_analyze.py" \
    --nondet-log "${NONDET_LOG}" \
    --det-log "${DET_LOG}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --output "${SUMMARY_FILE}"

echo ""
echo "============================================================"
echo " Benchmark Complete"
echo "============================================================"
echo "Results:     ${SUMMARY_FILE}"
echo "Logs:        ${NONDET_LOG}, ${DET_LOG}"
if [ "${ENABLE_PROFILING}" = "1" ]; then
    echo "Traces:      ${OUTPUT_DIR}/profile_traces/{nondet,det}/"
    echo ""
    echo "To compare traces, open them side-by-side in chrome://tracing"
    echo "or use https://ui.perfetto.dev/ for a richer diff view."
fi
if [ "${ENABLE_MEMORY_SNAPSHOT}" = "1" ]; then
    echo "Mem snapshots: ${OUTPUT_DIR}/memory_snapshot/{nondet,det}/"
    echo "Visualize at:  https://pytorch.org/memory_viz"
fi
echo ""
