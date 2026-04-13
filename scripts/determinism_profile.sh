#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Benchmark and profile deterministic vs non-deterministic training in TorchTitan.
#
# Runs training twice (non-deterministic, then deterministic), compares throughput/
# MFU/loss metrics, and captures PyTorch profiler traces for kernel-level analysis
# of which CUDA kernels are slower under deterministic mode.
#
# Usage:
#   # Quick run with debugmodel
#   ./scripts/determinism_profile.sh
#
#   # gpt_oss_20b MoE on 8 GPUs
#   MODULE=gpt_oss CONFIG=gpt_oss_20b NGPU=8 ./scripts/determinism_profile.sh
#
#   # Custom steps and profile window
#   MODULE=gpt_oss CONFIG=gpt_oss_20b STEPS=30 PROFILE_STEP=20 ./scripts/determinism_profile.sh
#
#   # With memory snapshots
#   ENABLE_MEMORY_SNAPSHOT=1 MODULE=gpt_oss CONFIG=gpt_oss_20b ./scripts/determinism_profile.sh
#
# Environment variables:
#   MODULE               Model module (default: gpt_oss)
#   CONFIG               Config name (default: gpt_oss_debugmodel)
#   NGPU                 Number of GPUs (default: 8)
#   STEPS                Training steps (default: 20)
#   WARMUP_STEPS         Steps to skip in metric averages (default: 3)
#   SEED                 Random seed (default: 42)
#   OUTPUT_DIR           Output directory (default: determinism_profile_output)
#   PROFILE_STEP         Step at which to capture profile (default: 10)
#   PROFILER_WARMUP      Profiler warmup iterations (default: 3)
#   PROFILER_ACTIVE      Profiler active iterations (default: 1)
#   ENABLE_MEMORY_SNAPSHOT  Enable GPU memory snapshots (0 or 1, default: 0)
#   EXTRA_OPTS           Extra options passed to run_train.sh (default: "")

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- Configuration via environment variables ---
MODULE="${MODULE:-gpt_oss}"
CONFIG="${CONFIG:-gpt_oss_debugmodel}"
NGPU="${NGPU:-8}"
STEPS="${STEPS:-20}"
WARMUP_STEPS="${WARMUP_STEPS:-3}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-determinism_profile_output}"
PROFILE_STEP="${PROFILE_STEP:-10}"
PROFILER_WARMUP="${PROFILER_WARMUP:-3}"
PROFILER_ACTIVE="${PROFILER_ACTIVE:-1}"
ENABLE_MEMORY_SNAPSHOT="${ENABLE_MEMORY_SNAPSHOT:-0}"
EXTRA_OPTS="${EXTRA_OPTS:-}"

# --- Validation ---
if [ -d "${OUTPUT_DIR}" ]; then
    echo "WARNING: Output directory '${OUTPUT_DIR}' already exists. Removing it."
    rm -rf "${OUTPUT_DIR}"
fi

mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "  TorchTitan Determinism Benchmark & Profiling"
echo "============================================================"
echo "  MODULE:           ${MODULE}"
echo "  CONFIG:           ${CONFIG}"
echo "  NGPU:             ${NGPU}"
echo "  STEPS:            ${STEPS}"
echo "  WARMUP_STEPS:     ${WARMUP_STEPS}"
echo "  SEED:             ${SEED}"
echo "  OUTPUT_DIR:       ${OUTPUT_DIR}"
echo "  PROFILE_STEP:     ${PROFILE_STEP}"
echo "  PROFILER_WARMUP:  ${PROFILER_WARMUP}"
echo "  PROFILER_ACTIVE:  ${PROFILER_ACTIVE}"
echo "  MEM_SNAPSHOT:     ${ENABLE_MEMORY_SNAPSHOT}"
echo "============================================================"

# --- Build common training arguments ---
COMMON_OPTS="--training.steps=${STEPS} --metrics.log_freq=1 --debug.seed=${SEED}"
COMMON_OPTS="${COMMON_OPTS} --profiling.enable_profiling"
COMMON_OPTS="${COMMON_OPTS} --profiling.profile_freq=${PROFILE_STEP}"

if [ "${ENABLE_MEMORY_SNAPSHOT}" = "1" ]; then
    COMMON_OPTS="${COMMON_OPTS} --profiling.enable_memory_snapshot"
fi

if [ -n "${EXTRA_OPTS}" ]; then
    COMMON_OPTS="${COMMON_OPTS} ${EXTRA_OPTS}"
fi

# --- Run 1: Non-deterministic with profiling ---
echo ""
echo ">>> Run 1/2: Non-deterministic training (profiled)"
echo ""

NONDET_LOG="${OUTPUT_DIR}/nondet_training.log"
NONDET_DUMP="${OUTPUT_DIR}/nondet"
mkdir -p "${NONDET_DUMP}"

NONDET_OPTS="${COMMON_OPTS} --dump_folder=${NONDET_DUMP} --profiling.save_traces_folder=profile_traces"

(
    cd "${REPO_DIR}"
    MODULE="${MODULE}" CONFIG="${CONFIG}" NGPU="${NGPU}" \
        ./run_train.sh ${NONDET_OPTS} 2>&1 | tee "${NONDET_LOG}"
)
echo ">>> Non-deterministic profiled run complete. Log: ${NONDET_LOG}"

# --- Run 2: Deterministic with profiling ---
echo ""
echo ">>> Run 2/2: Deterministic training (profiled)"
echo ""

DET_LOG="${OUTPUT_DIR}/det_training.log"
DET_DUMP="${OUTPUT_DIR}/det"
mkdir -p "${DET_DUMP}"

DET_OPTS="${COMMON_OPTS} --debug.deterministic --dump_folder=${DET_DUMP} --profiling.save_traces_folder=profile_traces"

(
    cd "${REPO_DIR}"
    MODULE="${MODULE}" CONFIG="${CONFIG}" NGPU="${NGPU}" \
        ./run_train.sh ${DET_OPTS} 2>&1 | tee "${DET_LOG}"
)
echo ">>> Deterministic profiled run complete. Log: ${DET_LOG}"

# --- Locate rank-0 trace files ---
echo ""
echo ">>> Locating rank-0 trace files..."

NONDET_TRACE=$(find "${NONDET_DUMP}/profile_traces" -name "rank0_trace.json" 2>/dev/null | head -1 || true)
DET_TRACE=$(find "${DET_DUMP}/profile_traces" -name "rank0_trace.json" 2>/dev/null | head -1 || true)

if [ -z "${NONDET_TRACE}" ]; then
    echo "WARNING: Could not find rank0_trace.json under ${NONDET_DUMP}/profile_traces"
    echo "  Available files:"
    find "${NONDET_DUMP}" -type f -name "*.json" 2>/dev/null | head -10 || echo "  (none)"
fi

if [ -z "${DET_TRACE}" ]; then
    echo "WARNING: Could not find rank0_trace.json under ${DET_DUMP}/profile_traces"
    echo "  Available files:"
    find "${DET_DUMP}" -type f -name "*.json" 2>/dev/null | head -10 || echo "  (none)"
fi

# --- Analysis ---
echo ""
echo "============================================================"
echo "  Analyzing results..."
echo "============================================================"
echo ""

ANALYZE_ARGS=(
    --nondet-log "${NONDET_LOG}"
    --det-log "${DET_LOG}"
    --warmup-steps "${WARMUP_STEPS}"
    --output "${OUTPUT_DIR}/profile_comparison_report.txt"
)

if [ -n "${NONDET_TRACE}" ] && [ -n "${DET_TRACE}" ]; then
    ANALYZE_ARGS+=(--nondet-trace "${NONDET_TRACE}" --det-trace "${DET_TRACE}")
    echo "  Trace files:"
    echo "    Non-det: ${NONDET_TRACE}"
    echo "    Det:     ${DET_TRACE}"
else
    echo "  WARNING: Skipping kernel diff (missing trace files)."
fi

python3 "${SCRIPT_DIR}/determinism_analyze.py" "${ANALYZE_ARGS[@]}"

echo ""
echo ">>> Profiling benchmark complete. Results in: ${OUTPUT_DIR}/"
