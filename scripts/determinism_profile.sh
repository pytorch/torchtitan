#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# =============================================================================
# Kernel-Level Profiling: Deterministic vs Non-Deterministic
# =============================================================================
#
# This script captures PyTorch profiler traces for a single profiling window
# in both deterministic and non-deterministic modes, then runs kernel-level
# diff analysis to identify the specific CUDA kernels responsible for
# deterministic mode slowdowns.
#
# Unlike determinism_benchmark.sh (which focuses on aggregate throughput),
# this script focuses on kernel-level granularity for root-cause analysis.
#
# Usage:
#   # Profile gpt_oss_20b MoE (default: 15 steps, profile at step 10)
#   MODULE=gpt_oss CONFIG=gpt_oss_20b ./scripts/determinism_profile.sh
#
#   # Profile debugmodel for quick validation
#   ./scripts/determinism_profile.sh
#
#   # Custom profile window
#   MODULE=gpt_oss CONFIG=gpt_oss_20b PROFILE_STEP=20 STEPS=25 \
#     ./scripts/determinism_profile.sh
#
#   # With memory snapshots
#   ENABLE_MEMORY_SNAPSHOT=1 MODULE=gpt_oss CONFIG=gpt_oss_20b \
#     ./scripts/determinism_profile.sh
#
# Output:
#   $OUTPUT_DIR/
#     nondet/
#       profile_traces/iteration_$PROFILE_STEP/rank0_trace.json
#     det/
#       profile_traces/iteration_$PROFILE_STEP/rank0_trace.json
#     kernel_diff.txt    — top kernels sorted by absolute slowdown
#
# The Chrome traces can be opened in chrome://tracing or https://ui.perfetto.dev/
# for visual side-by-side comparison.
# =============================================================================

set -euo pipefail

# --------------- Configuration ---------------
MODULE=${MODULE:-"gpt_oss"}
CONFIG=${CONFIG:-"gpt_oss_debugmodel"}
NGPU=${NGPU:-"8"}
STEPS=${STEPS:-"15"}
SEED=${SEED:-"42"}
OUTPUT_DIR=${OUTPUT_DIR:-"determinism_profile_output"}
EXTRA_OPTS=${EXTRA_OPTS:-""}

# Profiling parameters
# Profile window: warmup=3, active=1 at PROFILE_STEP
# profile_freq = PROFILE_STEP (profile at that step)
PROFILE_STEP=${PROFILE_STEP:-"10"}
PROFILER_WARMUP=${PROFILER_WARMUP:-"3"}
PROFILER_ACTIVE=${PROFILER_ACTIVE:-"1"}
ENABLE_MEMORY_SNAPSHOT=${ENABLE_MEMORY_SNAPSHOT:-"0"}

# Derived
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# --------------- Pre-flight ---------------
echo "============================================================"
echo " Kernel-Level Determinism Profiling"
echo "============================================================"
echo "Module:            ${MODULE}"
echo "Config:            ${CONFIG}"
echo "GPUs:              ${NGPU}"
echo "Steps:             ${STEPS}"
echo "Profile at step:   ${PROFILE_STEP}"
echo "Profiler warmup:   ${PROFILER_WARMUP}"
echo "Profiler active:   ${PROFILER_ACTIVE}"
echo "Output dir:        ${OUTPUT_DIR}"
echo "============================================================"
echo ""

if [ -d "${OUTPUT_DIR}" ]; then
    echo "ERROR: Output directory '${OUTPUT_DIR}' already exists."
    echo "       Remove it first: rm -rf ${OUTPUT_DIR}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}/nondet" "${OUTPUT_DIR}/det"
cd "${SCRIPT_DIR}"

# --------------- Common profiling options ---------------
PROFILING_OPTS="--profiling.enable_profiling"
PROFILING_OPTS="${PROFILING_OPTS} --profiling.profile_freq=${PROFILE_STEP}"
PROFILING_OPTS="${PROFILING_OPTS} --profiling.profiler_warmup=${PROFILER_WARMUP}"
PROFILING_OPTS="${PROFILING_OPTS} --profiling.profiler_active=${PROFILER_ACTIVE}"

if [ "${ENABLE_MEMORY_SNAPSHOT}" = "1" ]; then
    PROFILING_OPTS="${PROFILING_OPTS} --profiling.enable_memory_snapshot"
fi

COMMON_OPTS="--debug.seed=${SEED} --training.steps=${STEPS} --metrics.log_freq=1"
if [ -n "${EXTRA_OPTS}" ]; then
    COMMON_OPTS="${COMMON_OPTS} ${EXTRA_OPTS}"
fi

# --------------- Run 1: Non-Deterministic with profiling ---------------
echo ""
echo "============================================================"
echo " Run 1/2: Non-Deterministic (profiled)"
echo "============================================================"
NONDET_LOG="${OUTPUT_DIR}/nondet_training.log"

PYTORCH_ALLOC_CONF="expandable_segments:True" \
PYTHONUNBUFFERED=1 \
NGPU=${NGPU} MODULE=${MODULE} CONFIG=${CONFIG} \
  ./run_train.sh \
    ${COMMON_OPTS} \
    ${PROFILING_OPTS} \
    --dump_folder="${OUTPUT_DIR}/nondet" \
  2>&1 | tee "${NONDET_LOG}"

echo ""
echo "Non-deterministic profiling run completed."

# --------------- Run 2: Deterministic with profiling ---------------
echo ""
echo "============================================================"
echo " Run 2/2: Deterministic (profiled)"
echo "============================================================"
DET_LOG="${OUTPUT_DIR}/det_training.log"

PYTORCH_ALLOC_CONF="expandable_segments:True" \
PYTHONUNBUFFERED=1 \
NGPU=${NGPU} MODULE=${MODULE} CONFIG=${CONFIG} \
  ./run_train.sh \
    ${COMMON_OPTS} \
    ${PROFILING_OPTS} \
    --debug.deterministic \
    --dump_folder="${OUTPUT_DIR}/det" \
  2>&1 | tee "${DET_LOG}"

echo ""
echo "Deterministic profiling run completed."

# --------------- Kernel diff analysis ---------------
echo ""
echo "============================================================"
echo " Kernel Diff Analysis"
echo "============================================================"

# Find the trace files
NONDET_TRACE=$(find "${OUTPUT_DIR}/nondet/profile_traces" -name "rank0_trace.json" 2>/dev/null | head -1)
DET_TRACE=$(find "${OUTPUT_DIR}/det/profile_traces" -name "rank0_trace.json" 2>/dev/null | head -1)

if [ -z "${NONDET_TRACE}" ] || [ -z "${DET_TRACE}" ]; then
    echo "WARNING: Could not find profiler trace files."
    echo "  Non-det trace: ${NONDET_TRACE:-NOT FOUND}"
    echo "  Det trace:     ${DET_TRACE:-NOT FOUND}"
    echo ""
    echo "Skipping kernel diff. You can run it manually once traces are available:"
    echo "  python scripts/determinism_analyze.py \\"
    echo "    --nondet-log ${NONDET_LOG} --det-log ${DET_LOG} \\"
    echo "    --nondet-trace <nondet_trace.json> --det-trace <det_trace.json> \\"
    echo "    --output ${OUTPUT_DIR}/kernel_diff.txt"
else
    echo "Non-det trace: ${NONDET_TRACE}"
    echo "Det trace:     ${DET_TRACE}"
    echo ""

    python3 "${SCRIPT_DIR}/scripts/determinism_analyze.py" \
        --nondet-log "${NONDET_LOG}" \
        --det-log "${DET_LOG}" \
        --nondet-trace "${NONDET_TRACE}" \
        --det-trace "${DET_TRACE}" \
        --top-kernels 50 \
        --output "${OUTPUT_DIR}/kernel_diff.txt"
fi

echo ""
echo "============================================================"
echo " Profiling Complete"
echo "============================================================"
echo "Output directory:  ${OUTPUT_DIR}/"
echo "  nondet_training.log       — non-deterministic log"
echo "  det_training.log          — deterministic log"
echo "  nondet/profile_traces/    — non-deterministic Chrome traces"
echo "  det/profile_traces/       — deterministic Chrome traces"
echo "  kernel_diff.txt           — kernel slowdown analysis"
echo ""
echo "To visually compare traces:"
echo "  1. Open chrome://tracing or https://ui.perfetto.dev/"
echo "  2. Load ${NONDET_TRACE:-nondet trace}"
echo "  3. Load ${DET_TRACE:-det trace}"
echo "  4. Compare kernel durations side-by-side"
echo ""
