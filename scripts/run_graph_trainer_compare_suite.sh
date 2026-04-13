#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/outputs/graph_trainer_compare/${TIMESTAMP}}"
MASTER_LOG="${LOG_ROOT}/run.log"

mkdir -p "${LOG_ROOT}"

log() {
  local message="$1"
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${message}" | tee -a "${MASTER_LOG}"
}

run_case() {
  local name="$1"
  local description="$2"
  shift 2
  local log_file="${LOG_ROOT}/${name}.log"

  log "START ${name}: ${description}"
  log "COMMAND ${name}: $*"

  if "$@" 2>&1 | tee "${log_file}"; then
    log "PASS ${name}: log=${log_file}"
    return 0
  fi

  log "FAIL ${name}: log=${log_file}"
  return 1
}

failures=0

log "Graph trainer compare suite logs will be written to ${LOG_ROOT}"
log "DeepSeek V3 cases require H100 GPUs. Pytest will skip them if the machine does not satisfy the requirement."

run_case \
  "llama3_aot_vs_aot_fx_trace" \
  "Llama3 debug model: compare aot vs aot_fx_trace. Prints losses, first-step time, first-step throughput, and peak memory." \
  python -m pytest -s torchtitan/experiments/graph_trainer/tests/test_compile_mode_compare.py::test_llama3_aot_vs_aot_fx_trace -v \
  || failures=$((failures + 1))

run_case \
  "deepseek_v3_aot_vs_aot_fx_trace" \
  "DeepSeek V3 debug model: compare aot vs aot_fx_trace. Prints losses, first-step time, first-step throughput, and peak memory." \
  python -m pytest -s torchtitan/experiments/graph_trainer/tests/test_compile_mode_compare.py::test_deepseek_v3_aot_vs_aot_fx_trace -v \
  || failures=$((failures + 1))

run_case \
  "llama3_regional_aot_vs_aot_fx_trace" \
  "Llama3 flex-attention debug model: compare aot+regional_inductor vs aot_fx_trace+regional_inductor. Prints losses, first-step time, first-step throughput, and peak memory." \
  python -m pytest -s torchtitan/experiments/graph_trainer/tests/test_compile_mode_compare_regional.py::test_llama3_regional_aot_vs_aot_fx_trace -v \
  || failures=$((failures + 1))

run_case \
  "deepseek_v3_regional_aot_vs_aot_fx_trace" \
  "DeepSeek V3 flex-attention debug model: compare aot+regional_inductor vs aot_fx_trace+regional_inductor. Prints losses, first-step time, first-step throughput, and peak memory." \
  python -m pytest -s torchtitan/experiments/graph_trainer/tests/test_compile_mode_compare_regional.py::test_deepseek_v3_regional_aot_vs_aot_fx_trace -v \
  || failures=$((failures + 1))

if [[ "${failures}" -ne 0 ]]; then
  log "DONE with ${failures} failing command(s)"
  exit 1
fi

log "DONE all compare commands passed"
