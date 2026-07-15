#!/usr/bin/env bash
# Compare BF16 and blockwise FP8 DeepSeek V3 16B loss convergence on 8 GPUs.
#
# Usage:
#   ./scripts/fp8_blockwise/compare_deepseek_v3_16b.sh
#   STEPS=1000 NGPU=8 ./scripts/fp8_blockwise/compare_deepseek_v3_16b.sh
#
# Requires a torchao build with torchao.prototype.blockwise_fp8_training and
# torchao.prototype.moe_training.blockwise_fp8 (set TORCHAO_REPO to a checkout
# to use it via PYTHONPATH).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

TORCHAO_REPO="${TORCHAO_REPO:-}"
NGPU="${NGPU:-8}"
STEPS="${STEPS:-1000}"
SEED="${SEED:-42}"
DUMP_ROOT="${DUMP_ROOT:-./outputs/fp8_blockwise_16b_compare}"
LOG_DIR="${LOG_DIR:-${DUMP_ROOT}/logs}"

if [[ -n "${TORCHAO_REPO}" ]]; then
    export PYTHONPATH="${TORCHAO_REPO}:${PYTHONPATH:-}"
fi

mkdir -p "${LOG_DIR}"

echo "Running BF16 DeepSeek V3 16B on ${NGPU} GPU(s)"
NGPU="${NGPU}" MODULE=deepseek_v3 CONFIG=deepseek_v3_16b ./run_train.sh \
    --training.steps "${STEPS}" \
    --debug.seed "${SEED}" \
    --dump_folder "${DUMP_ROOT}/bf16" "$@" \
    2>&1 | tee "${LOG_DIR}/bf16.log"

echo "Running blockwise FP8 DeepSeek V3 16B on ${NGPU} GPU(s)"
NGPU="${NGPU}" MODULE=deepseek_v3 CONFIG=deepseek_v3_16b_fp8_blockwise ./run_train.sh \
    --training.steps "${STEPS}" \
    --debug.seed "${SEED}" \
    --dump_folder "${DUMP_ROOT}/fp8_blockwise" "$@" \
    2>&1 | tee "${LOG_DIR}/fp8_blockwise.log"

echo "Done. Compare TensorBoard loss curves under ${DUMP_ROOT}/{bf16,fp8_blockwise}."
