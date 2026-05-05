#!/usr/bin/env bash
# Compare BF16 and experimental blockwise FP8 DeepSeek V3 debug-model runs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

TORCHAO_REPO="${TORCHAO_REPO:-/home/dev/ao}"
BATCH_SIZES="${BATCH_SIZES:-1 4 8}"
NGPU="${NGPU:-4}"
STEPS="${STEPS:-10}"
SEED="${SEED:-123}"
DUMP_ROOT="${DUMP_ROOT:-./outputs/fp8_blockwise_compare}"
LOG_DIR="${LOG_DIR:-${DUMP_ROOT}/logs}"

export PYTHONPATH="${TORCHAO_REPO}:${PYTHONPATH:-}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
if [[ "${NCCL_SOCKET_IFNAME}" != *,* && "${NCCL_SOCKET_IFNAME}" != ^* && ! -d "/sys/class/net/${NCCL_SOCKET_IFNAME}" ]]; then
    echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} is not present on this node; using lo"
    NCCL_SOCKET_IFNAME="lo"
fi
export NCCL_SOCKET_IFNAME
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

mkdir -p "${LOG_DIR}"

for batch_size in ${BATCH_SIZES}; do
    bf16_config="deepseek_v3_debugmodel"
    fp8_config="deepseek_v3_debugmodel_fp8_blockwise_bs${batch_size}"
    suffix="chunked"

    bf16_dump="${DUMP_ROOT}/bf16_bs${batch_size}_${suffix}"
    fp8_dump="${DUMP_ROOT}/fp8_blockwise_bs${batch_size}_${suffix}"

    echo "Running BF16 DeepSeek V3 debugmodel bs=${batch_size} on ${NGPU} GPU(s)"
    NGPU="${NGPU}" MODULE=deepseek_v3 CONFIG="${bf16_config}" ./run_train.sh \
        --training.local-batch-size "${batch_size}" \
        --training.steps "${STEPS}" \
        --debug.seed "${SEED}" \
        --dump_folder "${bf16_dump}" "$@" \
        2>&1 | tee "${LOG_DIR}/bf16_bs${batch_size}_${suffix}.log"

    echo "Running blockwise FP8 DeepSeek V3 debugmodel bs=${batch_size} on ${NGPU} GPU(s)"
    NGPU="${NGPU}" MODULE=deepseek_v3 CONFIG="${fp8_config}" ./run_train.sh \
        --training.steps "${STEPS}" \
        --debug.seed "${SEED}" \
        --dump_folder "${fp8_dump}" "$@" \
        2>&1 | tee "${LOG_DIR}/fp8_blockwise_bs${batch_size}_${suffix}.log"
done
