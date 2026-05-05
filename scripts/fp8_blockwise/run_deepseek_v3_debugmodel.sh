#!/usr/bin/env bash
# Run DeepSeek V3 debug model with experimental torchao blockwise FP8 linears.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

TORCHAO_REPO="${TORCHAO_REPO:-/home/dev/ao}"
BATCH_SIZES="${BATCH_SIZES:-1 4 8}"
NGPU="${NGPU:-4}"
DUMP_ROOT="${DUMP_ROOT:-./outputs/fp8_blockwise_deepseek_v3_debugmodel}"

export PYTHONPATH="${TORCHAO_REPO}:${PYTHONPATH:-}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
if [[ "${NCCL_SOCKET_IFNAME}" != *,* && "${NCCL_SOCKET_IFNAME}" != ^* && ! -d "/sys/class/net/${NCCL_SOCKET_IFNAME}" ]]; then
    echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} is not present on this node; using lo"
    NCCL_SOCKET_IFNAME="lo"
fi
export NCCL_SOCKET_IFNAME
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

for batch_size in ${BATCH_SIZES}; do
    CONFIG="deepseek_v3_debugmodel_fp8_blockwise_bs${batch_size}"
    dump_folder="${DUMP_ROOT}/bs${batch_size}"

    echo "Running ${CONFIG} on ${NGPU} GPU(s); dump_folder=${dump_folder}"
    NGPU="${NGPU}" MODULE=deepseek_v3 CONFIG="${CONFIG}" ./run_train.sh \
        --dump_folder "${dump_folder}" \
        "$@"
done
