#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
proof_root="${repo_root}/outputs/dsv3_16b_minimal_async_ep_fake"
run_dir="${proof_root}/run"

mkdir -p "${run_dir}"
cd "${repo_root}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NGPU=8
export LOG_RANK=0
export COMM_MODE=fake_backend
export MODULE=dsv3_16b_fake_profile_config
export CONFIG=dsv3_16b_minimal_async_ep_fake_profile
export PYTHONPATH="${proof_root}:${repo_root}:${PYTHONPATH:-}"

./run_train.sh \
    --dump-folder "${run_dir}" \
    --training.steps 2 \
    --profiler.enable-profiling \
    --profiler.save-traces-folder gpu_trace \
    --profiler.profile-freq 2 \
    --profiler.profiler-warmup 1 \
    --profiler.profiler-active 1 \
    --debug.print-config
