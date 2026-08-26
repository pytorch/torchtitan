#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
proof_root="${repo_root}/outputs/kimi_k3_full_minimal_async_ep_fake"
benchmark_steps="${BENCHMARK_STEPS:-20}"
sequence_length="${SEQ_LEN:-4096}"
profile_warmup_steps="${PROFILE_WARMUP_STEPS:-2}"
profile_active_steps="${PROFILE_ACTIVE_STEPS:-1}"
run_id="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
run_dir="${proof_root}/benchmark_runs/${run_id}"

if [[ ! "${benchmark_steps}" =~ ^[0-9]+$ ]]; then
    echo "BENCHMARK_STEPS must be an integer, got ${benchmark_steps}" >&2
    exit 1
fi

if ((benchmark_steps < 10)); then
    echo "BENCHMARK_STEPS must be at least 10, got ${benchmark_steps}" >&2
    exit 1
fi

if [[ ! "${sequence_length}" =~ ^[1-9][0-9]*$ ]]; then
    echo "SEQ_LEN must be a positive integer" >&2
    exit 1
fi

if [[ ! "${profile_warmup_steps}" =~ ^[0-9]+$ ]]; then
    echo "PROFILE_WARMUP_STEPS must be a non-negative integer" >&2
    exit 1
fi

if [[ ! "${profile_active_steps}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PROFILE_ACTIVE_STEPS must be a positive integer" >&2
    exit 1
fi

if ((profile_warmup_steps + profile_active_steps > benchmark_steps)); then
    echo "Profiler warmup and active steps must fit within BENCHMARK_STEPS" >&2
    exit 1
fi

global_tokens=$((sequence_length * 256))

mkdir -p "${run_dir}"
cd "${repo_root}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export NGPU=256
export LOG_RANK=0
export COMM_MODE=fake_backend
export MODULE=kimi_k3_fake_profile_config
export CONFIG=kimi_k3_full_minimal_async_ep_fake_profile
export PYTHONPATH="${proof_root}:${repo_root}:${PYTHONPATH:-}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

echo "Benchmark output: ${run_dir}"
echo "Configuration: Kimi-K3 BF16, fake world size 256, FSDP256, EP64"
echo "Batch: 1 sequence x ${sequence_length} tokens per rank (${global_tokens} global tokens)"
echo "Steps: ${benchmark_steps}; final profiler window: ${profile_warmup_steps} warmup + ${profile_active_steps} active"

./run_train.sh \
    --dump-folder "${run_dir}" \
    --training.steps "${benchmark_steps}" \
    --training.num-tokens-per-microbatch-per-dp-rank "${sequence_length}" \
    --training.num-tokens-per-train-step "${global_tokens}" \
    --training.max-context-length "${sequence_length}" \
    --profiler.enable-profiling \
    --profiler.save-traces-folder gpu_trace \
    --profiler.profile-freq "${benchmark_steps}" \
    --profiler.profiler-warmup "${profile_warmup_steps}" \
    --profiler.profiler-active "${profile_active_steps}" \
    2>&1 | tee "${run_dir}/console.log"
