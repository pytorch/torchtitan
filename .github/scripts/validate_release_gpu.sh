#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

required_env_vars=(
  VALIDATION_SUITE
  TORCH_VERSION
  TORCHVISION_VERSION
  TORCHAO_VERSION
  PYTORCH_INDEX_URL
)
for var in "${required_env_vars[@]}"; do
  if [[ -z "${!var:-}" ]]; then
    echo "Missing required environment variable: ${var}"
    exit 1
  fi
done

case "${VALIDATION_SUITE}" in
  core|graph-trainer|graph-trainer-h100|h100|torchft|transformers-modeling-backend) ;;
  *)
    echo "Unknown validation suite: ${VALIDATION_SUITE}" >&2
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RC_VERSION="$(tr -d '[:space:]' < "${REPO_ROOT}/assets/version.txt")"
if [[ "${RC_VERSION}" != *rc* ]]; then
  echo "Expected an RC version, got: ${RC_VERSION}"
  exit 1
fi
export RC_VERSION VALIDATION_SUITE TORCH_VERSION TORCHVISION_VERSION TORCHAO_VERSION

eval "$(conda shell.bash hook)"
conda activate "$(conda env list --json | jq -r '.envs[-1]')"

export HF_HOME="${RUNNER_TEMP}/hf_home"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PYTHONUNBUFFERED=1

DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 || true)
echo "CUDA driver version: ${DRIVER_VERSION}"
nvidia-smi

python -m pip uninstall -y torch torchvision torchao torchtitan

USE_CPP=0 PIP_EXTRA_INDEX_URL='' python -m pip install --pre \
  --index-url "${PYTORCH_INDEX_URL}" \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchao==${TORCHAO_VERSION}"

install_release_candidate() {
  for attempt in {1..5}; do
    echo "Installing torchtitan==${RC_VERSION} from TestPyPI (attempt ${attempt}/5)"
    if python -m pip install \
      --index-url https://test.pypi.org/simple/ \
      --extra-index-url https://pypi.org/simple/ \
      "torchtitan==${RC_VERSION}"; then
      return 0
    fi
    if [[ "${attempt}" -lt 5 ]]; then
      sleep 15
    fi
  done

  echo "Failed to install torchtitan==${RC_VERSION} after 5 attempts" >&2
  return 1
}
install_release_candidate

ARTIFACTS="${RUNNER_ARTIFACT_DIR:-${RUNNER_TEMP}/artifacts-to-be-uploaded}"
VALIDATION_ROOT="${RUNNER_TEMP}/validate-release-gpu"

ensure_writable_directory() {
  local directory="$1"
  if mkdir -p "${directory}" 2>/dev/null && [[ -w "${directory}" ]]; then
    return
  fi
  sudo mkdir -p "${directory}"
  sudo chown -R "$(id -u):$(id -g)" "${directory}"
}

ensure_writable_directory "${ARTIFACTS}"
ensure_writable_directory "${HF_HOME}"
ensure_writable_directory "${VALIDATION_ROOT}"

cleanup_artifacts() {
  find "${ARTIFACTS}" -type d -name checkpoint -prune -exec rm -rf {} +
  chmod -R a+rX "${ARTIFACTS}"
}
trap cleanup_artifacts EXIT

mkdir -p \
  "${VALIDATION_ROOT}/scripts" \
  "${VALIDATION_ROOT}/torchtitan/experiments/graph_trainer" \
  "${VALIDATION_ROOT}/torchtitan/models/flux/inference"

# Copy test support files without copying TorchTitan Python packages. Training
# subprocesses must import the installed TestPyPI RC rather than the checkout.
cp "${REPO_ROOT}/run_train.sh" "${VALIDATION_ROOT}/run_train.sh"
cp "${REPO_ROOT}/pyproject.toml" "${VALIDATION_ROOT}/pyproject.toml"
cp "${REPO_ROOT}/scripts/loss_compare.py" "${VALIDATION_ROOT}/scripts/loss_compare.py"
cp -R "${REPO_ROOT}/tests" "${VALIDATION_ROOT}/tests"
cp \
  "${REPO_ROOT}/torchtitan/models/flux/run_infer.sh" \
  "${VALIDATION_ROOT}/torchtitan/models/flux/run_infer.sh"
cp \
  "${REPO_ROOT}/torchtitan/models/flux/inference/prompts.txt" \
  "${VALIDATION_ROOT}/torchtitan/models/flux/inference/prompts.txt"
cp \
  "${REPO_ROOT}/torchtitan/experiments/graph_trainer/run_train_precompile.sh" \
  "${VALIDATION_ROOT}/torchtitan/experiments/graph_trainer/run_train_precompile.sh"

cd "${VALIDATION_ROOT}"
unset PYTHONPATH

python - <<'PY'
import os

import torch
import torchao
import torchtitan
import torchvision
import triton

assert torchtitan.__version__ == os.environ["RC_VERSION"]
assert "site-packages" in torchtitan.__file__
assert torch.__version__.split("+")[0] == os.environ["TORCH_VERSION"]
assert torchvision.__version__.split("+")[0] == os.environ["TORCHVISION_VERSION"]
assert torchao.__version__.split("+")[0] == os.environ["TORCHAO_VERSION"]
assert torch.version.cuda == "13.0"
assert torch.cuda.is_available()
assert torch.cuda.device_count() >= 8
print(f"torchtitan={torchtitan.__version__} ({torchtitan.__file__})")
print(f"torch={torch.__version__} ({torch.version.git_version})")
print(f"torchvision={torchvision.__version__}")
print(f"torchao={torchao.__version__}")
print(f"triton={triton.__version__}")
print(f"num_cuda_devices={torch.cuda.device_count()}")
PY

TORCHTITAN_PACKAGE_ROOT="$(python - <<'PY'
from pathlib import Path

import torchtitan

print(Path(torchtitan.__file__).parent)
PY
)"

run_core_tests() {
  CUDA_VISIBLE_DEVICES=0 python -m pytest \
    tests/unit_tests/flex_shard/test_dist_muon.py \
    --durations=20 \
    -vv

  local baseline_options="--parallelism.data_parallel_replicate_degree=1"
  local test_options="--parallelism.data_parallel_replicate_degree=4"

  python scripts/loss_compare.py . . \
    --baseline-options="${baseline_options}" \
    --test-options="${test_options}" \
    --job-dump-folder="${ARTIFACTS}/llama3_fsdp_hsdp" \
    --assert-equal \
    --steps=1

  python scripts/loss_compare.py . . \
    --baseline-options="${baseline_options}" \
    --job-dump-folder="${ARTIFACTS}/llama3_golden" \
    --import-result=tests/assets/losses/llama3_cuda.txt \
    --assert-equal \
    --steps=100

  python scripts/loss_compare.py . . \
    --baseline-module=qwen3 \
    --baseline-config=qwen3_moe_debug \
    --baseline-options="--parallelism.tensor_parallel_degree 2 --parallelism.expert_parallel_degree 4 --parallelism.spmd_backend spmd_types --training.disable_cuda_graphs" \
    --test-options="--parallelism.tensor_parallel_degree 2 --parallelism.expert_parallel_degree 4 --parallelism.spmd_backend spmd_types --training.disable_cuda_graphs" \
    --job-dump-folder="${ARTIFACTS}/qwen3_moe_golden" \
    --import-result=tests/assets/losses/qwen3_moe_cuda.txt \
    --assert-equal \
    --steps=100

  python -m tests.integration_tests.run_tests \
    --gpu_arch_type cuda \
    --test_suite features \
    --ngpu 8 \
    "${ARTIFACTS}/integration_tests/features"

  python -m tests.integration_tests.run_tests \
    --gpu_arch_type cuda \
    --test_suite models \
    --ngpu 8 \
    "${ARTIFACTS}/integration_tests/models"

  python -m tests.integration_tests.flux \
    --ngpu 8 \
    "${ARTIFACTS}/flux_tests"
}

run_torchft_tests() {
  local lighthouse_pid
  local status=0

  python -m pip install torchft==0.2.0

  RUST_BACKTRACE=1 torchft_lighthouse \
    --min_replicas 1 \
    --quorum_tick_ms 100 \
    --join_timeout_ms 10000 > /dev/null 2>&1 &
  lighthouse_pid=$!

  python -m torchtitan.experiments.torchft.tests.integration_tests \
    "${ARTIFACTS}/torchft" \
    --ngpu 8 || status=$?

  kill "${lighthouse_pid}" 2>/dev/null || true
  wait "${lighthouse_pid}" 2>/dev/null || true
  return "${status}"
}

run_transformers_modeling_backend_tests() {
  python -m pip install transformers==5.9.0
  python -m torchtitan.experiments.transformers_modeling_backend.tests.integration_tests \
    "${ARTIFACTS}/transformers_modeling_backend" \
    --ngpu 8
}

run_graph_trainer_tests() {
  local tests_root="${TORCHTITAN_PACKAGE_ROOT}/experiments/graph_trainer/tests"

  python -m torchtitan.experiments.graph_trainer.tests.integration_tests \
    --test_suite graph_trainer_default \
    --gpu_arch_type cuda \
    "${ARTIFACTS}/graph_trainer" \
    --ngpu 8
  python -m pytest "${tests_root}/test_numerics.py::TestSimpleFSDP" -v
  python -m pytest "${tests_root}/test_numerics.py::TestGraphTrainerNumerics" \
    -v -k dense
  python -m pytest "${tests_root}/test_passes.py" -v
  python -m pytest "${tests_root}/test_profiler.py" -v
  python -m pytest "${tests_root}/test_trace_module.py" -v
  python -m pytest "${tests_root}/test_precompile.py" -v
  python -m torchtitan.experiments.graph_trainer.tests.run_precompile_tests \
    "${ARTIFACTS}/graph_trainer_precompile" \
    --ngpu 8 \
    --test_name aot_fx_trace_llama3_precompile_fsdp_tp
  python -m pytest "${tests_root}/test_bitwise_deterministic.py" -v
  python -m pytest "${tests_root}/test_sac_peak_memory.py" -v
}

install_hybrid_ep() {
  export CUDA_HOME=/usr/local/cuda
  export NCCL_NVLS_ENABLE=0
  export TORCH_SHOW_CPP_STACKTRACES=1
  bash /install_deepep.sh
}

run_h100_tests() {
  python -m pip install flash-attn-3 \
    --extra-index-url "${PYTORCH_INDEX_URL}"

  install_hybrid_ep
  CUDA_HOME=/usr/local/cuda NCCL_NVLS_ENABLE=0 TORCH_SHOW_CPP_STACKTRACES=1 \
    python -m tests.integration_tests.run_tests \
    --test_suite h100 \
    --gpu_arch_type cuda \
    --ngpu 8 \
    "${ARTIFACTS}/h100"
}

run_graph_trainer_h100_tests() {
  local tests_root="${TORCHTITAN_PACKAGE_ROOT}/experiments/graph_trainer/tests"

  install_hybrid_ep
  python -m torchtitan.experiments.graph_trainer.tests.integration_tests \
    --test_suite graph_trainer_h100 \
    --gpu_arch_type cuda \
    "${ARTIFACTS}/graph_trainer_h100" \
    --ngpu 8
  python -m pytest "${tests_root}/test_numerics.py::TestGraphTrainerNumerics" \
    -v -k moe
  python -m torchtitan.experiments.graph_trainer.tests.run_precompile_tests \
    "${ARTIFACTS}/graph_trainer_h100_precompile" \
    --ngpu 8 \
    --test_name aot_fx_trace_deepseek_v3_precompile_fsdp_tp_ep
  python -m pytest "${tests_root}/test_bitwise_deterministic.py" -v
}

case "${VALIDATION_SUITE}" in
  core) run_core_tests ;;
  graph-trainer) run_graph_trainer_tests ;;
  graph-trainer-h100) run_graph_trainer_h100_tests ;;
  h100) run_h100_tests ;;
  torchft) run_torchft_tests ;;
  transformers-modeling-backend) run_transformers_modeling_backend_tests ;;
esac
