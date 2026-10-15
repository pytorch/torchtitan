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
  core|h100|b200|torchft|transformers-modeling-backend) ;;
  *)
    echo "Unknown validation suite: ${VALIDATION_SUITE}" >&2
    exit 1
    ;;
esac

REPO_ROOT="$(git rev-parse --show-toplevel)"
RC_VERSION="$(tr -d '[:space:]' < "${REPO_ROOT}/assets/version.txt")"
if [[ "${RC_VERSION}" != *rc* ]]; then
  echo "Expected an RC version, got: ${RC_VERSION}"
  exit 1
fi
export RC_VERSION VALIDATION_SUITE TORCH_VERSION TORCHVISION_VERSION TORCHAO_VERSION

if [[ "${VALIDATION_SUITE}" == "b200" ]]; then
  # The B200 CUDA container has no Conda environment.
  uv venv --seed --python 3.12 "${RUNNER_TEMP}/validate-release-b200-venv"
  source "${RUNNER_TEMP}/validate-release-b200-venv/bin/activate"
else
  eval "$(conda shell.bash hook)"
  conda activate "$(conda env list --json | jq -r '.envs[-1]')"
fi

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

if [[ "${VALIDATION_SUITE}" == "b200" ]]; then
  python -m pip install \
    "pytest==7.3.2" "expecttest>=0.2.0" \
    "flash-attn-4[cu13]>=4.0.0b31" \
    "git+https://github.com/meta-pytorch/dist_moe.git@main"
fi

ARTIFACTS="${RUNNER_TEMP}/artifacts-to-be-uploaded"
VALIDATION_ROOT="${RUNNER_TEMP}/validate-release-gpu"
mkdir -p \
  "${ARTIFACTS}" \
  "${VALIDATION_ROOT}/scripts" \
  "${VALIDATION_ROOT}/torchtitan/models/flux/inference"

# Copy test support files without copying TorchTitan Python packages. Training
# subprocesses must import the installed TestPyPI RC rather than the checkout.
cp "${REPO_ROOT}/run_train.sh" "${VALIDATION_ROOT}/run_train.sh"
cp "${REPO_ROOT}/scripts/loss_compare.py" "${VALIDATION_ROOT}/scripts/loss_compare.py"
cp -R "${REPO_ROOT}/tests" "${VALIDATION_ROOT}/tests"
cp "${REPO_ROOT}/pyproject.toml" "${VALIDATION_ROOT}/pyproject.toml"
cp \
  "${REPO_ROOT}/torchtitan/models/flux/inference/prompts.txt" \
  "${VALIDATION_ROOT}/torchtitan/models/flux/inference/prompts.txt"

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
if os.environ["VALIDATION_SUITE"] == "b200":
    capability = torch.cuda.get_device_capability()
    if capability not in ((10, 0), (10, 3)):
        raise ValueError(
            f"B200 validation requires SM100 or SM103, got SM{capability[0]}{capability[1]}"
        )
print(f"torchtitan={torchtitan.__version__} ({torchtitan.__file__})")
print(f"torch={torch.__version__} ({torch.version.git_version})")
print(f"torchvision={torchvision.__version__}")
print(f"torchao={torchao.__version__}")
print(f"triton={triton.__version__}")
print(f"num_cuda_devices={torch.cuda.device_count()}")
PY

run_core_tests() {
  CUDA_VISIBLE_DEVICES=0 python -m pytest tests/unit_tests/gpu \
    -m "not multi_gpu" \
    --strict-markers \
    --durations=20 \
    -vv

  python -m pytest tests/unit_tests/gpu \
    -m multi_gpu \
    --strict-markers \
    --durations=20 \
    -vv

  python -m tests.integration_tests.run_tests \
    --gpu_arch_type cuda \
    --test_suite features,models \
    --execution_mode real_pg \
    --ngpu 8 \
    "${ARTIFACTS}/integration_tests"

  python -m tests.integration_tests.run_tests \
    --gpu_arch_type cuda \
    --test_suite flux \
    --execution_mode real_pg \
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

install_hybrid_ep() {
  export CUDA_HOME=/usr/local/cuda
  export NCCL_NVLS_ENABLE=0
  export TORCH_SHOW_CPP_STACKTRACES=1
  bash /install_deepep.sh
}

run_h100_tests() {
  local status=0

  python -m pip install flash-attn-3 \
    --extra-index-url "${PYTORCH_INDEX_URL}"

  if ! CUDA_HOME=/usr/local/cuda NCCL_NVLS_ENABLE=0 TORCH_SHOW_CPP_STACKTRACES=1 \
    python -m tests.integration_tests.run_tests \
    --test_suite h100 \
    --execution_mode real_pg \
    --exclude qwen3_fsdp+deepep,deepseek_v3_fsdp+hybridep \
    --gpu_arch_type cuda \
    --gpu_arch h100 \
    --ngpu 8 \
    "${ARTIFACTS}/h100/base"; then
    status=1
  fi

  bash "${REPO_ROOT}/.github/scripts/install_deepep_v2.sh"
  if ! CUDA_HOME=/usr/local/cuda NCCL_NVLS_ENABLE=0 EP_DISABLE_GIN=1 \
    python -m pytest \
    tests/unit_tests/gpu/distributed/test_deepep_activation_checkpoint.py -v; then
    status=1
  fi
  if ! CUDA_HOME=/usr/local/cuda NCCL_NVLS_ENABLE=0 EP_DISABLE_GIN=1 \
    TORCH_SHOW_CPP_STACKTRACES=1 python -m tests.integration_tests.run_tests \
    --test_suite h100 \
    --execution_mode real_pg \
    --test_name qwen3_fsdp+deepep \
    --gpu_arch_type cuda \
    --gpu_arch h100 \
    --ngpu 8 \
    "${ARTIFACTS}/h100/deepep_v2"; then
    status=1
  fi

  install_hybrid_ep
  if ! python -m tests.integration_tests.run_tests \
    --test_suite h100 \
    --execution_mode real_pg \
    --test_name deepseek_v3_fsdp+hybridep \
    --gpu_arch_type cuda \
    --gpu_arch h100 \
    --ngpu 8 \
    "${ARTIFACTS}/h100/hybrid_ep"; then
    status=1
  fi

  return "${status}"
}

run_b200_tests() {
  # The shared MXFP8 test file also contains GraphTrainer SimpleFSDP cases.
  NCCL_NVLS_ENABLE=0 python -m pytest \
    tests/unit_tests/cpu/models/common/test_dist_moe.py \
    tests/unit_tests/gpu/models/common/test_dist_moe.py \
    tests/unit_tests/gpu/quantization/test_mxfp8_fsdp.py \
    --deselect 'tests/unit_tests/gpu/quantization/test_mxfp8_fsdp.py::test_mxfp8_fsdp_tensor_lifecycle[simple-fsdp]' \
    --deselect tests/unit_tests/gpu/quantization/test_mxfp8_fsdp.py::test_simple_fsdp_disable_active_parametrization \
    -v

  CUDA_HOME=/usr/local/cuda TORCH_SHOW_CPP_STACKTRACES=1 \
    python -m tests.integration_tests.run_tests \
    "${ARTIFACTS}/b200" \
    --test_suite b200 \
    --execution_mode real_pg \
    --gpu_arch_type cuda \
    --gpu_arch b200 \
    --ngpu 8
}

# Run the selected suite without letting `set -e` abort the script on
# failure: we still need to prune checkpoint directories from the artifacts
# below and report the correct exit code afterwards. The subshell re-enables
# `set -e` so a failing step inside the chosen run_*_tests function still
# fails fast exactly as before; only the outer script's early exit is
# deferred until after cleanup.
suite_status=0
set +e
(
  set -e
  case "${VALIDATION_SUITE}" in
    core) run_core_tests ;;
    h100) run_h100_tests ;;
    b200) run_b200_tests ;;
    torchft) run_torchft_tests ;;
    transformers-modeling-backend) run_transformers_modeling_backend_tests ;;
  esac
)
suite_status=$?
set -e

find "${ARTIFACTS}" -type d -name checkpoint -prune -exec rm -rf {} +

exit "${suite_status}"
