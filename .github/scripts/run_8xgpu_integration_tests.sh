#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Shared setup for the H100 integration suite.
#
# The calling workflow passes the matrix values as env vars:
#   INDEX_URL      torch/torchao --index-url (required)
#   GPU_ARCH_TYPE  "cuda" or "rocm" (required)
#   TORCH_VERSION  torch version pin, empty for the latest nightly (optional)

set -eux

: "${INDEX_URL:?INDEX_URL must be set}"
: "${GPU_ARCH_TYPE:?GPU_ARCH_TYPE must be set}"
TORCH_VERSION="${TORCH_VERSION:-}"

ARTIFACTS="$RUNNER_TEMP/artifacts-to-be-uploaded"

# The generic Linux job chooses to use base env, not the one setup by the image
eval "$(conda shell.bash hook)"
CONDA_ENV=$(conda env list --json | jq -r ".envs | .[-1]")
conda activate "${CONDA_ENV}"

# ARC H100 runners mount the shared HF cache at /mnt/hf_cache read-only.
# Point datasets at per-job writable storage so c4_test can create its cache root.
export HF_HOME="$RUNNER_TEMP/hf_home"
export HF_DATASETS_CACHE="$RUNNER_TEMP/hf_home/datasets"

# Log CUDA driver version for debugging.
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 || true)
echo "CUDA driver version: ${DRIVER_VERSION}"

pip config --user set global.progress_bar off

TORCH_SPEC="torch"
if [ -n "${TORCH_VERSION}" ]; then
  TORCH_SPEC="torch==${TORCH_VERSION}"
fi

# Pre-install torch's pure-python deps from the in-cluster pypi-cache for speed.
python -m pip install filelock typing-extensions "setuptools<82" sympy networkx jinja2 fsspec numpy
# Uninstall any pre-existing torch so the nightly below installs cleanly
# without --force-reinstall (which would re-download torch's deps from
# the public PyPI CDN instead of the in-cluster cache).
python -m pip uninstall -y torch
# Clear PIP_EXTRA_INDEX_URL so the default cpu index can't supply a +cpu torch.
PIP_EXTRA_INDEX_URL= python -m pip install --pre "${TORCH_SPEC}" --index-url "${INDEX_URL}"

if [[ "${GPU_ARCH_TYPE}" == "rocm" ]]; then
  export HIPBLASLT_TENSILE_LIBPATH="$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib", "hipblaslt", "library"))')"
  echo "HIPBLASLT_TENSILE_LIBPATH=${HIPBLASLT_TENSILE_LIBPATH}"
fi

USE_CPP=0 PIP_EXTRA_INDEX_URL= python -m pip install --pre torchao --index-url "${INDEX_URL}"
# GPT-OSS production configs use PyTorch's FA3 backend.
python -m pip install flash-attn-3 \
  --extra-index-url https://download.pytorch.org/whl/test/cu130

# RUNNER_TEMP is owned by the host uid. The v2 ROCm runner's container user
# can't write it, so create + chown via sudo (as the other ROCm workflows do);
# the v3 CUDA runner can create it directly.
if [[ "${GPU_ARCH_TYPE}" == "rocm" ]]; then
  sudo mkdir -p "${ARTIFACTS}"
  sudo chown -R "$(id -u):$(id -g)" "${ARTIFACTS}"
else
  mkdir -p "${ARTIFACTS}"
fi

# Enable CPP stacktraces for debugging symmetric memory initialization errors.
# Disable Nvlink Sharp. The CI machine seems to be unstable state to support
# NLVS according to several CI runs.
# DeepEP needs CUDA_HOME specified to JIT kernels.
STATUS=0
if ! CUDA_HOME=/usr/local/cuda NCCL_NVLS_ENABLE=0 TORCH_SHOW_CPP_STACKTRACES=1 python -m tests.integration_tests.run_tests \
  --test_suite h100 \
  --execution_mode real_pg \
  --exclude qwen3_fsdp+deepep,deepseek_v3_fsdp+hybridep+compile \
  --gpu_arch_type "${GPU_ARCH_TYPE}" \
  --ngpu 8 \
  "${ARTIFACTS}/h100/base"; then
  STATUS=1
fi

# DeepEP v2 and HybridEP currently live on incompatible DeepEP branches and
# export different Python APIs. Install and test each backend sequentially.
if [[ "${GPU_ARCH_TYPE}" != "rocm" ]]; then
  bash .github/scripts/install_deepep_v2.sh
  if ! CUDA_HOME=/usr/local/cuda NCCL_NVLS_ENABLE=0 EP_DISABLE_GIN=1 TORCH_SHOW_CPP_STACKTRACES=1 python -m tests.integration_tests.run_tests \
    --test_suite h100 \
    --execution_mode real_pg \
    --test_name qwen3_fsdp+deepep \
    --gpu_arch_type "${GPU_ARCH_TYPE}" \
    --ngpu 8 \
    "${ARTIFACTS}/h100/deepep_v2"; then
    STATUS=1
  fi

  # The H100 CI image provides the pinned hybrid-ep installer used by the
  # existing HybridEP workflow.
  bash /install_deepep.sh
  if ! CUDA_HOME=/usr/local/cuda NCCL_NVLS_ENABLE=0 TORCH_SHOW_CPP_STACKTRACES=1 python -m tests.integration_tests.run_tests \
    --test_suite h100 \
    --execution_mode real_pg \
    --test_name deepseek_v3_fsdp+hybridep+compile \
    --gpu_arch_type "${GPU_ARCH_TYPE}" \
    --ngpu 8 \
    "${ARTIFACTS}/h100/hybrid_ep"; then
    STATUS=1
  fi
fi
rm -rf "${ARTIFACTS}"/*/checkpoint
exit "${STATUS}"
