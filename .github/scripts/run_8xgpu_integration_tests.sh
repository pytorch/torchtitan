#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Shared setup + run for the H100 integration test suite. Both jobs in
# integration_test_8gpu_h100.yaml call this -- the CUDA build-test on
# linux_job_v3 and the ROCm build-test-rocm on linux_job_v2 -- with GPU_ARCH_TYPE
# set accordingly. The only arch-specific step is the ROCm HIPBLASLT export
# below, a no-op on CUDA.
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

# RUNNER_TEMP is owned by the host uid. The v2 ROCm runner's container user
# can't write it, so create + chown via sudo (as the other ROCm workflows do);
# the v3 CUDA runner can create it directly.
if [[ "${GPU_ARCH_TYPE}" == "rocm" ]]; then
  sudo mkdir -p "${ARTIFACTS}"
  sudo chown -R "$(id -u):$(id -g)" "${ARTIFACTS}"
else
  mkdir -p "${ARTIFACTS}"
fi

# Install DeepEP for the HybridEP integration test. DeepEP (NVSHMEM) is
# CUDA-only, so skip it on ROCm.
if [[ "${GPU_ARCH_TYPE}" != "rocm" ]]; then
  bash /install_deepep.sh
fi

# Enable CPP stacktraces for debugging symmetric memory initialization errors.
# Disable Nvlink Sharp. The CI machine seems to be unstable state to support
# NLVS according to several CI runs.
# DeepEP needs CUDA_HOME specified to JIT kernels.
CUDA_HOME=/usr/local/cuda NCCL_NVLS_ENABLE=0 TORCH_SHOW_CPP_STACKTRACES=1 python -m tests.integration_tests.run_tests --test_suite h100 --gpu_arch_type "${GPU_ARCH_TYPE}" "${ARTIFACTS}" --ngpu 8
rm -rf "${ARTIFACTS}"/*/checkpoint
