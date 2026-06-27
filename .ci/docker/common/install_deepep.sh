#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Install DeepEP v2 (the unified ElasticBuffer API) for MoE expert parallelism
# integration tests. v2 lives on the `main` branch, which ships both
# `deep_ep.Buffer` (legacy) and `deep_ep.ElasticBuffer` (the unified HT/LL
# dispatch/combine used by moe_comm_backend="deepep"). Pinned to a known-good
# commit to avoid breakage from upstream changes; update the hash when upgrading.
#
# NOTE: main does NOT include `hybrid_ep_buffer` (HybridEP) -- that is a separate
# experimental branch. Tests that need the HybridEP backend are skipped here.

set -eux

# v2.0.0 (deepseek-ai/DeepEP main, ElasticBuffer). Override with DEEPEP_COMMIT.
DEEPEP_COMMIT=${DEEPEP_COMMIT:-af9a040}

# Dependencies for DeepEP compilation (NVSHMEM needs libcudacxx, IB headers).
sudo apt-get update -qq && sudo apt-get install -y -qq rdma-core libibverbs1 libmlx5-1 libibverbs-dev
sudo apt-get autoclean && sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/* /var/tmp/*

# CCCL headers live under include/cccl/ in CUDA 13+ — symlink so
# NVSHMEM's #include "cuda/std/tuple" resolves correctly.
if [ -d /usr/local/cuda/include/cccl/cuda ] && [ ! -e /usr/local/cuda/include/cuda ]; then
    sudo ln -sf /usr/local/cuda/include/cccl/cuda /usr/local/cuda/include/cuda
fi

# NVSHMEM + NCCL are provided as pip wheels (nvidia-nvshmem-cu13, nvidia-nccl-cu13).
# v2 links BOTH (csrc/kernels/backend/nccl.cu is new vs v1), so expose their
# headers/libs to the build and add the unversioned `lib*.so` symlinks the linker
# resolves by exact name (`-l:libnvshmem_host.so`, `-l:libnccl.so`).
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
NVSHMEM_DIR="${SITE_PACKAGES}/nvidia/nvshmem"
NCCL_DIR="${SITE_PACKAGES}/nvidia/nccl"
if [ -e "${NVSHMEM_DIR}/lib/libnvshmem_host.so.3" ] && [ ! -e "${NVSHMEM_DIR}/lib/libnvshmem_host.so" ]; then
    ln -sf libnvshmem_host.so.3 "${NVSHMEM_DIR}/lib/libnvshmem_host.so"
fi

# Non-fatal: DeepEP may fail to compile against newer PyTorch
# nightlies (ABI changes). DeepEP tests are skipped when it is unavailable.
git clone --branch main https://github.com/deepseek-ai/deepep.git /tmp/deepep
cd /tmp/deepep && git checkout $DEEPEP_COMMIT
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} \
    TORCH_CUDA_ARCH_LIST="9.0" \
    NVSHMEM_DIR="${NVSHMEM_DIR}" \
    EP_NCCL_ROOT_DIR="${NCCL_DIR}" \
    CPATH="${NVSHMEM_DIR}/include:${NCCL_DIR}/include:${CPATH:-}" \
    LIBRARY_PATH="${NVSHMEM_DIR}/lib:${NCCL_DIR}/lib:${LIBRARY_PATH:-}" \
    pip install --no-build-isolation . || echo "WARNING: DeepEP installation failed; DeepEP tests will be skipped"
cd / && rm -rf /tmp/deepep
