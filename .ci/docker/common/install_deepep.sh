#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Install DeepEP (HybridEP) for MoE expert parallelism integration tests.
# Pinned to a known-good commit on the hybrid-ep branch to avoid
# breakage from upstream changes. Update the commit hash when
# upgrading to a newer DeepEP version.

set -eux

DEEPEP_COMMIT=${DEEPEP_COMMIT:-1b8f467}

# Dependencies for DeepEP compilation (NVSHMEM needs libcudacxx, IB headers).
sudo apt-get update -qq && sudo apt-get install -y -qq rdma-core libibverbs1 libmlx5-1 libibverbs-dev
sudo apt-get autoclean && sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/* /var/tmp/*

# CCCL headers live under include/cccl/ in CUDA 13+ — symlink so
# NVSHMEM's #include "cuda/std/tuple" resolves correctly.
if [ -d /usr/local/cuda/include/cccl/cuda ] && [ ! -e /usr/local/cuda/include/cuda ]; then
    sudo ln -sf /usr/local/cuda/include/cccl/cuda /usr/local/cuda/include/cuda
fi

# Non-fatal: DeepEP may fail to compile against newer PyTorch
# nightlies (ABI changes). HybridEP tests are skipped when
# DeepEP is unavailable.
git clone --branch hybrid-ep https://github.com/deepseek-ai/deepep.git /tmp/deepep
cd /tmp/deepep && git checkout $DEEPEP_COMMIT
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} TORCH_CUDA_ARCH_LIST="9.0" pip install --no-build-isolation . || echo "WARNING: DeepEP installation failed; HybridEP tests will be skipped"
cd / && rm -rf /tmp/deepep
