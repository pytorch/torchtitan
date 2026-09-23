#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux

DEEPEP_V2_COMMIT="01dc3aaac82068020353dce2c302e38153c0bfaa"
DEEPEP_V2_DIR=$(mktemp -d)
trap 'rm -rf "${DEEPEP_V2_DIR}"' EXIT

sudo apt-get update -qq
sudo apt-get install -y -qq rdma-core libibverbs1 libmlx5-1 libibverbs-dev

git clone --recursive https://github.com/deepseek-ai/DeepEP.git "${DEEPEP_V2_DIR}"
git -C "${DEEPEP_V2_DIR}" checkout "${DEEPEP_V2_COMMIT}"
git -C "${DEEPEP_V2_DIR}" submodule update --init --recursive

CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=9.0 \
  python -m pip install --no-build-isolation --force-reinstall "${DEEPEP_V2_DIR}"

python -c "from deep_ep import ElasticBuffer"
