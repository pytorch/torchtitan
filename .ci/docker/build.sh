#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Note CUDA uses devel container since comm_backend like DeepEP
# needs the CUDA toolkit during runtime.

set -exu

IMAGE_NAME="$1"
shift

echo "Building ${IMAGE_NAME} Docker image"

OS=ubuntu
CLANG_VERSION=""
PYTHON_VERSION=3.12
MINICONDA_VERSION=24.3.0-0

case "${IMAGE_NAME}" in
  torchtitan-ubuntu-22.04-clang12)
    OS_VERSION=22.04
    CLANG_VERSION=12
    BASE_IMAGE=nvidia/cuda:13.0.3-cudnn-devel-ubuntu${OS_VERSION}
    ;;
  torchtitan-ubuntu-22.04-clang12:rl)
    OS_VERSION=22.04
    CLANG_VERSION=12
    BASE_IMAGE=nvidia/cuda:13.0.3-cudnn-devel-ubuntu${OS_VERSION}
    INSTALL_RL_DEPS=1
    ;;
  torchtitan-rocm-ubuntu-22.04-clang12)
    OS_VERSION=22.04
    CLANG_VERSION=12
    BASE_IMAGE=rocm/dev-ubuntu-${OS_VERSION}:latest
    ;;
  *)
    echo "Invalid image name ${IMAGE_NAME}"
    exit 1
esac

# On OSDC the image is built by the out-of-cluster BuildKit pool through a
# remote buildx builder. That builder has nowhere to load an image into, so the
# result has to go straight to the registry.
if [[ -n "${REMOTE_BUILDKIT:-}" ]]; then
  BUILD_CMD=(docker buildx build --push)
else
  BUILD_CMD=(docker build)
fi

"${BUILD_CMD[@]}" \
  --no-cache \
  --progress=plain \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "OS_VERSION=${OS_VERSION}" \
  --build-arg "CLANG_VERSION=${CLANG_VERSION}" \
  --build-arg "PYTHON_VERSION=${PYTHON_VERSION}" \
  --build-arg "MINICONDA_VERSION=${MINICONDA_VERSION}" \
  --build-arg "INSTALL_RL_DEPS=${INSTALL_RL_DEPS:-0}" \
  --shm-size=1g \
  -f "${OS}"/Dockerfile \
  "$@" \
  .
