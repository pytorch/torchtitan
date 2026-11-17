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
DOCKERFILE_DIR="${OS}"
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
  torchtitan-xpu-ubuntu-24.04-py3)
    # Layers torchtitan deps on pytorch's official XPU CI image (noble/24.04),
    # which already ships driver, oneAPI, conda and the jenkins user. Unlike
    # CUDA/ROCm's pinned vendor tags, this base rolls with pytorch nightly, so
    # there is no fixed tag to hardcode here: the caller (docker-builds.yml)
    # must resolve XPU_BASE_IMAGE from the current nightly .ci/docker tree
    # hash via test-infra's calculate-docker-image action and pass it in. No
    # fallback - a stale pinned tag would silently build against whatever
    # nightly snapshot happened to exist when someone last wrote it down.
    OS_VERSION=24.04
    BASE_IMAGE="${XPU_BASE_IMAGE}"
    DOCKERFILE_DIR=xpu
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
  --progress=plain \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "OS_VERSION=${OS_VERSION}" \
  --build-arg "CLANG_VERSION=${CLANG_VERSION}" \
  --build-arg "PYTHON_VERSION=${PYTHON_VERSION}" \
  --build-arg "MINICONDA_VERSION=${MINICONDA_VERSION}" \
  --build-arg "INSTALL_RL_DEPS=${INSTALL_RL_DEPS:-0}" \
  --shm-size=1g \
  -f "${DOCKERFILE_DIR}"/Dockerfile \
  "$@" \
  .
