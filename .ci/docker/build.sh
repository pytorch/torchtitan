#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

IMAGE_NAME="$1"
shift

echo "Building ${IMAGE_NAME} Docker image"

# set operating system
OS=ubuntu

# set Dockerfile
DOCKERFILE="${OS}/Dockerfile"
if [[ "$IMAGE_NAME" == *cuda* ]]; then
  DOCKERFILE="${OS}-cuda/Dockerfile"
elif [[ "$IMAGE_NAME" == *rocm* ]]; then
  DOCKERFILE="${OS}-rocm/Dockerfile"
fi

case "${IMAGE_NAME}" in
  torchtitan-ubuntu-20.04-clang12)
    OS_VERSION=20.04
    CLANG_VERSION=12
    PYTHON_VERSION=3.11
    MINICONDA_VERSION=24.3.0-0
    ;;
  torchtitan-rocm-pytorch-nightly-ubuntu-22.04-clang19-py3)
    OS_VERSION=22.04
    CLANG_VERSION=19
    PYTHON_VERSION=3.10
    MINICONDA_VERSION=25.3.1-0
    ;;
  *)
    echo "Invalid image name ${IMAGE_NAME}"
    exit 1
esac

docker build \
  --no-cache \
  --progress=plain \
  --build-arg "OS_VERSION=${OS_VERSION}" \
  --build-arg "CLANG_VERSION=${CLANG_VERSION}" \
  --build-arg "PYTHON_VERSION=${PYTHON_VERSION}" \
  --build-arg "MINICONDA_VERSION=${MINICONDA_VERSION}" \
  -f $(dirname ${DOCKERFILE})/Dockerfile \
  "$@" \
  .

