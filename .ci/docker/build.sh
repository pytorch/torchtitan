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

OS=ubuntu
OS_VERSION=20.04
CLANG_VERSION=""
PYTHON_VERSION=3.11
MINICONDA_VERSION=24.3.0-0

case "${IMAGE_NAME}" in
  torchtitan-ubuntu-20.04-clang12)
    CLANG_VERSION=12
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
  --shm-size=1g \
  -f "${OS}"/Dockerfile \
  "$@" \
  .
