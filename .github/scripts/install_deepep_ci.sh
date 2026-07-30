#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux

DEEPEP_INSTALL_SCRIPT=/install_deepep.sh

# The pinned DeepEP commit builds hybrid_ep_cpp with -std=c++17, but current
# PyTorch nightly headers require C++20. Patch the image-provided installer
# after it clones DeepEP and before it runs pip install.
if ! sudo grep -Fq "pip install --no-build-isolation" "${DEEPEP_INSTALL_SCRIPT}"; then
  echo "Unexpected DeepEP installer layout: missing pip build command"
  exit 1
fi

if ! sudo grep -Fq 's/-std=c++17/-std=c++20/g' "${DEEPEP_INSTALL_SCRIPT}"; then
  sudo sed -i '/pip install --no-build-isolation/i sed -i "s/-std=c++17/-std=c++20/g" /tmp/deepep/setup.py' "${DEEPEP_INSTALL_SCRIPT}"
fi

bash "${DEEPEP_INSTALL_SCRIPT}"
