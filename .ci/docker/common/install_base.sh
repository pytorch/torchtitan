#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

install_ubuntu() {
  apt-get update

  apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    wget \
    sudo \
    vim \
    jq \
    vim \
    unzip \
    gdb \
    rsync \
    libssl-dev \
    zip

  # Cleanup package manager
  apt-get autoclean && apt-get clean
  rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
}

# Install base packages depending on the base OS
ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')
case "$ID" in
  ubuntu)
    install_ubuntu
    ;;
  *)
    echo "Unable to determine OS..."
    exit 1
    ;;
esac
