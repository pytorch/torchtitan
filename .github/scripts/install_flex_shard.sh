#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Installs flex_shard for parallelism.fsdp_backend=FlexShardConfig(). It has no
# release yet, so this pins a commit; --no-deps keeps the torch installed before.

set -eux

FLEX_SHARD_COMMIT="d6bb54635b00c55a271d60f45bff7ebef7a03bc4"
python -m pip install --no-deps \
  "flex-shard @ git+https://github.com/meta-pytorch/flex_shard@${FLEX_SHARD_COMMIT}"
