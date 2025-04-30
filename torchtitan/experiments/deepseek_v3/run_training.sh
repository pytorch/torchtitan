#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

NGPU=${NGPU:-"8"}
export LOG_RANK=${LOG_RANK:-0}

# Run the model with basic training loop
torchrun --standalone --local-ranks-filter ${LOG_RANK} --nproc-per-node ${NGPU} train.py
