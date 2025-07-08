#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Script to test expert swapping (0 <-> 1) in MoE layers
# Usage: ./test_expert_swap.sh [NGPU]

NGPU=${2:-"2"}

echo "Testing expert swap with ${NGPU} GPU(s)..."
echo "This will swap experts 0 and 1 in all MoE layers"

# Run the expert swap test
torchrun --standalone --nproc-per-node ${NGPU} test_expert_swap.py

echo "Expert swap test completed!"
