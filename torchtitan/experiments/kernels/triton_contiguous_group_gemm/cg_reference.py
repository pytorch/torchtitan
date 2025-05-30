# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


# Simple reference implementation for verification
def pytorch_reference(
    inputs: torch.Tensor,
    expert_weights: torch.Tensor,
    expert_indices: torch.Tensor,
    group_size_m: int = 128,
) -> torch.Tensor:
    """
    Reference implementation using PyTorch for verification.
    """
    M_total, K = inputs.shape
    num_experts, N, _ = expert_weights.shape

    output = torch.empty((M_total, N), device=inputs.device, dtype=inputs.dtype)

    # Process each group
    for i in range(0, M_total, group_size_m):
        end_idx = min(i + group_size_m, M_total)

        # Get expert index for this group
        expert_idx = expert_indices[i].item()

        # Get expert weights
        expert_weight = expert_weights[expert_idx]

        # Compute output for this group
        output[i:end_idx] = torch.matmul(inputs[i:end_idx], expert_weight.T)

    return output
