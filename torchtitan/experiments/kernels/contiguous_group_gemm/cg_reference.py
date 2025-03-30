# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


def reference_moe_forward(inputs, weights, indices):
    """
    Reference implementation of MoE forward pass using PyTorch.

    Args:
        inputs: Input tensor of shape [M, K]
        weights: Expert weight tensor of shape [num_experts, N, K]
        indices: Indices tensor of shape [M] mapping each token to its expert

    Returns:
        Output tensor of shape [M, N]
    """
    result = torch.zeros(
        (inputs.shape[0], weights.shape[1]),
        device=inputs.device,
        dtype=inputs.dtype,
    )

    for i in range(inputs.shape[0]):
        expert_idx = indices[i].item()
        result[i] = inputs[i] @ weights[expert_idx].T

    return result
