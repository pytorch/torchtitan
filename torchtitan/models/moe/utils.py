# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Literal

import torch

from torchtitan.tools.utils import _round_up

from .kernels import generate_permute_indices

TOKEN_GROUP_ALIGN_SIZE_M = 8
ValidTokenGroupAlignmentSize = Literal[8, 16, 32]


def set_token_group_alignment_size_m(
    alignment_size: ValidTokenGroupAlignmentSize,
) -> None:
    """
    Set the token group alignment size for token groups in MoE. This is implemented by
    padding each token group size to the next multiple of TOKEN_GROUP_ALIGN_SIZE_M.

    Valid values are: 8, 16, or 32.
    Different values are needed for different cases:

    * For bf16, 8 is enough (16 byte alignment / 2 bytes per elem = 8 elements).
    * For fp8, 16 byte alignment / 1 byte per elem = 16 elements.
    * For mxfp8, we need 32 (or block_size) because scaling block size is (1 x 32),
      so when doing per-token-group quantization on each logically distinct subtensor,
      we need to ensure the contracting dim is divisible by block_size.
      In the backward pass, grad_weight = (grad_output_t @ input).t() has gemm dims
      of (N, M) @ (M, K) so M is the contracting dim, and group offsets are along M,
      so we need 32 element alignment.
    """
    global TOKEN_GROUP_ALIGN_SIZE_M
    TOKEN_GROUP_ALIGN_SIZE_M = alignment_size


def _permute(
    tensor: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    ep_degree: int,
    num_local_experts: int,
) -> tuple[torch.Size, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Permute and pad the input tokens x based on the number of tokens per expert.
    This ensures that each expert receives a number of tokens that is a multiple of TOKEN_GROUP_ALIGN_SIZE_M.

    Args:
        x (torch.Tensor): Input tensor of shape (num_tokens, feature_dim).
        num_tokens_per_expert (torch.Tensor): Tensor containing the number of tokens assigned to each expert.
        ep_degree (int): Expert parallelism degree.
        num_local_experts (int): Number of local experts.

    Returns:
        input_shape (torch.Size): shape of input tensor after a dummy padding row is added.
        x (torch.Tensor): Permuted and padded input tensor.
        permuted_indices (torch.Tensor): Indices used for permutation.
        num_tokens_per_expert (torch.Tensor): Updated number of tokens per expert after padding.
    """
    global TOKEN_GROUP_ALIGN_SIZE_M

    # Unwrap MXTensor into quantized data and scales if applicable.
    x = tensor.qdata if isinstance(tensor, MXTensor) else tensor
    scales = tensor.scales if isinstance(tensor, MXTensor) else None

    # Assume worst case where each token group needs to be padded with TOKEN_GROUP_ALIGN_SIZE_M tokens.
    x_padded_per_expert = x.shape[0] + num_local_experts * TOKEN_GROUP_ALIGN_SIZE_M
    padded_max_len = _round_up(x_padded_per_expert, TOKEN_GROUP_ALIGN_SIZE_M)

    # Generate permuted indices and updated num_tokens_per_expert with padding.
    with torch.no_grad():
        (permuted_indices, num_tokens_per_expert, _offsets,) = generate_permute_indices(
            num_tokens_per_expert,
            num_local_experts,
            ep_degree,
            padded_max_len,
            TOKEN_GROUP_ALIGN_SIZE_M,
        )

    # Append row of zeros to x to act as a dummy padding row. Every time `permuted_indices` selects this index,
    # it will correspond to a padded token.
    x = torch.vstack((x, x.new_zeros((x.shape[-1]))))
    input_shape = x.shape
    x = x[permuted_indices, :]

    # Permute scales to match data if applicable. If so, wrap back into MXTensor.
    if isinstance(tensor, MXTensor):
        scales = torch.vstack((scales, scales.new_zeros((scales.shape[-1]))))
        scales = scales[
            permuted_indices:,
        ]
        x = MXTensor(
            x,
            scales,
            tensor._elem_dtype,
            tensor.block_size,
            tensor._orig_dtype,
            tensor.kernel_preference,
            tensor.act_quant_kwargs,
            tensor._is_swizzled_scales,
        )

    return input_shape, x, permuted_indices, num_tokens_per_expert


def _unpermute(out, input_shape, permuted_indices):
    out_unpermuted = out.new_empty(input_shape)
    out_unpermuted[permuted_indices, :] = out
    out = out_unpermuted[:-1]
    return out


def indices_padding_wrapper(func: Callable) -> Callable:
    """
    In order to use torch._grouped_mm, we need to make sure the number of
    tokens each expert gets is a multiple of TOKEN_GROUP_ALIGN_SIZE_M. The
    generate_permute_indices kernel also helps achieve this via padding,
    without incurring synchronization between device and host.
    """

    def wrapper(
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        num_local_experts = w1.shape[0]
        ep_degree = num_tokens_per_expert.shape[0] // num_local_experts

        input_shape, x, permuted_indices, num_tokens_per_expert = _permute(
            x, num_tokens_per_expert, ep_degree, num_local_experts
        )

        out = func(w1, w2, w3, x, num_tokens_per_expert)

        out = _unpermute(out, input_shape, permuted_indices)

        return out

    return wrapper
