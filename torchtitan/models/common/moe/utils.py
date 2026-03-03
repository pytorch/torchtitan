# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from typing import Literal

import torch

from torchtitan.tools.utils import _round_up

from .kernels import generate_permute_indices

TOKEN_GROUP_ALIGN_SIZE_M = 1
ValidTokenGroupAlignmentSize = Literal[1, 8, 16, 32]


def set_token_group_alignment_size_m(
    alignment_size: ValidTokenGroupAlignmentSize,
) -> None:
    """
    Set the token group alignment size for token groups in MoE. This is implemented by
    padding each token group size to the next multiple of TOKEN_GROUP_ALIGN_SIZE_M.

    Valid values are: 1, 8, 16, or 32.
    Different values are needed for different cases:

    * For bf16, 1 (no padding needed). BF16 grouped GEMM natively supports
      variable-size token groups without alignment constraints.
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


def _permute(x, num_tokens_per_expert, ep_degree, num_local_experts):
    global TOKEN_GROUP_ALIGN_SIZE_M

    if TOKEN_GROUP_ALIGN_SIZE_M == 1:
        # BF16 path: no padding needed, just reorder tokens from rank-major
        # to expert-major order.
        padded_max_len = x.shape[0]
    else:
        x_padded_per_expert = x.shape[0] + num_local_experts * TOKEN_GROUP_ALIGN_SIZE_M
        padded_max_len = _round_up(x_padded_per_expert, TOKEN_GROUP_ALIGN_SIZE_M)

    with torch.no_grad():
        (permuted_indices, num_tokens_per_expert, _offsets,) = generate_permute_indices(
            num_tokens_per_expert,
            num_local_experts,
            ep_degree,
            padded_max_len,
            TOKEN_GROUP_ALIGN_SIZE_M,
        )

    if TOKEN_GROUP_ALIGN_SIZE_M > 1:
        # Append a zero row so that padding indices (-1 wrapped to last index)
        # fetch zeros instead of garbage.
        x = torch.vstack((x, x.new_zeros((x.shape[-1]))))

    input_shape = x.shape
    x = x[permuted_indices, :]

    return input_shape, x, permuted_indices, num_tokens_per_expert


def _unpermute(out, input_shape, permuted_indices):
    global TOKEN_GROUP_ALIGN_SIZE_M

    out_unpermuted = out.new_empty(input_shape)
    out_unpermuted[permuted_indices, :] = out
    if TOKEN_GROUP_ALIGN_SIZE_M == 1:
        out = out_unpermuted
    else:
        # Strip the extra zero row appended in _permute for the padding path.
        out = out_unpermuted[:-1]
    return out


def indices_padding_wrapper(func: Callable) -> Callable:
    """
    In order to use torch._grouped_mm, we need to make sure the number of
    tokens each expert gets is a multiple of TOKEN_GROUP_ALIGN_SIZE_M. The
    generate_permute_indices kernel also helps achieve this via padding,
    without incurring synchronization between device and host.

    When TOKEN_GROUP_ALIGN_SIZE_M == 1 (BF16 path), this wrapper is a no-op
    since BF16 grouped GEMM does not require token group alignment.
    """

    def wrapper(
        w1: torch.Tensor,
        w2: torch.Tensor,
        w3: torch.Tensor,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        global TOKEN_GROUP_ALIGN_SIZE_M

        if TOKEN_GROUP_ALIGN_SIZE_M == 1:
            # BF16 path: no padding needed, call function directly.
            return func(w1, w2, w3, x, num_tokens_per_expert)

        # FP8/MXFP8 path: pad token groups before computation,
        # then unpad after.
        num_local_experts = w1.shape[0]
        ep_degree = num_tokens_per_expert.shape[0] // num_local_experts

        input_shape, x, permuted_indices, num_tokens_per_expert = _permute(
            x, num_tokens_per_expert, ep_degree, num_local_experts
        )

        out = func(w1, w2, w3, x, num_tokens_per_expert)

        out = _unpermute(out, input_shape, permuted_indices)

        return out

    return wrapper
