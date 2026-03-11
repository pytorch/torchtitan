# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from .kernels import generate_permute_indices


def _permute(x, num_tokens_per_expert, ep_degree, num_local_experts):
    """Reorder tokens from rank-major to expert-major layout.

    After EP all-to-all, tokens are ordered by source rank then expert.
    This function reorders them so all tokens for each expert are contiguous,
    which is the layout required by grouped GEMM.
    """
    with torch.no_grad():
        (permuted_indices, num_tokens_per_expert, _offsets) = generate_permute_indices(
            num_tokens_per_expert,
            num_local_experts,
            ep_degree,
            x.shape[0],
        )

    input_shape = x.shape
    x = x[permuted_indices, :]

    return input_shape, x, permuted_indices, num_tokens_per_expert


def _unpermute(out, input_shape, permuted_indices):
    """Reverse the permutation applied by _permute."""
    out_unpermuted = out.new_empty(input_shape)
    out_unpermuted[permuted_indices, :] = out
    return out_unpermuted
