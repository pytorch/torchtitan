# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch

from .kernels import generate_permute_indices


def _permute(x, num_tokens_per_expert, ep_degree, num_local_experts):
    max_len = x.shape[0]

    with torch.no_grad():
        (permuted_indices, num_tokens_per_expert, _offsets,) = generate_permute_indices(
            num_tokens_per_expert,
            num_local_experts,
            ep_degree,
            max_len,
        )

    input_shape = x.shape
    x = torch.index_select(x, 0, permuted_indices)

    return input_shape, x, permuted_indices, num_tokens_per_expert


def _unpermute(
    out: torch.Tensor, input_shape, permuted_indices: torch.Tensor
) -> torch.Tensor:
    out_unpermuted = out.new_empty(input_shape)
    out_unpermuted.index_copy_(0, permuted_indices.long(), out)
    return out_unpermuted
