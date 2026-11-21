# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan.rl.generator.local_compile import inference_local_compile

# T = packed tokens, S = requests.


def _vllm_query_requests(
    query_start_loc: torch.Tensor, num_requests: int, num_tokens: int
):
    rows_T = torch.arange(num_tokens, device=query_start_loc.device)
    request_T = torch.searchsorted(query_start_loc[1:], rows_T, right=True).clamp_max(
        num_requests - 1
    )
    return rows_T, request_T, rows_T < query_start_loc[-1]


@inference_local_compile(batch_invariant=True)
def vllm_query_requests(
    query_start_loc: torch.Tensor, num_requests: int, num_tokens: int
):
    return _vllm_query_requests(query_start_loc, num_requests, num_tokens)


@inference_local_compile(batch_invariant=True)
def vllm_query_positions(
    query_start_loc: torch.Tensor, seq_lens: torch.Tensor, num_tokens: int
):
    rows_T, request_T, valid_T = _vllm_query_requests(
        query_start_loc, seq_lens.numel(), num_tokens
    )
    query_lens_S = query_start_loc.diff()
    positions_T = (
        seq_lens[request_T]
        - query_lens_S[request_T]
        + rows_T
        - query_start_loc[request_T]
    )
    return request_T, positions_T.masked_fill(~valid_T, 0), valid_T
