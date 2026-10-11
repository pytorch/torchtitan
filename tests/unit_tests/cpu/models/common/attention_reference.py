# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch

from torchtitan.models.common.attention import InnerAttention


class CausalAttentionReference(InnerAttention):
    """Math-only causal attention for CPU tests of training infrastructure."""

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__()

    def forward(self, q_THK, k_THK, v_THV, *, scale=None, **kwargs):
        num_repeats = q_THK.shape[1] // k_THK.shape[1]
        k_THK = k_THK.repeat_interleave(num_repeats, dim=1)
        v_THV = v_THV.repeat_interleave(num_repeats, dim=1)
        scores_HTT = torch.einsum("thk,shk->hts", q_THK, k_THK)
        scores_HTT = scores_HTT * (q_THK.shape[-1] ** -0.5 if scale is None else scale)
        causal_TT = torch.ones(
            q_THK.shape[0], k_THK.shape[0], device=q_THK.device, dtype=torch.bool
        ).tril()
        probs_HTT = scores_HTT.masked_fill(~causal_TT, -torch.inf).softmax(dim=-1)
        return torch.einsum("hts,shv->thv", probs_HTT, v_THV)
