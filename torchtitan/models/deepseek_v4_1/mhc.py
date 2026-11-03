# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Single-pass mHC: use the preceding sublayer's input coefficients.

T: tokens, C: hyper-connection branches, D: hidden width, Q: coefficient logits,
X: flattened branch features.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from torchtitan.models.deepseek_v4.mhc import HcPre, HcSplitSinkhorn


class HcSinkhorn(HcSplitSinkhorn):
    @dataclass(kw_only=True, slots=True)
    class Config(HcSplitSinkhorn.Config):
        pass

    def forward(  # pyrefly: ignore [bad-param-name-override]
        self, mixes_TQ, scale, base_Q
    ):
        num_branches = self.hc_mult
        pre_TC, post_TC, comb_TCC = mixes_TQ.split(
            [num_branches, num_branches, num_branches**2], dim=-1
        )
        pre_TC = torch.sigmoid(pre_TC * scale[0] + base_Q[:num_branches]) + self.eps
        post_TC = 2 * torch.sigmoid(
            post_TC * scale[1] + base_Q[num_branches : 2 * num_branches]
        )
        comb_TCC = (comb_TCC * scale[2] + base_Q[2 * num_branches :]).unflatten(
            -1, (num_branches, num_branches)
        )
        # V4.1 adds epsilon to softmax entries before column normalization.
        comb_TCC = comb_TCC.softmax(-1) + self.eps
        comb_TCC = comb_TCC / (comb_TCC.sum(-2, keepdim=True) + self.eps)
        for _ in range(self.sinkhorn_iters - 1):
            comb_TCC = comb_TCC / (comb_TCC.sum(-1, keepdim=True) + self.eps)
            comb_TCC = comb_TCC / (comb_TCC.sum(-2, keepdim=True) + self.eps)
        return pre_TC, post_TC, comb_TCC


class HcCoefficients(HcPre):
    """Predict coefficients without reducing the residual stream."""

    @dataclass(kw_only=True, slots=True)
    class Config(HcPre.Config):
        pass

    def __init__(self, config: Config):
        super().__init__(config)
        self.sinkhorn = HcSinkhorn.Config(
            hc_mult=config.hc_mult,
            sinkhorn_iters=config.sinkhorn_iters,
            eps=config.eps,
        ).build()
        self.float()

    def forward(self, x_TCD):  # pyrefly: ignore [bad-param-name-override]
        flat_TX = x_TCD.flatten(-2).float()
        rstd_T1 = torch.rsqrt(flat_TX.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes_TQ = F.linear(flat_TX, self.hc_fn.float()) * rstd_T1
        return self.sinkhorn(mixes_TQ, self.hc_scale.float(), self.hc_base.float())


def mix_branches(x_TCD: torch.Tensor, pre_TC: torch.Tensor) -> torch.Tensor:
    """Reduce branches using the carried coefficients, accumulating in FP32."""
    return (x_TCD.float() * pre_TC.unsqueeze(-1)).sum(-2).to(x_TCD.dtype)


def initial_pre_mix(x_TCD: torch.Tensor) -> torch.Tensor:
    """The reference starts with branch zero; all initial branches are equal."""
    pre_TC = torch.zeros_like(x_TCD[..., 0], dtype=torch.float32)
    pre_TC[:, 0] = 1
    return pre_TC
