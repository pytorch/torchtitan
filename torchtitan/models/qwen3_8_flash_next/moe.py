# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reference rounding at the Flash-Next routing boundary."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch_remat as remat

from torchtitan.models.common.activation import Softmax
from torchtitan.models.common.moe.moe import TokenChoiceTopKRouter

# T = packed tokens, D = model dimension, E = experts, K = selected experts.


class Float32Softmax(Softmax):
    """Apply FP32 softmax to logits materialized in the activation dtype."""

    @dataclass(kw_only=True, slots=True)
    class Config(Softmax.Config):
        pass

    def __call__(self, input_TE: torch.Tensor, **kwargs) -> torch.Tensor:
        return F.softmax(input_TE, dim=self.dim, dtype=torch.float32)


class FlashNextRouter(TokenChoiceTopKRouter):
    """Shared routing and auxiliary objectives with reference gating dtype."""

    @dataclass(kw_only=True, slots=True)
    class Config(TokenChoiceTopKRouter.Config):
        pass

    def forward(
        self, x_TD: torch.Tensor, expert_bias_E: torch.Tensor | None = None, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scores_TK, indices_TK, routing_TE = super().forward(
            x_TD, expert_bias_E, **kwargs
        )
        remat.recompute_needs_tensor(scores_TK)
        return scores_TK.to(x_TD.dtype), indices_TK, routing_TE
