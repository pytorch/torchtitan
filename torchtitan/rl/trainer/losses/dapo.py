# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DAPO loss: per-token clipped surrogate with asymmetric "clip-higher" bounds."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from torchtitan.rl.trainer.losses.policy_gradient import (
    PolicyGradientLoss,
    ScoredTokens,
)


class DAPOLoss(PolicyGradientLoss):
    """Per-token clipped surrogate loss with DAPO-style "clip-higher".

    The same PPO clip as GRPO, but the importance ratio's lower and upper bounds are
    set independently (https://arxiv.org/abs/2503.14476): a larger upper bound keeps
    more probability mass on up-weighted tokens, countering entropy collapse. The
    generator's logprobs are the old policy, so the ratio is trainer over generator.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(PolicyGradientLoss.Config):
        ratio_clip_low: float = 0.2
        """Lower clip: the importance ratio is clamped to ``>= 1 - ratio_clip_low``."""

        ratio_clip_high: float = 0.2
        """Upper clip: the ratio is clamped to ``<= 1 + ratio_clip_high``. Set larger
        than ``ratio_clip_low`` for DAPO "clip-higher" (e.g. 0.28)."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.ratio_clip_low = config.ratio_clip_low
        self.ratio_clip_high = config.ratio_clip_high

    def token_loss(self, tokens: ScoredTokens) -> torch.Tensor:
        ratio, advantages = tokens.ratio, tokens.advantages
        clipped_ratio = self._clip(ratio)
        return -torch.min(ratio * advantages, clipped_ratio * advantages)

    def token_metrics(self, tokens: ScoredTokens) -> dict[str, torch.Tensor]:
        ratio_clipped = torch.abs(tokens.ratio - self._clip(tokens.ratio)) > 1e-6
        return {"loss/ratio_clipped_frac": ratio_clipped.float()}

    def _clip(self, ratio: torch.Tensor) -> torch.Tensor:
        return torch.clamp(ratio, 1 - self.ratio_clip_low, 1 + self.ratio_clip_high)
