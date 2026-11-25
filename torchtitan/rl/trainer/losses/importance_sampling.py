# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""REINFORCE with a truncated (CISPO) or masked (IcePop) importance weight."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from torchtitan.rl.trainer.losses.policy_gradient import (
    MAX_LOG_RATIO,
    PolicyGradientLoss,
    ScoredTokens,
)


class CISPOLoss(PolicyGradientLoss):
    """REINFORCE with a truncated importance weight (https://arxiv.org/abs/2506.13585).

    Each token's loss is ``-A * sg(min(r, max_ratio)) * log p`` with ``r = p / q``:
    the weight is truncated, never zeroed, so every token keeps a gradient, unlike under the
    PPO clip. This is truncated importance sampling (TIS).

    Example (max_ratio=2.0):

        ratio:     [0.5, 1.0, 3.0]
        weight:    [0.5, 1.0, 2.0]
        gradient:  -A * weight * grad(log p)
    """

    @dataclass(kw_only=True, slots=True)
    class Config(PolicyGradientLoss.Config):
        max_ratio: float = 2.0
        """Truncate the detached ratio at this value. 2 is the cap of truncated importance
        sampling in verl and in arXiv 2609.20807."""

        def __post_init__(self) -> None:
            if self.max_ratio < 1:
                raise ValueError(
                    f"CISPOLoss.max_ratio must be >= 1, got {self.max_ratio}"
                )

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.max_ratio = config.max_ratio

    def token_loss(self, tokens: ScoredTokens) -> torch.Tensor:
        weight = torch.clamp(tokens.ratio, max=self.max_ratio).detach()
        return -tokens.advantages * weight * tokens.trainer_logprobs

    def log_token_weight(
        self,
        log_ratio: torch.Tensor,
        advantages: torch.Tensor,
        trainer_logprobs: torch.Tensor,
    ) -> torch.Tensor:
        del trainer_logprobs  # the weight reads the ratio alone
        # The weight reads PolicyGradientLoss's ratio, whose |log r| is clamped.
        clamped = torch.clamp(log_ratio, -MAX_LOG_RATIO, MAX_LOG_RATIO)
        return torch.clamp(clamped, max=math.log(self.max_ratio))

    def token_metrics(self, tokens: ScoredTokens) -> dict[str, torch.Tensor]:
        return {"loss/ratio_truncated_frac": (tokens.ratio > self.max_ratio).float()}


class IcePopLoss(PolicyGradientLoss):
    """REINFORCE with a masked importance weight (IcePop, https://arxiv.org/abs/2510.18855).

    Each token's loss is ``-A * sg(r) * log p`` while ``r = p / q`` lies in
    ``[ratio_mask_low, ratio_mask_high]``, and 0 outside: tokens the trainer and the
    generator disagree on too much are dropped. This is masked importance sampling.
    IcePop's separate PPO clip between policy versions is not part of this loss: the
    generator's logprobs are the old policy, so there is one ratio.

    Example (band [0.5, 5]):

        ratio:     [0.4, 1.0, 3.0, 6.0]
        weight:    [0.0, 1.0, 3.0, 0.0]
        gradient:  -A * weight * grad(log p)
    """

    @dataclass(kw_only=True, slots=True)
    class Config(PolicyGradientLoss.Config):
        ratio_mask_low: float = 0.5
        """Tokens with ``r < ratio_mask_low`` get no gradient. 0.5 is IcePop's default."""

        ratio_mask_high: float = 5.0
        """Tokens with ``r > ratio_mask_high`` get no gradient. 5.0 is IcePop's default."""

        def __post_init__(self) -> None:
            # A band containing 1 keeps every token of a generator that matches the trainer.
            if not 0 < self.ratio_mask_low <= 1 <= self.ratio_mask_high:
                raise ValueError(
                    "IcePopLoss needs 0 < ratio_mask_low <= 1 <= ratio_mask_high, got "
                    f"[{self.ratio_mask_low}, {self.ratio_mask_high}]"
                )

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.ratio_mask_low = config.ratio_mask_low
        self.ratio_mask_high = config.ratio_mask_high

    def token_loss(self, tokens: ScoredTokens) -> torch.Tensor:
        weight = (tokens.ratio * self._in_band(tokens.ratio)).detach()
        return -tokens.advantages * weight * tokens.trainer_logprobs

    def log_token_weight(
        self,
        log_ratio: torch.Tensor,
        advantages: torch.Tensor,
        trainer_logprobs: torch.Tensor,
    ) -> torch.Tensor:
        del trainer_logprobs  # the weight reads the ratio alone
        # The band lies inside PolicyGradientLoss's |log r| <= MAX_LOG_RATIO clamp.
        in_band = (log_ratio >= math.log(self.ratio_mask_low)) & (
            log_ratio <= math.log(self.ratio_mask_high)
        )
        return torch.where(in_band, log_ratio, -torch.inf)

    def token_metrics(self, tokens: ScoredTokens) -> dict[str, torch.Tensor]:
        return {"loss/ratio_masked_frac": (~self._in_band(tokens.ratio)).float()}

    def _in_band(self, ratio: torch.Tensor) -> torch.Tensor:
        return (ratio >= self.ratio_mask_low) & (ratio <= self.ratio_mask_high)
