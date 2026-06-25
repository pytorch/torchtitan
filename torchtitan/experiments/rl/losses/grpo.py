# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GRPO loss: the special case of DAPO with a single symmetric clip."""

from __future__ import annotations

from dataclasses import dataclass

from torchtitan.components.loss import BaseLoss
from torchtitan.experiments.rl.losses.dapo import DAPOLoss


class GRPOLoss(DAPOLoss):
    """Per-token clipped surrogate loss for GRPO.

    GRPO is ``DAPOLoss`` with equal lower/upper clip bounds (no clip-higher), so this
    just exposes a single ``clip_eps`` and reuses DAPOLoss's per-token surrogate.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseLoss.Config):
        clip_eps: float = 0.2
        """Symmetric PPO clip: the ratio is clamped to ``[1 - clip_eps, 1 + clip_eps]``."""

    def __init__(self, config: Config, **kwargs) -> None:
        super().__init__(
            DAPOLoss.Config(
                ratio_clip_low=config.clip_eps,
                ratio_clip_high=config.clip_eps,
            ),
            **kwargs,
        )
