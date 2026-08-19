# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import math
from dataclasses import dataclass

from torchtitan.components.lr_scheduler import LRSchedulersContainer


class Olmo3CosWithWarmup(LRSchedulersContainer):
    """OLMo-core CosWithWarmup schedule for OLMo3 pretraining."""

    @dataclass(kw_only=True, slots=True)
    class Config(LRSchedulersContainer.Config):
        warmup_steps: int = 2000
        alpha_f: float = 0.1

    def __init__(self, config: Config, *, optimizers, training_steps):
        lr_lambda = functools.partial(
            self._lr_lambda,
            warmup_steps=config.warmup_steps,
            total_steps=config.total_steps if config.total_steps is not None else training_steps,
            alpha_f=config.alpha_f,
        )
        super().__init__(optimizers, lr_lambda)

    @staticmethod
    def _lr_lambda(
        current_step: int,
        *,
        warmup_steps: int,
        total_steps: int,
        alpha_f: float,
    ) -> float:
        if current_step < warmup_steps:
            return current_step / warmup_steps
        if current_step >= total_steps:
            return alpha_f

        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        return alpha_f + (1.0 - alpha_f) * (1.0 + math.cos(math.pi * progress)) / 2.0
