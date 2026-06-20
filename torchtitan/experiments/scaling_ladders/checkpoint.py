# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checkpoint manager that saves at an explicit set of steps.

The WSD-S pre-decay / post-decay checkpoint pairs are the ladder deliverable, so
saving is driven by an explicit step list rather than the inherited interval
modulo. Overriding ``_should_save`` is sufficient because the base ``save()``
calls it first (precedent: ``TorchFTCheckpointManager`` customizes save behavior
via a ``CheckpointManager.Config`` subclass).
"""

from dataclasses import dataclass, field

from torchtitan.components.checkpoint import CheckpointManager


class LadderCheckpointManager(CheckpointManager):
    @dataclass(kw_only=True, slots=True)
    class Config(CheckpointManager.Config):
        checkpoint_steps: list[int] = field(default_factory=list)
        """Explicit steps to save at (the WSD-S pre/post-decay pairs)."""

    def __init__(self, config: Config, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.checkpoint_steps = set(config.checkpoint_steps)

    def _should_save(self, curr_step: int, last_step: bool = False) -> bool:
        if not self.enable or self.load_only:
            return False
        if last_step:
            return True
        return curr_step in self.checkpoint_steps
