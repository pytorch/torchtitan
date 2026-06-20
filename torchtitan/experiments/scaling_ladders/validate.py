# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Validator that fires at an explicit list of steps.

TorchTitan's ``Validator`` is frequency-modulo only; the ladder compares rungs by
validation loss at the matched-Chinchilla post-decay checkpoints, so validation
must fire at exactly those steps. ``fixed_steps`` is set to the post-decay steps;
step 1 is also validated to capture a baseline.
"""

from dataclasses import dataclass, field

from torchtitan.components.validate import Validator


class LadderValidator(Validator):
    @dataclass(kw_only=True, slots=True)
    class Config(Validator.Config):
        fixed_steps: list[int] = field(default_factory=list)
        """Steps at which to run validation (the post-decay checkpoint steps)."""

    def should_validate(self, step: int) -> bool:
        # self.config is the Config subclass above; pyrefly sees the base type.
        # pyrefly: ignore [missing-attribute]
        return step == 1 or step in self.config.fixed_steps
