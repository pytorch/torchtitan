# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Nullary default-recipe configs for ``run_train.sh`` / ``ConfigManager``.

Each function returns the default recipe for one rung (no policy overrides); the
API-driven and swept runs go through ``train.py`` instead. ``llama3_ladder_debug``
uses a tiny rung, ``c4_test``, and the test tokenizer for fake-backend/smoke runs.
"""

from torchtitan.trainer import Trainer

from . import LADDER
from .ladder import debug_ladder


def llama3_ladder_60m() -> Trainer.Config:
    return LADDER.trainer_config("60M")


def llama3_ladder_100m() -> Trainer.Config:
    return LADDER.trainer_config("100M")


def llama3_ladder_190m() -> Trainer.Config:
    return LADDER.trainer_config("190M")


def llama3_ladder_370m() -> Trainer.Config:
    return LADDER.trainer_config("370M")


def llama3_ladder_760m() -> Trainer.Config:
    return LADDER.trainer_config("760M")


def llama3_ladder_1b() -> Trainer.Config:
    return LADDER.trainer_config("1B")


def llama3_ladder_3b() -> Trainer.Config:
    return LADDER.trainer_config("3B")


def llama3_ladder_8b() -> Trainer.Config:
    return LADDER.trainer_config("8B")


def llama3_ladder_debug() -> Trainer.Config:
    config = debug_ladder().trainer_config("debug")
    # The planner derives a full-horizon schedule; cap the debug recipe to a
    # genuinely short run here (matching llama3_debugmodel's steps=10) instead of
    # branching the planner. The WSD-S period checkpoint steps are never reached.
    config.training.steps = 10
    # Disable checkpointing for the smoke recipe: DCP cannot build a multi-rank
    # save plan from a single fake_backend process, so (like llama3_debugmodel)
    # the debug config builds and steps without saving.
    config.checkpoint.enable = False
    return config
