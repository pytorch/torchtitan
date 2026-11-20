# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SLAKE medical VQA: image + question -> short answer, graded by exact match."""

from torchtitan.rl.examples.slake.data import SlakeDataset, SlakeSample
from torchtitan.rl.examples.slake.env import SlakeEnv
from torchtitan.rl.examples.slake.rubric import RewardSlakeAnswer

__all__ = [
    "RewardSlakeAnswer",
    "SlakeDataset",
    "SlakeEnv",
    "SlakeSample",
]
