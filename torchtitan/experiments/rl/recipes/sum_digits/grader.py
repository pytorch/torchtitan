# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re

from torchtitan.experiments.rl.envs import last_assistant_text, Rollout
from torchtitan.experiments.rl.recipes.sum_digits.data import SumDigitsInput


_ANSWER_RE = re.compile(r"\[ANSWER\]\s*(-?\d+)")
_FORMAT_RE = re.compile(r"\[ANSWER\]\s*-?\d+")


async def reward_correct(rollout: Rollout, env_input: object) -> float:
    """1.0 if the last ``[ANSWER] <n>`` matches the target, else 0.0."""
    assert isinstance(env_input, SumDigitsInput)
    text = last_assistant_text(rollout)
    matches = _ANSWER_RE.findall(text)
    if not matches:
        return 0.0
    return 1.0 if int(matches[-1]) == env_input.target else 0.0


async def reward_format(rollout: Rollout, env_input: object) -> float:
    """1.0 if a ``[ANSWER] <n>`` tag is present anywhere in the response."""
    return 1.0 if _FORMAT_RE.search(last_assistant_text(rollout)) else 0.0
