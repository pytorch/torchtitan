# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SumDigits env (single-turn) — fast smoke + regression gate.

Task: given a list of integers, return their per-digit sum wrapped in
``<answer>NUMBER</answer>``. Used as the 10-step smoke gate that
exercises the full rollout → trainer → weight-sync loop on a tiny
model. Qwen3-0.6B starts at ~0.3 reward and reaches ~0.7 by step 10.

Example::

    builder = SumDigitsBuilder.Config().build()
    dataset = SumDigitsDataset.Config(seed=0).build()
    examples = dataset.sample_groups(step=5, num_groups=4)
    envs = await builder.make_envs(examples[0], group_size=8)
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass

from renderers import Message

from torchtitan.config import Configurable
from torchtitan.experiments.rl.envs.types import (
    EnvExample,
    EnvReset,
    EnvStep,
    MessageEnv,
)
from torchtitan.experiments.rl.types import RolloutStatus

__all__ = ["SumDigitsBuilder", "SumDigitsDataset", "SumDigitsEnv"]


SYSTEM_PROMPT = (
    "You are a careful arithmetic assistant. Given a list of integers, "
    "compute the sum of all their digits and respond with exactly "
    "`<answer>NUMBER</answer>` where NUMBER is the digit sum."
)

_ANSWER_RE = re.compile(r"<answer>\s*(-?\d+)\s*</answer>")


class SumDigitsEnv(MessageEnv):
    """Single-turn digit-sum task. ``reset`` → system+user; ``step`` scores."""

    def __init__(self, *, numbers: Sequence[int]) -> None:
        self._numbers = list(numbers)
        self._target = sum(int(d) for n in self._numbers for d in str(abs(n)))

    async def reset(self) -> EnvReset:
        return EnvReset(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"What is the digit sum of {self._numbers}?",
                },
            ]
        )

    async def step(self, assistant_message: Message) -> EnvStep:
        text = str(assistant_message.get("content") or "")
        m = _ANSWER_RE.search(text)
        prediction = int(m.group(1)) if m else None
        format_score = 1.0 if prediction is not None else 0.0
        correctness = 1.0 if prediction == self._target else 0.0
        return EnvStep(
            reward=correctness + 0.1 * format_score,
            reward_components={"correctness": correctness, "format": format_score},
            done=True,
            status=RolloutStatus.COMPLETED,
        )

    async def close(self) -> None:
        pass


class SumDigitsBuilder(Configurable):
    """Stateless builder: each call materializes ``group_size`` siblings."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass  # No per-builder state; payload carries the numbers.

    def __init__(self, config: Config) -> None:
        self.config = config

    async def make_envs(
        self, example: EnvExample, *, group_size: int
    ) -> Sequence[MessageEnv]:
        numbers = example.payload["numbers"]
        assert isinstance(
            numbers, list
        ), f"payload['numbers'] must be list, got {type(numbers).__name__}"
        return [SumDigitsEnv(numbers=numbers) for _ in range(group_size)]


class SumDigitsDataset(Configurable):
    """Deterministic synthetic dataset of digit-sum problems."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        seed: int = 0
        """``(seed, step, group_idx)`` is the per-example random seed."""

        min_count: int = 2
        """Minimum length of the integer list per example."""

        max_count: int = 4
        """Maximum length of the integer list per example."""

        min_value: int = 10
        """Minimum value of each integer (inclusive)."""

        max_value: int = 99
        """Maximum value of each integer (inclusive)."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def sample_groups(self, *, step: int, num_groups: int) -> Sequence[EnvExample]:
        cfg = self.config
        out: list[EnvExample] = []
        for i in range(num_groups):
            rng = random.Random(f"{cfg.seed}:{step}:{i}")
            n = rng.randint(cfg.min_count, cfg.max_count)
            numbers = [rng.randint(cfg.min_value, cfg.max_value) for _ in range(n)]
            out.append(
                EnvExample(
                    task_id=f"sum_digits/{step}/{i}",
                    payload={"numbers": numbers},
                    tags=("sum_digits",),
                )
            )
        return out
