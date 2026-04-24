# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch


@dataclass(kw_only=True, slots=True)
class Step:
    """Env transition: named reward components, done flag, optional next observation.

    ``rewards`` is a dict of component-name to value (e.g.
    ``{"correctness": 1.0, "format": 0.3}``); envs are free to define
    any decomposition. Trainers read the scalar ``reward`` property
    (sum of components); loggers iterate ``rewards.items()`` for
    per-component reporting without needing to know the keys.

    ``observation`` (the next prompt the agent will see) is only
    populated by multi-turn envs. Single-turn envs leave it None.
    """

    rewards: dict[str, float]
    done: bool
    observation: str | None = None

    @property
    def reward(self) -> float:
        return sum(self.rewards.values())


@dataclass(kw_only=True, slots=True)
class Completion:
    """A single generated sequence from the generator.

    Pure generation artifact - no reward, no advantage. ``prompt_idx``
    is the position of the source prompt in the input ``prompts`` list.
    """

    policy_version: int
    prompt_idx: int
    prompt_token_ids: list[int]
    text: str
    token_ids: list[int]
    token_logprobs: list[float]


@dataclass(kw_only=True, slots=True)
class Trajectory:
    """One rollout: a sequence of ``(Completion, Step)`` transitions.

    Single-turn tasks produce trajectories with one transition. The
    Completion carries the generator's token-level metadata; the Step
    carries the env's reward and done flag.
    """

    sample_idx: int
    transitions: list[tuple[Completion, Step]]

    @property
    def total_reward(self) -> float:
        return sum(s.reward for _, s in self.transitions)


@dataclass(kw_only=True, slots=True)
class Episode:
    """Training sample: flattened trajectory + GRPO advantage.

    Flat shape (rather than composition) because the trainer collate
    path and logging read these fields directly.
    """

    policy_version: int
    prompt_idx: int
    prompt_token_ids: list[int]
    text: str
    token_ids: list[int]
    token_logprobs: list[float]
    reward: float
    advantage: float


@dataclass(kw_only=True, slots=True)
class TrainBatch:
    token_ids: torch.Tensor  # [1, total_tokens]
    prompt_lens: list[int]  # [num_episodes]
    response_lens: list[int]  # [num_episodes]
    seq_lens: list[int]  # [num_episodes] (prompt_lens + response_lens)
    advantages: torch.Tensor  # [num_episodes]
    token_logprobs: list[
        list[float]
    ]  # [num_episodes][response_len_i] per-token logprobs from rollout
