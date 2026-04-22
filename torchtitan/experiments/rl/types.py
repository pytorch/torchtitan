# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch


@dataclass
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


@dataclass
class ScoredCompletion:
    """A Completion with a scalar reward attached by the grader."""

    completion: Completion
    reward: float


@dataclass
class Episode:
    """Training sample: flattened scored completion + GRPO advantage.

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


@dataclass
class TrainBatch:
    token_ids: torch.Tensor  # [1, total_tokens]
    prompt_lens: list[int]  # [num_episodes]
    response_lens: list[int]  # [num_episodes]
    seq_lens: list[int]  # [num_episodes] (prompt_lens + response_lens)
    advantages: torch.Tensor  # [num_episodes]
    token_logprobs: list[
        list[float]
    ]  # [num_episodes][response_len_i] per-token logprobs from rollout
