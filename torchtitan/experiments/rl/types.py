# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch


@dataclass
class Episode:
    """
    A single prompt + completion pair with reward and advantage.

    The generator creates Episodes (with group_id, no reward yet).
    The grader fills in the reward.
    The controller computes advantages across episodes sharing a group_id.
    The trainer consumes the final Episodes with advantages set.

    Attributes:
        policy_version: Version of policy that produced this episode.
        prompt_token_ids: Token IDs for the prompt.
        text: Decoded completion text.
        token_ids: Completion token IDs.
        token_log_probs: Per-token log probabilities from the generator.
        expected_answer: Expected answer for reward computation.
            Passed to Episode by the generator — the generator
            does not read this field.
        reward: Scalar reward assigned by the grader.
        group_id: Identifies which group this episode belongs to.
            Episodes with the same group_id share a prompt and have
            their advantages normalized together.
        advantage: Advantage value computed by the controller (GRPO:
            reward minus group mean reward).
    """

    policy_version: int
    prompt_token_ids: list[int]
    text: str
    token_ids: list[int]
    token_log_probs: list[float]
    expected_answer: str = ""
    reward: float = 0.0
    group_id: str = ""
    advantage: float = 0.0


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
