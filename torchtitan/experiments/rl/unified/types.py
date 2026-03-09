# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass


@dataclass
class Episode:
    """
    A single prompt + completion pair produced by the generator.

    Each Episode is self-contained: it holds the prompt tokens and one
    completion's tokens, text, log-probs, and reward. When the generator
    samples N completions per prompt, it emits N Episode objects grouped
    into an :data:`EpisodeGroup`.

    Attributes:
        policy_version: Version of policy that produced this episode.
        prompt_token_ids: Token IDs for the prompt.
        text: Decoded completion text.
        token_ids: Completion token IDs.
        token_log_probs: Per-token log probabilities from the generator.
        expected_answer: Expected answer for reward computation.
        reward: Scalar reward assigned by the grader.
    """

    policy_version: int
    prompt_token_ids: list[int]
    text: str
    token_ids: list[int]
    token_log_probs: list[float]
    expected_answer: str = ""
    reward: float = 0.0


# A Group holds all episodes that share the same prompt (the "G" in GRPO).
# Advantages are normalized within each group.
EpisodeGroup = list[Episode]
