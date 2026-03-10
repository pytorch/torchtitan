# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field


@dataclass
class Episode:
    """
    Data from one prompt in a generation batch.
    Each Episode holds a single prompt and its group of completions.

    Attributes:
        policy_version: Version of policy that produced this episode
        prompt_token_ids: Token IDs for this prompt (no duplication)
        expected_answer: Expected answer for reward computation
        completions: List of Completion objects (len=num_samples_per_prompt)
    """

    @dataclass
    class Completion:
        """A single completion for a prompt."""

        text: str
        token_ids: list[int]
        token_log_probs: list[float]
        reward: float = 0.0

    policy_version: int
    prompt_token_ids: list[int]
    expected_answer: str = ""
    completions: list[Completion] = field(default_factory=list)
