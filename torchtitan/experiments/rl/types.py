# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Legacy type -- used by existing code during migration. Will be removed once
# all callers have been updated to use the new stage-typed data model.
# ---------------------------------------------------------------------------


@dataclass
class LegacyEpisode:
    """A single prompt + completion pair with reward and advantage.

    The generator creates LegacyEpisodes (with group_id, no reward yet).
    The grader fills in the reward.
    The controller computes advantages across episodes sharing a group_id.
    The trainer consumes the final LegacyEpisodes with advantages set.
    """

    policy_version: int
    prompt_token_ids: list[int]
    text: str
    token_ids: list[int]
    logprobs: list[float]
    expected_answer: str = ""
    reward: float = 0.0
    group_id: str = ""
    advantage: float = 0.0


# Keep backward-compatible alias so existing imports still work.
Episode = LegacyEpisode


# ---------------------------------------------------------------------------
# New stage-typed data model.
#
# Data flows through the RL pipeline as:
#   Generator -> Completion
#   Environment/Grader -> ScoredCompletion
#   Advantage computation -> RLEpisode
#   Trainer <- RLEpisode (via forward_backward)
# ---------------------------------------------------------------------------


@dataclass
class Completion:
    """Output of the Generator. Prompt + model response + logprobs."""

    prompt_tokens: list[int]
    response_tokens: list[int]
    logprobs: list[float]
    group_id: str
    text: str
    metadata: dict[str, Any] | None = None


@dataclass
class ScoredCompletion:
    """Output of Environment scoring. Wraps a Completion with a reward."""

    completion: Completion
    reward: float


@dataclass
class RLEpisode:
    """Ready for training. Wraps a ScoredCompletion with an advantage."""

    scored_completion: ScoredCompletion
    advantage: float


@dataclass
class TrainBatch:
    """Input to Trainer.forward_backward().

    One TrainBatch per DP rank. The controller's collate function
    pre-shards data into a list[TrainBatch] before sending to the trainer.
    All sequences are right-padded to the same length within the batch.
    """

    token_ids: torch.Tensor  # [B, L] padded (prompt + response + padding)
    prompt_lens: torch.Tensor  # [B]
    response_lens: torch.Tensor  # [B]
    advantages: torch.Tensor  # [B]
    ref_logprobs: torch.Tensor  # [B, L] (0 for prompt/padding positions)
    pad_token_id: int


@dataclass
class ForwardBackwardResult:
    """Output of Trainer.forward_backward()."""

    loss: float
    metrics: dict[str, float]


@dataclass
class OptimStepResult:
    """Output of Trainer.optim_step()."""

    grad_norm: float
    policy_version: int
