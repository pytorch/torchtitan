# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class Completion:
    """Output of the Generator. Prompt + model response + logprobs."""

    prompt_tokens: list[int]
    response_tokens: list[int]
    logprobs: list[float]
    group_id: str
    text: str
    policy_version: int = 0
    metadata: dict[str, Any] | None = None


@dataclass
class ScoredCompletion:
    """Completion with a reward from the grader."""

    completion: Completion
    reward: float


@dataclass
class Episode:
    """Ready for training. Flat structure for ergonomic access in _collate.

    Built by the orchestrator from ScoredCompletion + computed advantage.
    """

    prompt_tokens: list[int]
    response_tokens: list[int]
    logprobs: list[float]
    group_id: str
    text: str
    reward: float
    advantage: float
    policy_version: int = 0


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
    old_logprobs: torch.Tensor  # [B, L] (0 for prompt/padding positions)
    policy_version: int
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
