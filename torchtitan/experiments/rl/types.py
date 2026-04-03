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
    """Output of the generator."""

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
    """Training episode consumed by the trainer."""

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
    """Pre-collated batch for one trainer DP rank."""

    token_ids: torch.Tensor
    prompt_lens: torch.Tensor
    response_lens: torch.Tensor
    advantages: torch.Tensor
    token_log_probs: torch.Tensor
    pad_token_id: int


@dataclass
class ForwardBackwardResult:
    loss: float
    metrics: dict[str, float]


@dataclass
class OptimStepResult:
    grad_norm: float
    policy_version: int
