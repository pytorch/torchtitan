# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch


@dataclass(kw_only=True, slots=True)
class Completion:
    """A single generated sequence from the generator."""

    policy_version: int
    request_id: str
    """Echoes the id the caller passed to `generate`, so callers can validate
    ordered completions or map by id."""
    token_ids: list[int]
    token_logprobs: list[float]
    finish_reason: str | None = None
    """vLLM `CompletionOutput.finish_reason` ("stop" | "length" | "abort")"""


# TODO: rename `Episode` -> `TrainingSample`
# and `rollout_to_episode` -> `rollout_to_training_sample`
@dataclass(kw_only=True, slots=True)
class Episode:
    """Training sample: flattened Rollout turns + GRPO advantage,
    ready for collation into a batch."""

    policy_version: int
    sample_id: str
    prompt_token_ids: list[int]
    completion_text: str
    completion_token_ids: list[int]
    completion_logprobs: list[float]
    reward: float
    advantage: float


@dataclass(kw_only=True, slots=True)
class TrainingBatch:
    """Packed training batch for the RL trainer.

    Each episode's raw tokens (length N) are split into
    ``token_ids = raw[:-1]`` and ``labels = raw[1:]`` (both length
    N-1), matching the pre-training dataloader convention.
    """

    token_ids: torch.Tensor  # [B, L]
    labels: torch.Tensor  # [B, L]
    positions: torch.Tensor  # [B, L]
    # TODO(naming): rename generator_logprobs -> old_logprobs (PPO π_old) vs policy_logprobs,
    # incl. GRPOLoss/trainer/batcher.
    generator_logprobs: torch.Tensor  # [B, L]
    loss_mask: torch.Tensor  # [B, L]
    advantages: torch.Tensor  # [B, L]


@dataclass(frozen=True, slots=True)
class OptimStepOutput:
    """Result returned by `PolicyTrainer.optim_step` to the controller."""

    policy_version: int
    metrics: dict[str, float]
