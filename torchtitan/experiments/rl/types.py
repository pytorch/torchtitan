# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch


@dataclass(kw_only=True, slots=True)
class Completion:
    """A single generated sequence from the generator.

    Pure generation artifact - no reward, no advantage. ``prompt_idx``
    is the position of the source prompt in the input ``prompts`` list.
    """

    policy_version: int
    prompt_idx: int
    text: str
    token_ids: list[int]
    token_logprobs: list[float]
    finish_reason: str | None = None
    """vLLM `CompletionOutput.finish_reason` ("stop" | "length" | "abort")"""


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
class TrainingBatch:
    """Packed training batch for the RL trainer.

    Per-token fields use "probability of current token" convention:
    ``logprobs[i] = log p(token_i | context up to i-1)``.
    Position 0 holds a dummy value (no prior context) and is excluded
    by ``loss_mask``.

    Cross-document logprobs at pack boundaries are incorrect but
    harmless: ``loss_mask`` is 0 at prompt positions (every episode
    has a prompt), so they never contribute to the loss.
    """

    token_ids: torch.Tensor  # [B, L]
    positions: torch.Tensor  # [B, L]
    generator_logprobs: torch.Tensor  # [B, L] — 0.0 for prompt/padding
    loss_mask: torch.Tensor  # [B, L] — 1.0 for response, 0.0 for prompt/padding
    advantages: torch.Tensor  # [B, L] — per-token, 0.0 for prompt/padding


@dataclass(frozen=True, slots=True)
class OptimStepOutput:
    """Result returned by ``PolicyTrainer.optim_step`` to the controller."""

    policy_version: int
    metrics: dict[str, float]
