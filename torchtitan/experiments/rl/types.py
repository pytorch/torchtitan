# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch

from torchtitan.experiments.rl.observability import metrics as m


@dataclass(kw_only=True, slots=True)
class Completion:
    """A single generated sequence from the generator.

    Example:

        Completion(policy_version=7, request_id="r0", token_ids=[12, 9],
                   token_logprobs=[-0.2, -1.1], finish_reason="stop",
                   metrics=[Metric("generator/queue_time_ms", ...)])
    """

    # TODO(async): change `int` to per-token version intervals when hot-swap in async is enabled.
    policy_version: int
    request_id: str
    """Echoes the id the caller passed to `generate`, so callers can validate
    ordered completions or map by id."""
    token_ids: list[int]
    token_logprobs: list[float]
    finish_reason: str | None = None
    """vLLM `CompletionOutput.finish_reason` ("stop" | "length" | "abort")"""

    metrics: list[m.Metric] = field(default_factory=list)
    """Per-generation metrics measured by the generator (latencies); the
    controller attaches them to the rollout turn."""

    version_intervals: list[tuple[int, int]] = field(default_factory=list)
    """Policy-version boundaries `(start_token_index, version)`; today the single conservative
    interval `[(0, admission_version)]` — the version this completion was sampled at."""


@dataclass(kw_only=True, slots=True)
class Episode:
    """A single processed multi-turn rollout, ready for training.

    Example (single-turn): token_ids=[P, P, a, a], loss_mask=[0, 0, 1, 1],
                           logprobs=[0, 0, l, l], advantage=[0, 0, A, A]
    """

    policy_version: int
    sample_id: str
    token_ids: list[int]  # [L] packed prompt + completions + env replies
    loss_mask: list[bool]  # [L] True on assistant tokens to train
    logprobs: list[float]  # [L] generator logprobs; 0.0 where loss_mask is False
    advantage: list[float]  # [L] advantage on assistant tokens, 0.0 elsewhere

    version_intervals: list[tuple[int, int]] = field(default_factory=list)
    """Policy-version boundaries `(start_token_index, version)` in this packed sequence, one per
    turn's completion; the off-policy filter reads the earliest version (see `earliest_version`)."""


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
