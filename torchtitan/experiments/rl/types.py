# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch

from torchtitan.experiments.rl.observability import metrics as m


@dataclass(frozen=True, slots=True)
class RolloutTurnID:
    """A turn's id: (group, sibling rollout, turn index); renders to the generator request_id.

    Example:

        RolloutTurnID(group_id=5, rollout_id=2, turn_id=0).to_string()
        # -> "group=5/rollout=2/turn=0"
        RolloutTurnID(group_id=5, rollout_id=2, turn_id=0).to_string(include_turn=False)
        # -> "group=5/rollout=2"
    """

    group_id: int
    """Globally-unique GRPO group id; siblings share it (the sticky-routing key, sans turn)."""
    rollout_id: int
    """Sibling index within the group (0..num_samples_per_prompt-1)."""
    turn_id: int
    """Turn index within the rollout; for a TrainingSample, the turn where begins.
    This is not 0 when a single rollout is split into multiple training samples."""

    def to_string(self, *, include_turn: bool = True) -> str:
        base = f"group={self.group_id}/rollout={self.rollout_id}"
        return f"{base}/turn={self.turn_id}" if include_turn else base


@dataclass(kw_only=True, slots=True)
class Completion:
    """A single generated sequence from the generator.

    Example:

        Completion(min_policy_version=7, max_policy_version=7, request_id="r0", token_ids=[12, 9],
                   token_logprobs=[-0.2, -1.1], finish_reason="stop",
                   metrics=[Metric("generator/queue_time_ms", ...)])
    """

    min_policy_version: int
    """Oldest policy version among this turn's decode."""
    max_policy_version: int
    """Newest policy version among this turn's decode."""
    # TODO(async-rl): for exact per-token version attribution, switch the engine to
    #   RequestOutputKind.CUMULATIVE and record (start_token, version) boundaries; today we keep only
    #   the per-turn min (min_policy_version) / max (max_policy_version).
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


@dataclass(kw_only=True, slots=True)
class TrainingSample:
    """A trainable token sequence from a rollout.

    Example:
        # Turn 0: prompt P0 -> assistant A0 -> env reply E0
        # Turn 1: prompt [P0, A0, E0] -> assistant A1
        TrainingSample(
            rollout_id=RolloutTurnID(group_id=3, rollout_id=1, turn_id=1),
            min_policy_version=7,
            max_policy_version=9,            # weights updated during rollout
            token_ids=P0 + A0 + E0 + A1,
            loss_mask=[False]*len(P0) + [True]*len(A0) + [False]*len(E0) + [True]*len(A1),
            logprobs=[0.0]*len(P0) + logprobs_A0 + [0.0]*len(E0) + logprobs_A1,
            advantage=[0.0]*len(P0) + [adv]*len(A0) + [0.0]*len(E0) + [adv]*len(A1),
        )
    """

    min_policy_version: int
    """Oldest policy version among this branch's trained turns."""
    max_policy_version: int
    """Newest policy version among this branch's trained turns."""
    rollout_id: RolloutTurnID
    """This sample identifier."""
    token_ids: list[int]
    """[L] packed prompt + completions + env replies."""
    loss_mask: list[bool]
    """[L] True on assistant tokens to train."""
    logprobs: list[float]
    """[L] generator logprobs; 0.0 where loss_mask is False."""
    advantage: list[float]
    """[L] advantage on assistant tokens, 0.0 elsewhere."""


@dataclass(frozen=True, slots=True)
class TrainingSampleGroup:
    """The training samples + metrics built from one rollout group.

    Example:
        TrainingSampleGroup(group_id=3, training_samples=[], metrics=[failure_metric])
        # -> failed / filtered / zero-std group; metrics still reach the trainer logger
    """

    group_id: int
    training_samples: list[TrainingSample]
    metrics: list[m.Metric]


@dataclass(kw_only=True, slots=True)
class TrainingMicrobatch:
    """Packed training batch for the RL trainer.

    Each training_sample's raw tokens (length N) are split into
    ``token_ids = raw[:-1]`` and ``labels = raw[1:]`` (both length
    N-1), matching the pre-training dataloader convention.
    """

    token_ids: torch.Tensor  # [B, L]
    labels: torch.Tensor  # [B, L]
    positions: torch.Tensor  # [B, L]
    generator_logprobs: torch.Tensor  # [B, L]
    loss_mask: torch.Tensor  # [B, L]
    advantages: torch.Tensor  # [B, L]


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    """Packed microbatches for one optimizer step.

    Example:
        # 5 training samples, effective length 5 each; seq_len=10, local_batch_size=2, dp_degree=1
        # next-fit rows -> [[s5, s5], [s5, s5], [s5]] = 3 rows; rows_per_microbatch = 2 * 1 = 2
        # -> 2 microbatches (3 rows padded to 4 with one pad-only row):
        #    microbatches = [[TrainingMicrobatch(token_ids=[2, 10])],
        #                    [TrainingMicrobatch(token_ids=[2, 10])]]   # mb1 = 1 real row + 1 pad row
        # num_global_valid_tokens = count of loss_mask=True tokens across the 5 samples
    """

    microbatches: list[list[TrainingMicrobatch]]  # [num_microbatches][dp_degree]
    num_global_valid_tokens: int
    metrics: list[m.Metric]
    # one per packed training_sample; trainer computes policy_age at consume time
    min_policy_versions: list[int]


@dataclass(frozen=True, slots=True)
class OptimStepOutput:
    """Result returned by `PolicyTrainer.optim_step` to the controller."""

    policy_version: int
    metrics: dict[str, float]
