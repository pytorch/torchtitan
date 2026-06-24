# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch

from torchtitan.experiments.rl.observability import metrics as m


@dataclass(frozen=True, slots=True)
class RolloutID:
    """Identifies a rollout/turn end-to-end.

    Example:

        RolloutID(group_id=5, rollout_id=2, turn_id=0).to_string()
        # -> "group=5/rollout=2/turn=0"
        RolloutID(group_id=5, rollout_id=2, turn_id=0).to_string(include_turn=False)
        # -> "group=5/rollout=2"
    """

    group_id: int  # globally unique per GRPO group (e.g. group_id=5)
    rollout_id: int  # sibling index within the group (0..group_size-1)
    turn_id: int  # turn index within the rollout

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
    """Min policy version this turn was sampled under = the version at admission (a request only swaps
    to NEWER weights mid-decode, so admission is the min)."""
    max_policy_version: int
    """Max policy version this turn was sampled under = the version live when it finished. Equals
    min_policy_version unless a weight pull landed mid-generation (hotswap)."""
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
    """A single processed multi-turn rollout, ready for training.

    Example (single-turn): token_ids=[P, P, a, a], loss_mask=[0, 0, 1, 1],
                           logprobs=[0, 0, l, l], advantage=[0, 0, A, A]
    """

    min_policy_version: int
    """Min policy version across this packed segment's turns (the oldest; the off-policy filter reads it)."""
    max_policy_version: int
    """Max policy version across this packed segment's turns (newest weights any of its tokens saw)."""
    rollout_id: RolloutID  # turn_id = the turn this training_sample's segment begins at
    token_ids: list[int]  # [L] packed prompt + completions + env replies
    loss_mask: list[bool]  # [L] True on assistant tokens to train
    logprobs: list[float]  # [L] generator logprobs; 0.0 where loss_mask is False
    advantage: list[float]  # [L] advantage on assistant tokens, 0.0 elsewhere


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
