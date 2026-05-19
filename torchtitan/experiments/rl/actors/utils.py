# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trainer-side helpers: per-token logprob computation + drift verification.

All extraction is mask-based now: the trainer reads ``loss_mask`` (1
on tokens to learn from, 0 elsewhere) instead of doing arithmetic on
``(prompt_lens, response_lens)``. This unblocks multi-turn rollouts
where there's no single prompt/response boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from torchtitan.observability import structured_logger as sl


@sl.log_trace_span("compute_logprobs")
def compute_logprobs(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Per-token log p(next | <=t) from logits + targets.

    Args:
        logits: ``[B, T, V]`` next-token logits (typically B=1 for varlen).
        token_ids: ``[B, T]`` token sequence.

    Returns:
        ``[B, T - 1]`` per-token logprobs. Position ``t`` is
        ``log p(token_{t+1} | token_{<=t})``. Cross-sample-boundary
        positions in a packed batch are present but unused; the loss
        mask zeros them out.
    """
    from torch.distributed.tensor import DTensor

    # Config-based TP can return logits as a Replicate DTensor.
    # Downstream code does plain-tensor indexing; materialize once.
    if isinstance(logits, DTensor):
        # TODO: pass ``grad_placements=[Replicate(), ...]`` to make the
        # autograd contract explicit (see .claude/rules/distributed.md).
        logits = logits.to_local()
    shift_logits = logits[:, :-1, :].float()
    shift_targets = token_ids[:, 1:]
    logprobs = F.log_softmax(shift_logits, dim=-1)
    return logprobs.gather(2, shift_targets.unsqueeze(-1)).squeeze(-1)


@dataclass(frozen=True, slots=True)
class PartialLogprobDrift:
    """Per-rank generator-vs-trainer drift; ready for cross-rank reduction."""

    logprob_diff_mean: torch.Tensor  # SUM-reduce
    logprob_diff_max: torch.Tensor  # MAX-reduce
    ratio_tokens_different: torch.Tensor  # SUM-reduce


@torch.no_grad()
@sl.log_trace_span("verify_logprob_identity")
def verify_logprob_identity(
    *,
    behavior_logprobs: torch.Tensor,
    policy_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    num_global_valid_tokens: torch.Tensor,
) -> PartialLogprobDrift:
    """Mask-aware drift between rollout-time and trainer-time logprobs.

    Args:
        behavior_logprobs: ``[1, T - 1]`` rollout-time logprobs (shifted
            to align with the model's predicted positions; 0 where the
            mask is 0).
        policy_logprobs: ``[1, T - 1]`` trainer-time logprobs.
        loss_mask: ``[1, T - 1]`` 1 on loss positions.
        num_global_valid_tokens: scalar; sum of ``loss_mask`` across all
            DP ranks. Used to normalize the SUM-reduced metrics.

    Returns:
        :class:`PartialLogprobDrift` with scalars on the trainer device.
    """
    diff = (policy_logprobs - behavior_logprobs) * loss_mask
    abs_diff = diff.abs()
    above_eps = (abs_diff > 1e-6).float() * loss_mask
    return PartialLogprobDrift(
        logprob_diff_mean=diff.sum() / num_global_valid_tokens,
        logprob_diff_max=abs_diff.max(),
        ratio_tokens_different=above_eps.sum() / num_global_valid_tokens,
    )
