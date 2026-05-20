# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchtitan.observability import structured_logger as sl


@sl.log_trace_span("compute_logprobs")
def compute_logprobs(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Compute per-token logprobs from logits.

    Returns logprobs for positions 1..N (the predicted tokens).
    Output shape is ``[batch, seq_len - 1]``.
    """
    from torch.distributed.tensor import DTensor

    # Config-based TP returns logits as a Replicate DTensor. Downstream RL
    # code (gather with plain-tensor indices, slicing per-sample) expects a
    # plain tensor - materialize once here.
    if isinstance(logits, DTensor):
        # TODO: pass `grad_placements=[Replicate(), ...]` to make the autograd
        # contract explicit (see .claude/rules/distributed.md).
        logits = logits.to_local()
    shift_logits = logits[:, :-1, :].float()
    shift_targets = token_ids[:, 1:]
    logprobs = F.log_softmax(shift_logits, dim=-1)
    return logprobs.gather(2, shift_targets.unsqueeze(-1)).squeeze(-1)


@dataclass(frozen=True, slots=True)
class PartialLogprobDrift:
    """Per-rank generator-vs-trainer logprob drift awaiting reduction across the loss-mesh.

    Args:
        logprob_diff_mean: Scalar tensor; To be sum-reduced.
        logprob_diff_max: Scalar tensor; To be max-reduced.
        ratio_tokens_different: Scalar tensor; To be sum-reduced.
    """

    logprob_diff_mean: torch.Tensor
    logprob_diff_max: torch.Tensor
    ratio_tokens_different: torch.Tensor


@torch.no_grad()
@sl.log_trace_span("verify_logprob_identity")
def verify_logprob_identity(
    ref_logprobs: torch.Tensor,
    policy_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    num_global_valid_tokens: torch.Tensor,
) -> PartialLogprobDrift:
    """Compute per-rank drift between generator and trainer logprobs.

    Args:
        ref_logprobs: [B, L] reference (generator) logprobs from TrainBatch.
        policy_logprobs: [B, L] trainer-computed logprobs.
        response_mask: [B, L] binary mask; 1.0 for response tokens.
        num_global_valid_tokens: Scalar tensor holding global token count
            across DP ranks. Used to normalize the output metrics.

    Returns:
        PartialLogprobDrift.
    """
    mask = response_mask.bool()
    ref_flat = ref_logprobs[mask].float()
    policy_flat = policy_logprobs[mask].float()

    if ref_flat.numel() == 0:
        zero = torch.zeros((), dtype=torch.float32, device=ref_logprobs.device)
        return PartialLogprobDrift(zero, zero, zero)

    diff = policy_flat - ref_flat
    return PartialLogprobDrift(
        logprob_diff_mean=diff.sum() / num_global_valid_tokens,
        logprob_diff_max=diff.abs().max(),
        ratio_tokens_different=(diff.abs() > 1e-6).sum() / num_global_valid_tokens,
    )
