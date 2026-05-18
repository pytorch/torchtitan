# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch
import torch.nn.functional as F



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


def verify_logprob_identity(
    policy_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict:
    """Compare trainer logprobs against generator ref_logprobs on response tokens.

    All inputs are [B, L] tensors. Only positions where response_mask == 1
    are compared.

    Returns:
        Dict with bitwise_identical, max_delta, diff_mean, diff_max, tokens_checked.
    """
    mask = response_mask.bool()
    num_tokens = mask.sum().item()

    if num_tokens == 0:
        return {
            "logprob_bitwise_identical": True,
            "logprob_max_delta": 0.0,
            "logprob_diff_mean": 0.0,
            "logprob_diff_max": 0.0,
            "total_tokens_checked": 0,
        }

    policy_response = policy_logprobs[mask]
    ref_response = ref_logprobs[mask]

    bitwise_identical = torch.equal(policy_response, ref_response)
    deltas = (policy_response - ref_response).abs()
    log_ratio = policy_response - ref_response

    return {
        "logprob_bitwise_identical": bitwise_identical,
        "logprob_max_delta": deltas.max().item(),
        "logprob_diff_mean": log_ratio.mean().item(),
        "logprob_diff_max": log_ratio.abs().max().item(),
        "total_tokens_checked": int(num_tokens),
    }
