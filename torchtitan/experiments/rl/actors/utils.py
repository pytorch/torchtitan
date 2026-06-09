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


@sl.log_trace_span("extract_response_logprobs")
def extract_response_logprobs(
    packed_logprobs: torch.Tensor,
    seq_lens: list[int],
    prompt_lens: list[int],
    response_lens: list[int],
) -> list[torch.Tensor]:
    """Extract per-sample response logprobs from packed logprobs."""
    seq_start = 0
    result = []
    for i in range(len(seq_lens)):
        # Logprobs are shifted: position j holds logprob of token j+1,
        # so response start (seq_start + prompt_len) maps to index
        # (seq_start + prompt_len - 1) in the logprobs tensor.
        s = seq_start + prompt_lens[i] - 1
        e = s + response_lens[i]
        result.append(packed_logprobs[0, s:e])
        seq_start += seq_lens[i]
    return result


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
    generator_token_logprobs: list[list[float]],
    trainer_token_logprobs: list[torch.Tensor],
    *,
    num_global_valid_tokens: torch.Tensor,
    device: torch.device,
) -> PartialLogprobDrift:
    """Compute per-rank drift between generator and trainer logprobs.

    Args:
        generator_token_logprobs (list[list[float]]): generator-side per-token logprobs, shaped
            `[num_episodes_local][response_len_i]`.
        trainer_token_logprobs (list[torch.Tensor]): Trainer-side per-token logprobs, one
            GPU tensor per episode, each of shape `[response_len_i]`.
        num_global_valid_tokens (torch.Tensor): Scalar tensor holding global token count
             across DP ranks. Used to normalize the output metrics.
        device: Device to use for tensor allocation, so metrics are ready for
            reduction across loss_mesh.

    Returns:
        PartialLogprobDrift.
    """
    # Each tensor has a different number of tokens, so we flatten them.
    generator_flat = torch.as_tensor(
        [v for sample in generator_token_logprobs for v in sample],
        dtype=torch.float32,
        device=device,
    )
    trainer_flat = torch.cat(trainer_token_logprobs).to(
        device=device, dtype=torch.float32
    )

    if generator_flat.numel() == 0:
        zero = torch.zeros((), dtype=torch.float32, device=device)
        return PartialLogprobDrift(zero, zero, zero)

    # 1e-6 threshold ignores bf16-quantization-level diffs
    diff = trainer_flat - generator_flat
    return PartialLogprobDrift(
        logprob_diff_mean=diff.sum() / num_global_valid_tokens,
        logprob_diff_max=diff.abs().max(),
        ratio_tokens_different=(diff.abs() > 1e-6).sum() / num_global_valid_tokens,
    )
