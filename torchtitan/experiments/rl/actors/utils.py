# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass

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
    # plain tensor — materialize once here.
    if isinstance(logits, DTensor):
        logits = logits.to_local()
    shift_logits = logits[:, :-1, :].float()
    shift_targets = token_ids[:, 1:]
    logprobs = F.log_softmax(shift_logits, dim=-1)
    return logprobs.gather(2, shift_targets.unsqueeze(-1)).squeeze(-1)


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


@dataclass(frozen=True)
class LogprobVerification:
    """Local additive facts for the trainer-side reducer.

    All fields are local-rank values; the reducer SUMs/MAXes them across
    the loss mesh and divides by global token counts where applicable.
    """

    logprob_diff_sum: torch.Tensor  # SUM-lane: numerator for diff/mean
    logprob_diff_max: torch.Tensor  # MAX-lane
    num_tokens_different: torch.Tensor  # SUM-lane: numerator for ratio


def verify_logprob_identity(
    vllm_token_log_probs: list[list[float]],
    batch_token_log_probs: list[torch.Tensor],
    *,
    device: torch.device,
) -> LogprobVerification:
    """Generator/trainer logprob drift metrics.

    Inputs are response-only per-sample logprobs (no padding — assumes
    ``TrainBatch`` is varlen-packed).
    """
    if len(vllm_token_log_probs) != len(batch_token_log_probs):
        raise ValueError(
            f"verify_logprob_identity: sample count mismatch — "
            f"vllm={len(vllm_token_log_probs)}, "
            f"trainer={len(batch_token_log_probs)}"
        )

    if not vllm_token_log_probs:
        zero = torch.zeros((), dtype=torch.float32, device=device)
        return LogprobVerification(
            logprob_diff_sum=zero,
            logprob_diff_max=zero,
            num_tokens_different=zero,
        )

    vllm_flat = torch.tensor(
        [lp for sample in vllm_token_log_probs for lp in sample],
        dtype=torch.float32,
    )
    titan_flat = torch.cat([t.detach().cpu().float() for t in batch_token_log_probs])
    if vllm_flat.numel() != titan_flat.numel():
        raise ValueError(
            f"verify_logprob_identity: total token count mismatch — "
            f"vllm={vllm_flat.numel()}, trainer={titan_flat.numel()}"
        )
    if vllm_flat.numel() == 0:
        zero = torch.zeros((), dtype=torch.float32, device=device)
        return LogprobVerification(
            logprob_diff_sum=zero,
            logprob_diff_max=zero,
            num_tokens_different=zero,
        )

    diff = titan_flat - vllm_flat
    return LogprobVerification(
        logprob_diff_sum=diff.sum().to(device),
        logprob_diff_max=diff.abs().max().to(device),
        num_tokens_different=(vllm_flat != titan_flat)
        .sum()
        .to(device=device, dtype=torch.float32),
    )
