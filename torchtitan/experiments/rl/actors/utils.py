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
    # plain tensor - materialize once here.
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


@dataclass(frozen=True, slots=True)
class LogprobVerificationOutput:
    """Generator vs trainer drift metrics, pre-normalized by global tokens."""

    logprob_diff_mean: torch.Tensor
    logprob_diff_max: torch.Tensor
    ratio_tokens_different: torch.Tensor


def verify_logprob_identity(
    generator_token_logprobs: list[list[float]],
    trainer_token_logprobs: list[torch.Tensor],
    *,
    num_global_valid_tokens: torch.Tensor,
    device: torch.device,
) -> LogprobVerificationOutput:
    """Compare generator and trainer response-token logprobs.

    Returns:
        metrics pre-normalized by num_global_valid_tokens for later reduction.
    """
    with torch.no_grad():
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
            return LogprobVerificationOutput(zero, zero, zero)

        # Aggregate shape check prevents PyTorch broadcast from silently
        # masking a shape mismatch as wrong numbers.
        if trainer_flat.shape != generator_flat.shape:
            raise ValueError(
                f"verify_logprob_identity shape mismatch: "
                f"generator={generator_flat.shape}, trainer={trainer_flat.shape}"
            )

        diff = trainer_flat - generator_flat
        return LogprobVerificationOutput(
            logprob_diff_mean=diff.sum() / num_global_valid_tokens,
            logprob_diff_max=diff.abs().max(),
            ratio_tokens_different=(
                (generator_flat != trainer_flat).sum().to(torch.float32)
                / num_global_valid_tokens
            ),
        )
