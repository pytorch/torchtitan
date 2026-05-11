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
    """Local generator/trainer logprob comparison metrics."""

    logprob_diff_sum: torch.Tensor
    logprob_diff_max: torch.Tensor
    num_tokens_different: torch.Tensor


def verify_logprob_identity(
    generator_token_logprobs: list[list[float]],
    trainer_token_logprobs: list[torch.Tensor],
    *,
    device: torch.device,
) -> LogprobVerificationOutput:
    """Compare generator and trainer response-token logprobs.

    Raises:
        ValueError: If sample counts or per-sample token counts differ.
    """
    if len(generator_token_logprobs) != len(trainer_token_logprobs):
        raise ValueError(
            "verify_logprob_identity sample count mismatch: "
            f"generator={len(generator_token_logprobs)}, "
            f"trainer={len(trainer_token_logprobs)}"
        )

    generator_tensors: list[torch.Tensor] = []
    trainer_tensors: list[torch.Tensor] = []
    for sample_idx, (generator_values, trainer_values) in enumerate(
        zip(generator_token_logprobs, trainer_token_logprobs, strict=True)
    ):
        if len(generator_values) != trainer_values.numel():
            raise ValueError(
                "verify_logprob_identity token count mismatch for sample "
                f"{sample_idx}: generator={len(generator_values)}, "
                f"trainer={trainer_values.numel()}"
            )
        generator_tensors.append(
            torch.tensor(generator_values, dtype=torch.float32, device=device)
        )
        trainer_tensors.append(
            trainer_values.detach().to(device=device, dtype=torch.float32)
        )

    generator_flat = torch.cat(generator_tensors)
    trainer_flat = torch.cat(trainer_tensors)
    diff = trainer_flat - generator_flat
    return LogprobVerificationOutput(
        logprob_diff_sum=diff.sum(),
        logprob_diff_max=diff.abs().max(),
        num_tokens_different=(generator_flat != trainer_flat).sum().to(torch.float32),
    )
