# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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


def verify_logprob_identity(
    vllm_token_log_probs: list[list[float]],
    batch_token_log_probs: list[torch.Tensor],
) -> dict:
    """Check generator-vs-trainer logprob drift.

    Returns three public-named keys consumed by the trainer reducer:
    ``train/logprob_diff/mean`` (token-weighted mean log-ratio),
    ``train/logprob_diff/max`` (max abs log-ratio), and
    ``train/logprob/bitwise_identical`` (bool).

    Args:
        vllm_token_log_probs: Per-token log probs from vLLM (generator)
        batch_token_log_probs: Per-token log probs computed by the trainer model

    Returns:
        ``{"train/logprob_diff/mean": float,
           "train/logprob_diff/max": float,
           "train/logprob/bitwise_identical": bool}``
    """
    bitwise_identical = True
    all_log_ratios = []

    for vllm_lps, titan_lps in zip(vllm_token_log_probs, batch_token_log_probs):
        vllm_tensor = torch.tensor(vllm_lps, dtype=torch.float32)
        titan_tensor = titan_lps.detach().cpu().float()

        if not torch.equal(vllm_tensor, titan_tensor):
            bitwise_identical = False

        # Log ratio: log(pi_train / pi_generator) = logprob_train - logprob_generator.
        # Should be 0 when weights are identical (ratio = 1).
        all_log_ratios.append(titan_tensor - vllm_tensor)

    if all_log_ratios:
        combined_log_ratios = torch.cat(all_log_ratios)
        diff_mean = combined_log_ratios.mean().item()
        diff_max = combined_log_ratios.abs().max().item()
    else:
        diff_mean = 0.0
        diff_max = 0.0

    return {
        "train/logprob_diff/mean": diff_mean,
        "train/logprob_diff/max": diff_max,
        "train/logprob/bitwise_identical": bitwise_identical,
    }
