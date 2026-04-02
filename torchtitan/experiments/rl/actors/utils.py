# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from torchtitan.models.common.attention import VarlenMetadata


# TODO We should either unify all the mask creation for RL, or move them to a
#      single file.
def build_varlen_metadata(
    input_sequences: list[tuple[torch.Tensor, int, int]], device: torch.device
) -> VarlenMetadata:
    """Build VarlenMetadata for all sequences in a batch."""
    cu_seqs = torch.cumsum(
        torch.tensor(
            [0] + [token_ids.shape[0] for token_ids, _, _ in input_sequences],
            dtype=torch.int32,
            device=device,
        ),
        0,
        dtype=torch.int32,
    )
    max_len = max(token_ids.shape[0] for token_ids, _, _ in input_sequences)
    return VarlenMetadata(
        cu_seq_q=cu_seqs, cu_seq_k=cu_seqs, max_q=max_len, max_k=max_len
    )


def extract_logprobs_batched(
    logits: torch.Tensor, token_ids: torch.Tensor
) -> torch.Tensor:
    """Extract per-token logprobs from batched logits.

    Uses the shifted-by-1 pattern: logits[t] predicts token_ids[t+1].
    Position 0 is set to 0.0 since there is no prediction for the first token.

    Args:
        logits: [B, L, V] model output logits.
        token_ids: [B, L] input token IDs.

    Returns:
        [B, L] per-token logprobs, aligned with input positions.
    """
    shift_logits = logits[:, :-1, :].float()
    shift_targets = token_ids[:, 1:]
    logprobs = F.log_softmax(shift_logits, dim=-1)
    token_logprobs = logprobs.gather(2, shift_targets.unsqueeze(-1)).squeeze(-1)
    return F.pad(token_logprobs, (1, 0), value=0.0)


def build_response_mask(
    prompt_lens: torch.Tensor,
    response_lens: torch.Tensor,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Build a binary mask selecting response tokens only.

    Args:
        prompt_lens: [B] length of prompt per sequence.
        response_lens: [B] length of response per sequence.
        seq_len: total padded sequence length L.
        device: target device.

    Returns:
        [B, L] float mask, 1.0 for response tokens, 0.0 for prompt/padding.
    """
    positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, L]
    start = prompt_lens.unsqueeze(1)  # [B, 1]
    end = start + response_lens.unsqueeze(1)  # [B, 1]
    return ((positions >= start) & (positions < end)).float()


def verify_logprob_identity(
    vllm_token_log_probs: list[list[float]],
    batch_token_log_probs: list[torch.Tensor],
) -> dict:
    """
    Check if vLLM log probs and computed log probs are bit-wise identical,
    and compute the log ratio (train/generator) between them.

    Args:
        vllm_token_log_probs: Per-token log probs from vLLM (generator)
        batch_token_log_probs: Per-token log probs computed by the trainer model

    Returns:
        Verification result dict with identity status, delta info, and log ratio stats
    """
    result = {
        "logprob_bitwise_identical": True,
        "num_samples_checked": len(vllm_token_log_probs),
        "total_tokens_checked": 0,
        "num_tokens_different": 0,
        "logprob_max_delta": 0.0,
        "avg_delta": 0.0,
        "logprob_diff_mean": 0.0,
        "logprob_diff_max": 0.0,
    }

    all_deltas = []
    all_log_ratios = []

    for vllm_lps, titan_lps in zip(vllm_token_log_probs, batch_token_log_probs):
        # Convert vLLM log probs to tensor
        vllm_tensor = torch.tensor(vllm_lps, dtype=torch.float32)
        # Convert titan log probs to float32 for comparison
        titan_tensor = titan_lps.detach().cpu().float()

        num_tokens = len(vllm_lps)
        result["total_tokens_checked"] += num_tokens

        # Check bitwise identity
        bitwise_match = torch.equal(vllm_tensor, titan_tensor)

        if not bitwise_match:
            result["logprob_bitwise_identical"] = False
            num_different = (vllm_tensor != titan_tensor).sum().item()
            result["num_tokens_different"] += num_different
            deltas = (vllm_tensor - titan_tensor).abs()
            all_deltas.append(deltas)

        # Log ratio: log(pi_train / pi_generator) = logprob_train - logprob_generator
        # Should be 0 when weights are identical (ratio = 1)
        all_log_ratios.append(titan_tensor - vllm_tensor)

    # Compute aggregate delta stats
    if all_deltas:
        combined_deltas = torch.cat(all_deltas)
        result["logprob_max_delta"] = combined_deltas.max().item()
        result["avg_delta"] = combined_deltas.mean().item()

    # Compute log ratio stats
    if all_log_ratios:
        combined_log_ratios = torch.cat(all_log_ratios)
        result["logprob_diff_mean"] = combined_log_ratios.mean().item()
        result["logprob_diff_max"] = combined_log_ratios.abs().max().item()

    return result
