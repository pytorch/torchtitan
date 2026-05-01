# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import _DEFAULT_SPARSE_BLOCK_SIZE, BlockMask
from torchtitan.distributed.utils import is_in_batch_invariant_mode
from torchtitan.models.common.attention import create_attention_mask, VarlenMetadata


def _align_to(length: int, alignment: int) -> int:
    """Round up ``length`` to the next multiple of ``alignment``."""
    return ((length + alignment - 1) // alignment) * alignment


def create_flex_block_mask(
    seq_lens: list[int],
    device: torch.device,
    block_size: int | tuple[int, int] = _DEFAULT_SPARSE_BLOCK_SIZE,
) -> tuple[BlockMask, list[int]]:
    original_seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    maybe_padded_seq_lens = seq_lens
    maybe_padded_seq_lens_tensor = original_seq_lens_tensor

    if is_in_batch_invariant_mode():
        block_size_ = block_size if isinstance(block_size, int) else block_size[0]
        maybe_padded_seq_lens = [_align_to(sl, block_size_) for sl in seq_lens]
        maybe_padded_seq_lens_tensor = torch.tensor(
            maybe_padded_seq_lens, dtype=torch.int32, device=device
        )

    total = sum(maybe_padded_seq_lens)
    doc_ids = torch.repeat_interleave(
        torch.arange(len(seq_lens), dtype=torch.int32, device=device),
        maybe_padded_seq_lens_tensor,
    )
    offsets = F.pad(maybe_padded_seq_lens_tensor[:-1].cumsum(0), (1, 0))
    local_pos = torch.arange(
        total, dtype=torch.int32, device=device
    ) - torch.repeat_interleave(offsets, maybe_padded_seq_lens_tensor)

    if is_in_batch_invariant_mode():

        def mask_mod(b, h, q_idx, kv_idx):
            q_doc = doc_ids[q_idx]
            q_local = local_pos[q_idx]
            kv_local = local_pos[kv_idx]
            q_len = original_seq_lens_tensor[q_doc]
            return (
                (q_doc == doc_ids[kv_idx])
                & (q_local < q_len)
                & (kv_local < q_len)
                & (kv_local <= q_local)
            )

    else:

        def mask_mod(b, h, q_idx, kv_idx):
            return (doc_ids[q_idx] == doc_ids[kv_idx]) & (
                local_pos[kv_idx] <= local_pos[q_idx]
            )

    block_mask = create_attention_mask(
        mask_mod,
        B=1,
        H=None,
        Q_LEN=total,
        KV_LEN=total,
        BLOCK_SIZE=block_size,
        separate_full_blocks=not is_in_batch_invariant_mode(),
    )
    return block_mask, maybe_padded_seq_lens


def pad_to_block_aligned(
    flat: torch.Tensor,
    seq_lens: list[int],
    maybe_padded_seq_lens: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad packed sequences to block-aligned lengths.

    Returns ``(padded_ids, positions)`` each with shape ``(1, total_padded)``.
    """
    parts, pos_parts = [], []
    offset = 0
    for sl, psl in zip(seq_lens, maybe_padded_seq_lens):
        padded = torch.zeros(psl, dtype=flat.dtype, device=device)
        padded[:sl] = flat[offset : offset + sl]
        parts.append(padded)
        pos_parts.append(torch.arange(psl, device=device))
        offset += sl
    return torch.cat(parts).unsqueeze(0), torch.cat(pos_parts).unsqueeze(0)


def unpad_from_block_aligned(
    padded: torch.Tensor,
    seq_lens: list[int],
    maybe_padded_seq_lens: list[int],
) -> torch.Tensor:
    """Extract original-length slices from a block-padded ``(1, total_padded, ...)`` tensor."""
    parts = []
    offset = 0
    for sl, psl in zip(seq_lens, maybe_padded_seq_lens):
        parts.append(padded[0, offset : offset + sl])
        offset += psl
    return torch.cat(parts).unsqueeze(0)


# TODO We should either unify all the mask creation for RL, or move them to a
#      single file.
def create_varlen_metadata(seq_lens: list[int], device: torch.device) -> VarlenMetadata:
    """Build VarlenMetadata from sequence lengths.

    Example:
        seq_lens = [3, 5, 2]
        -> cu_seqs = [0, 3, 8, 10], max_len = 5
    """
    cu_seqs = torch.tensor(
        [0] + list(itertools.accumulate(seq_lens)), dtype=torch.int32, device=device
    )
    max_len = max(seq_lens)
    return VarlenMetadata(
        cu_seq_q=cu_seqs, cu_seq_k=cu_seqs, max_q=max_len, max_k=max_len
    )


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


def create_positions_from_seq_lens(
    seq_lens: list[int], device: torch.device
) -> torch.Tensor:
    """Build a ``[1, total_tokens]`` positions tensor that resets at each sequence boundary.

    Example:
        seq_lens = [3, 5, 2]
        -> positions = [[0, 1, 2, 0, 1, 2, 3, 4, 0, 1]]
    """
    return torch.cat([torch.arange(l, device=device) for l in seq_lens]).unsqueeze(0)


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
