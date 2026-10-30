# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from types import SimpleNamespace

import pytest
import torch

from torchtitan.rl.generator.vllm.mla import (
    TorchTitanMLAImpl,
    TorchTitanMLAPrefillBackend,
)
from torchtitan.tools.utils import has_cuda_capability

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not has_cuda_capability(10, 0),
    reason="FA4 absorbed MLA requires SM100",
)

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
SCALE = 1 / math.sqrt(192)


def _offsets(lengths: list[int]) -> torch.Tensor:
    return torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
    )


def _attention_reference(q_HQD, k_KD_or_HKD, v_KV_or_HKV, mask_QK):
    logits_HQK = SCALE * q_HQD.float() @ k_KD_or_HKD.float().transpose(-1, -2)
    logits_HQK = logits_HQK.masked_fill(~mask_QK, float("-inf"))
    out_HQV = torch.softmax(logits_HQK, dim=-1) @ v_KV_or_HKV.float()
    return out_HQV.transpose(0, 1), logits_HQK.logsumexp(-1)


def test_mla_decode_attends_to_latent_cache():
    """forward_mqa reads vLLM's [kv_c_normed | k_pe] cache as latent keys/values and rope keys."""
    query_lengths, key_lengths = [1, 3, 1, 2], [40, 300, 1, 129]
    num_heads, block_size = 16, 64
    device = torch.device("cuda")
    blocks = [-(-length // block_size) for length in key_lengths]
    cache = torch.randn(
        sum(blocks) + 2,
        block_size,
        KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    order = torch.randperm(cache.shape[0])
    block_table = torch.zeros(len(blocks), max(blocks), dtype=torch.int32)
    start = 0
    for sequence, count in enumerate(blocks):
        block_table[sequence, :count] = order[start : start + count]
        start += count
    q = torch.randn(
        sum(query_lengths),
        num_heads,
        KV_LORA_RANK + QK_ROPE_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    metadata = SimpleNamespace(
        decode=SimpleNamespace(
            block_table=block_table.to(device),
            query_start_loc=_offsets(query_lengths).to(device),
            seq_lens=torch.tensor(key_lengths, dtype=torch.int32, device=device),
            max_query_len=max(query_lengths),
        )
    )
    impl = TorchTitanMLAImpl.__new__(TorchTitanMLAImpl)
    impl.kv_lora_rank = KV_LORA_RANK
    impl.scale = SCALE

    out, lse = impl.forward_mqa(
        (q[..., :KV_LORA_RANK], q[..., KV_LORA_RANK:]), cache, metadata, None
    )

    assert lse is None
    start = 0
    for sequence, (num_queries, num_keys) in enumerate(zip(query_lengths, key_lengths)):
        keys = cache[block_table[sequence, : blocks[sequence]].long()].flatten(0, 1)
        keys = keys[:num_keys]
        mask = torch.ones(num_queries, num_keys, dtype=torch.bool, device=device)
        expected, _ = _attention_reference(
            q[start : start + num_queries].transpose(0, 1),
            keys,
            keys[:, :KV_LORA_RANK],
            mask.tril(num_keys - num_queries),
        )
        torch.testing.assert_close(
            out[start : start + num_queries].float(), expected, atol=2e-2, rtol=0
        )
        start += num_queries


def test_mla_prefill_new_tokens_and_context_chunk():
    """Prefill is causal over new tokens, unmasked over context, and returns (H, T) LSE."""
    query_lengths, context_lengths = [5, 130, 1], [7, 0, 64]
    num_heads, qk_head_dim, v_head_dim = 4, 192, 128
    device = torch.device("cuda")
    num_tokens, num_context = sum(query_lengths), sum(context_lengths)
    q = torch.randn(
        num_tokens, num_heads, qk_head_dim, dtype=torch.bfloat16, device=device
    )
    k = torch.randn_like(q)
    v = torch.randn(
        num_tokens, num_heads, v_head_dim, dtype=torch.bfloat16, device=device
    )
    context_k = torch.randn(
        num_context, num_heads, qk_head_dim, dtype=torch.bfloat16, device=device
    )
    context_v = torch.randn(
        num_context, num_heads, v_head_dim, dtype=torch.bfloat16, device=device
    )
    backend = TorchTitanMLAPrefillBackend(
        num_heads=num_heads,
        scale=SCALE,
        kv_lora_rank=KV_LORA_RANK,
        qk_nope_head_dim=qk_head_dim - QK_ROPE_HEAD_DIM,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        v_head_dim=v_head_dim,
        vllm_config=None,
    )
    backend.prepare_metadata(
        SimpleNamespace(
            query_start_loc=_offsets(query_lengths).to(device),
            max_query_len=max(query_lengths),
        )
    )
    chunk = SimpleNamespace(
        query_start_loc=_offsets(query_lengths).to(device),
        max_query_len=max(query_lengths),
        cu_seq_lens=_offsets(context_lengths).to(device),
        max_seq_len=max(context_lengths),
    )

    new_out, new_lse = backend.run_prefill_new_tokens(q, k, v, return_softmax_lse=True)
    context_out, context_lse = backend.run_prefill_context_chunk(
        chunk, q, context_k, context_v
    )

    assert new_lse.shape == context_lse.shape == (num_heads, num_tokens)
    q_start = context_start = 0
    for num_queries, num_context_keys in zip(query_lengths, context_lengths):
        rows = slice(q_start, q_start + num_queries)
        q_HQD = q[rows].transpose(0, 1)
        causal = torch.ones(
            num_queries, num_queries, dtype=torch.bool, device=device
        ).tril()
        expected_out, expected_lse = _attention_reference(
            q_HQD, k[rows].transpose(0, 1), v[rows].transpose(0, 1), causal
        )
        torch.testing.assert_close(
            new_out[rows].float(), expected_out, atol=2e-2, rtol=0
        )
        torch.testing.assert_close(new_lse[:, rows], expected_lse, atol=1e-3, rtol=0)
        if num_context_keys:
            context_rows = slice(context_start, context_start + num_context_keys)
            expected_out, expected_lse = _attention_reference(
                q_HQD,
                context_k[context_rows].transpose(0, 1),
                context_v[context_rows].transpose(0, 1),
                torch.ones(
                    num_queries, num_context_keys, dtype=torch.bool, device=device
                ),
            )
            torch.testing.assert_close(
                context_out[rows].float(), expected_out, atol=2e-2, rtol=0
            )
            torch.testing.assert_close(
                context_lse[:, rows], expected_lse, atol=1e-3, rtol=0
            )
        q_start += num_queries
        context_start += num_context_keys
