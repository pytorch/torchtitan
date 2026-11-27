# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from torchtitan.rl.generator.vllm.attention import attention
from torchtitan.rl.generator.vllm.attention.attention import (
    TorchTitanVarlenInnerAttentionImpl,
)
from vllm.config import set_current_vllm_config, VllmConfig
from vllm.forward_context import ForwardContext, override_forward_context

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="FA4's SwapAB decode kernel runs on SM100/SM103 only",
)


@pytest.mark.parametrize("kv_cache_dtype", ["auto", "fp8"])
@pytest.mark.parametrize(
    "query_lens, seq_lens, is_uniform_verify, num_fa4_calls",
    [
        # 8 rows of 1 + 3 tokens; the 30k row gets KV splits; a padded row (seq_len 0).
        ([4] * 8, [4, 130, 1000, 257, 30_000, 0, 513, 2_048], True, 1),
        # Fewer than 8 verify rows: one FA4 varlen call.
        ([4, 4, 4], [130, 30_000, 257], True, 1),
        # A leading 1-token row: FA4's decode call for it, the varlen call for the rest.
        ([1, 4, 4], [257, 130, 30_000], False, 2),
    ],
    ids=["verify", "few_rows", "mixed"],
)
def test_forward_matches_causal_attention(
    query_lens, seq_lens, is_uniform_verify, num_fa4_calls, kv_cache_dtype
):
    """Each query token attends causally to its row's keys, through the impl's dispatch.

    vLLM passes max_model_len as max_seq_len when it captures FULL graphs. An fp8 cache holds
    K and V in e4m3 and the query arrives in e4m3, as vLLM writes them with its scales of 1.
    fp8 verify steps make one varlen call with the step's rows (max_seqlen_q 4) at any row count.
    """
    is_fp8 = kv_cache_dtype == "fp8"
    # 8: the impl's row cutoff (`_MIN_VERIFY_ROWS_ON_DECODE_KERNEL`).
    takes_decode_kernel = is_uniform_verify and not is_fp8 and len(query_lens) >= 8
    max_seq_len = 262144  # Qwen3.8-27B's max_model_len
    torch.manual_seed(0)
    device = torch.device("cuda")
    num_heads, num_kv_heads, head_dim, page_size = 24, 4, 256, 128
    config = VllmConfig()
    config.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    with set_current_vllm_config(config):
        impl = TorchTitanVarlenInnerAttentionImpl(
            num_heads=num_heads,
            head_size=head_dim,
            scale=head_dim**-0.5,
            num_kv_heads=num_kv_heads,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype=kv_cache_dtype,
        )
    assert impl.verify_query_len == 4

    # The engine's cache: [pages, KV heads, page, K | V], with pages in random order.
    num_pages = [-(-length // page_size) for length in seq_lens]
    kv_cache = torch.randn(
        sum(num_pages) + 1,
        num_kv_heads,
        page_size,
        2 * head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    if is_fp8:
        kv_cache = kv_cache.to(torch.float8_e4m3fn)
    key_cache, value_cache = kv_cache.transpose(1, 2).split(head_dim, dim=-1)
    page_order = torch.randperm(kv_cache.shape[0])
    # As wide as vLLM sizes it (max_model_len / page_size), so FA4 runs its per-sequence splits.
    block_table = torch.zeros(
        len(seq_lens), max_seq_len // page_size, dtype=torch.int32
    )
    start = 0
    for row, count in enumerate(num_pages):
        block_table[row, :count] = page_order[start : start + count]
        start += count
    query_start_loc = torch.tensor([0, *query_lens]).cumsum(0).int()
    num_tokens = int(query_start_loc[-1])
    query = torch.randn(
        num_tokens, num_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    if is_fp8:
        query = query.to(torch.float8_e4m3fn)
    metadata = SimpleNamespace(
        num_actual_tokens=num_tokens,
        query_start_loc=query_start_loc.to(device),
        max_query_len=max(query_lens),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        max_seq_len=max_seq_len,
        block_table=block_table.to(device),
        use_cascade=False,
        causal=True,
    )
    # vLLM keeps an fp8 cache as uint8; the impl views it as e4m3.
    stored_cache = kv_cache.view(torch.uint8) if is_fp8 else kv_cache
    layer = SimpleNamespace(layer_name="attn", kv_cache=stored_cache)
    context = ForwardContext(
        no_compile_layers={"attn": layer},
        attn_metadata={"attn": metadata},
        slot_mapping={},
    )
    output = torch.zeros(
        num_tokens, num_heads, head_dim, dtype=torch.bfloat16, device=device
    )

    with (
        override_forward_context(context),
        patch.object(
            impl, "_verify_rows_as_decode", wraps=impl._verify_rows_as_decode
        ) as verify_rows_as_decode,
        patch.object(
            attention,
            "paged_varlen_attention_out",
            wraps=attention.paged_varlen_attention_out,
        ) as fa4_calls,
    ):
        impl.forward(layer, query, None, None, stored_cache, metadata, output=output)
    assert verify_rows_as_decode.called == takes_decode_kernel
    assert fa4_calls.call_count == num_fa4_calls
    if is_fp8 and is_uniform_verify:
        assert fa4_calls.call_args.kwargs["max_seqlen_q"] == 4
        assert fa4_calls.call_args.kwargs["cu_seqlens_q"] is metadata.query_start_loc

    for row, (num_queries, num_keys) in enumerate(zip(query_lens, seq_lens)):
        if num_keys == 0:
            continue
        pages = block_table[row, : num_pages[row]].long()
        keys = key_cache[pages].flatten(0, 1)[:num_keys].float()
        values = value_cache[pages].flatten(0, 1)[:num_keys].float()
        rows = slice(int(query_start_loc[row]), int(query_start_loc[row + 1]))
        # [heads, tokens, keys]; token j sees keys 0 .. num_keys - num_queries + j.
        scores = impl.scale * torch.einsum(
            "thd,khd->htk",
            query[rows].float(),
            keys.repeat_interleave(impl.num_queries_per_kv, dim=1),
        )
        is_visible = torch.ones(
            num_queries, num_keys, dtype=torch.bool, device=device
        ).tril(num_keys - num_queries)
        scores = scores.masked_fill(~is_visible, float("-inf"))
        expected = torch.einsum(
            "htk,khd->thd",
            scores.softmax(dim=-1),
            values.repeat_interleave(impl.num_queries_per_kv, dim=1),
        )
        if is_fp8:
            # FA4's fp8 kernels round the softmax probabilities to e4m3 before P @ V: rel L2
            # ~0.02 on this flat attention.
            rel_l2 = (output[rows].float() - expected).norm() / expected.norm()
            assert rel_l2 < 0.05, (row, rel_l2)
        else:
            torch.testing.assert_close(
                output[rows].float(), expected, atol=2e-3, rtol=2e-2
            )
