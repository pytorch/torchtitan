# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The SGLang attention backend over an fp8 KV cache. Runs in the B200 RL CI job, with the
SGLang environment (see torchtitan/rl/generator/sglang/README.md):
    pytest tests/rl/unit_tests/gpu/test_sglang_fp8_attention.py -v
"""

import pytest
import torch

# The SGLang environment: torchtitan/rl/generator/sglang/README.md.
pytest.importorskip("sglang")

# The backend's own rule (TorchTitanVarlenAttnBackend.__init__).
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="FA4's fp8 forward requires an SM100-family GPU",
)


# 256: Qwen3.5; 128: Qwen3.
@pytest.mark.parametrize("head_dim", [256, 128])
@pytest.mark.parametrize(
    "query_lens, seq_lens, query_dtype",
    [
        ([1, 1, 1], [9, 300, 1], torch.float8_e4m3fn),
        ([7, 3, 1], [20, 3, 129], torch.bfloat16),
    ],
    ids=["decode", "prefill"],
)
def test_fp8_paged_attention_matches_causal_attention(
    query_lens, seq_lens, query_dtype, head_dim
):
    """Each query token attends causally to its row's keys through FA4's fp8 forward.

    SGLang calls the backend with an e4m3 query for decode batches and a bf16 query for
    prefill batches. The cache holds e4m3 K and V, one token per page (SGLang's page
    size for this backend), pages in random order. The reference runs in fp32 on the
    same e4m3 values.
    """
    from torchtitan.rl.generator.sglang.attention import attention

    torch.manual_seed(0)
    device = torch.device("cuda")
    num_heads, num_kv_heads = 8, 2
    # The backend's state that _paged_attention reads; the rest is SGLang's metadata.
    backend = attention.TorchTitanVarlenAttnBackend.__new__(
        attention.TorchTitanVarlenAttnBackend
    )
    backend.kv_cache_dtype = torch.float8_e4m3fn
    backend.max_context_len = 4096

    # [pages, page size 1, KV heads, K]
    num_pages = sum(seq_lens) + 5
    k_cache, v_cache = (
        torch.randn(num_pages, 1, num_kv_heads, head_dim, device=device).to(
            torch.float8_e4m3fn
        )
        for _ in range(2)
    )
    page_order = torch.randperm(num_pages)
    page_table = torch.zeros(len(seq_lens), max(seq_lens), dtype=torch.int32)
    start = 0
    for row, length in enumerate(seq_lens):
        page_table[row, :length] = page_order[start : start + length]
        start += length
    cu_seqlens_q = torch.tensor(
        [0, *torch.tensor(query_lens).cumsum(0).tolist()], dtype=torch.int32
    )
    q = torch.randn(sum(query_lens), num_heads, head_dim, device=device).to(query_dtype)

    # torch's varlen attention takes no fp8 inputs, so only FA4's fp8 forward can return this.
    out = backend._paged_attention(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table.to(device),
        cache_seqlens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        cu_seqlens_q=cu_seqlens_q.to(device),
        max_seqlen_q=max(query_lens),
        softmax_scale=head_dim**-0.5,
        causal=True,
    )
    assert out.dtype == torch.bfloat16

    queries = q.to(torch.float8_e4m3fn).float()
    for row, (num_queries, num_keys) in enumerate(zip(query_lens, seq_lens)):
        pages = page_table[row, :num_keys].long()
        keys = k_cache[pages].flatten(0, 1).float()
        values = v_cache[pages].flatten(0, 1).float()
        rows = slice(int(cu_seqlens_q[row]), int(cu_seqlens_q[row + 1]))
        group = num_heads // num_kv_heads
        # [heads, tokens, keys]; token j sees keys 0 .. num_keys - num_queries + j.
        scores = head_dim**-0.5 * torch.einsum(
            "thd,khd->htk",
            queries[rows],
            keys.repeat_interleave(group, dim=1),
        )
        is_visible = torch.ones(
            num_queries, num_keys, dtype=torch.bool, device=device
        ).tril(num_keys - num_queries)
        scores = scores.masked_fill(~is_visible, float("-inf"))
        expected = torch.einsum(
            "htk,khd->thd",
            scores.softmax(dim=-1),
            values.repeat_interleave(group, dim=1),
        )
        # FA4's fp8 kernels round the softmax probabilities to e4m3 before P @ V.
        rel_l2 = (out[rows].float() - expected).norm() / expected.norm()
        assert rel_l2 < 0.05, (row, rel_l2)
