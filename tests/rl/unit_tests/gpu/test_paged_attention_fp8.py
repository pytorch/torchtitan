# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FA4's fp8 paged attention (`paged_varlen_attention_out` with descales) vs bf16 FA4."""

import pytest
import torch

from torchtitan.rl.generator.attention.attention import paged_varlen_attention_out
from torchtitan.tools.utils import activate_cuda_flash_attention_impl

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="FA4's fp8 forward needs an SM100-family GPU",
)

HEADS, KV_HEADS, HEAD_DIM, PAGE = 24, 4, 256, 128


@pytest.mark.parametrize(
    "q_scale, k_scale, v_scale", [(1.0, 1.0, 1.0), (2.0, 0.5, 4.0)]
)
def test_fp8_matches_bf16_on_the_same_e4m3_values(q_scale, k_scale, v_scale):
    """The cache holds K / k_scale and V / v_scale in e4m3, as vLLM writes it; descales undo that.

    The reference runs bf16 FA4 on the dequantized values (power-of-two scales keep them exact).
    The batch is 2 decode rows + a 300-token prefill row in the vLLM backend's layout: K and V
    packed in one page tensor, run as a decode call and a prefill call.
    """
    activate_cuda_flash_attention_impl()
    torch.manual_seed(0)
    device = "cuda"
    contexts = [700, 1500, 1100]
    cu_seqlens_q = torch.tensor([0, 1, 2, 302], dtype=torch.int32, device=device)
    pages_per_seq = [(context + PAGE - 1) // PAGE for context in contexts]
    block_table = torch.zeros(
        len(contexts), max(pages_per_seq), dtype=torch.int32, device=device
    )
    start = 0
    for i, num_pages in enumerate(pages_per_seq):
        block_table[i, :num_pages] = torch.arange(
            start, start + num_pages, device=device
        )
        start += num_pages

    kv_cache = torch.randn(start, KV_HEADS, PAGE, 2 * HEAD_DIM, device=device)
    kv_cache[..., :HEAD_DIM] /= k_scale
    kv_cache[..., HEAD_DIM:] /= v_scale
    kv_cache_fp8 = kv_cache.to(torch.float8_e4m3fn)
    query_fp8 = (torch.randn(302, HEADS, HEAD_DIM, device=device) / q_scale).to(
        torch.float8_e4m3fn
    )

    kv_cache_ref = kv_cache_fp8.to(torch.bfloat16)
    kv_cache_ref[..., :HEAD_DIM] *= k_scale
    kv_cache_ref[..., HEAD_DIM:] *= v_scale
    query_ref = query_fp8.to(torch.bfloat16) * q_scale

    def descale(value):
        return torch.tensor(value, device=device).expand(len(contexts), KV_HEADS)

    descales = (descale(q_scale), descale(k_scale), descale(v_scale))
    seqused_k = torch.tensor(contexts, dtype=torch.int32, device=device)
    outputs = []
    for query, cache, call_descales in (
        (query_fp8, kv_cache_fp8, descales),
        (query_ref, kv_cache_ref, None),
    ):
        out = torch.empty(302, HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
        key_cache, value_cache = cache.transpose(1, 2).split(HEAD_DIM, dim=-1)
        # (cu_seqlens_q, max_seqlen_q, seqused_k) of the decode call, then the prefill call.
        calls = [
            (
                cu_seqlens_q.clamp(max=2),
                1,
                seqused_k * torch.tensor([1, 1, 0], device=device),
            ),
            (
                cu_seqlens_q.clamp(min=2),
                300,
                seqused_k * torch.tensor([0, 0, 1], device=device),
            ),
        ]
        for call_cu_seqlens_q, call_max_seqlen_q, call_seqused_k in calls:
            paged_varlen_attention_out(
                out,
                query,
                key_cache,
                value_cache,
                cu_seqlens_q=call_cu_seqlens_q.int(),
                max_seqlen_q=call_max_seqlen_q,
                max_seqlen_k=max(contexts),
                seqused_k=call_seqused_k.int(),
                block_table=block_table,
                scale=HEAD_DIM**-0.5,
                window_size=(-1, 0),
                enable_gqa=True,
                return_lse=False,
                descales=call_descales,
            )
        outputs.append(out.float())

    fp8_out, bf16_out = outputs
    # FA4's fp8 kernel rounds the softmax probabilities to e4m3 before P @ V: rel L2 ~0.024 on
    # this flat attention. A dropped descale or a wrong decode page gives 0.5-2.6.
    for rows in (slice(0, 2), slice(2, None)):
        rel_l2 = (fp8_out[rows] - bf16_out[rows]).norm() / bf16_out[rows].norm()
        assert rel_l2 < 0.05, (rows, rel_l2)
