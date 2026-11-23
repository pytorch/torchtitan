# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cache addressing, causal tails and request isolation for Flash-Next generation."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.attention.backend import CommonAttentionMetadata

pytest.importorskip(
    "attn_gym.sparse.paged_decode",
    reason="Paged QSA tests require Attention Gym paged sparse attention",
)

from torchtitan.rl.generator.vllm.attention import qsa

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Paged QSA requires a GPU"
)


@pytest.fixture
def make_cache(monkeypatch):
    config = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=16),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16, enforce_eager=False, max_model_len=64
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=64, max_num_seqs=4),
        num_speculative_tokens=0,
        speculative_config=None,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        compilation_config=SimpleNamespace(
            static_forward_context={}, cudagraph_mode=qsa.CUDAGraphMode.NONE
        ),
    )
    monkeypatch.setattr(qsa, "get_current_vllm_config", lambda: config)
    metadata = {}
    monkeypatch.setattr(
        qsa, "get_forward_context", lambda: SimpleNamespace(attn_metadata=metadata)
    )

    def build(prefix, head_dim, num_kv_heads=1, num_pages=8):
        cache = qsa.PagedCache(
            prefix=prefix, head_dim=head_dim, num_kv_heads=num_kv_heads
        )
        cache.bind_kv_cache(
            torch.randn(
                num_pages,
                num_kv_heads,
                16,
                2 * head_dim,
                device="cuda",
                dtype=torch.bfloat16,
            )
        )
        return cache, metadata

    return build


def _common_metadata(
    *, block_table, slot_mapping, query_start_loc, seq_lens, num_tokens
):
    offsets_cpu = query_start_loc.cpu()
    return CommonAttentionMetadata(
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
        query_start_loc=query_start_loc,
        query_start_loc_cpu=offsets_cpu,
        seq_lens=seq_lens,
        num_reqs=seq_lens.numel(),
        num_actual_tokens=num_tokens,
        max_query_len=int(offsets_cpu.diff().max()),
        max_seq_len=int(seq_lens.max()),
        causal=True,
    )


def _metadata_builder(cache):
    config = qsa.get_current_vllm_config()
    return cache.get_attn_backend().get_builder_cls()(
        cache.get_kv_cache_spec(config), [cache.prefix], config, torch.device("cuda")
    )


def test_paged_qsa_mixed_queries_tail_and_padding(make_cache):
    """A reordered physical cache must match sparse causal attention, including tails."""
    torch.manual_seed(42)
    cache, context = make_cache("qsa", 32, num_kv_heads=2)
    lengths, query_lengths = [19, 15, 35], [3, 7, 1]
    block_table = torch.tensor(
        [[5, 1, 0], [3, 0, 0], [6, 2, 4]], device="cuda", dtype=torch.int32
    )
    offsets = torch.tensor([0, 3, 10, 11], device="cuda", dtype=torch.int32)
    slots = []
    for request, (length, query_length) in enumerate(zip(lengths, query_lengths)):
        for position in range(length - query_length, length):
            slots.append(block_table[request, position // 16] * 16 + position % 16)
    metadata = _metadata_builder(cache).build(
        0,
        _common_metadata(
            block_table=block_table,
            slot_mapping=torch.stack(slots).long(),
            query_start_loc=offsets,
            seq_lens=torch.tensor(lengths, device="cuda", dtype=torch.int32),
            num_tokens=11,
        ),
    )
    context["qsa"] = metadata
    query = torch.randn(13, 4, 32, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(13, 2, 32, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    # Select just the first complete block. Later tokens use it plus their tail.
    blocks = torch.full((13, 1), -1, device="cuda", dtype=torch.int32)
    for request, (length, query_length) in enumerate(zip(lengths, query_lengths)):
        start = int(offsets[request])
        for row in range(query_length):
            if length - query_length + row + 1 >= 4:
                blocks[start + row, 0] = 0
    output = cache.attend(query, key, value, blocks, scale=32**-0.5)
    key_cache, value_cache = cache.views()
    for request, (length, query_length) in enumerate(zip(lengths, query_lengths)):
        for row in range(query_length):
            index = int(offsets[request]) + row
            count = length - query_length + row + 1
            selected = list(range(4)) if count >= 4 else []
            selected += list(range(count // 4 * 4, count))
            selected = torch.tensor(selected, device="cuda")
            pages = block_table[request, selected // 16].long()
            keys = key_cache[pages, selected % 16].repeat_interleave(2, 1).float()
            values = value_cache[pages, selected % 16].repeat_interleave(2, 1).float()
            logits = torch.einsum("hd,shd->hs", query[index].float(), keys) * 32**-0.5
            expected = torch.einsum("hs,shd->hd", logits.softmax(-1), values)
            torch.testing.assert_close(
                output[index].float(), expected, atol=1e-2, rtol=1e-2
            )
    assert torch.count_nonzero(output[11:]) == 0


def test_decode_graph_refreshes_request_layout(make_cache):
    """Replay must consume vLLM's refreshed block tables, lengths and padding."""
    qsa.get_current_vllm_config().compilation_config.cudagraph_mode = (
        qsa.CUDAGraphMode.FULL_DECODE_ONLY
    )
    cache, context = make_cache("graph", 32)
    builder = _metadata_builder(cache)
    block_table = torch.zeros(4, 8, device="cuda", dtype=torch.int32)
    query_start_loc = torch.zeros(5, device="cuda", dtype=torch.int32)
    seq_lens = torch.zeros(4, device="cuda", dtype=torch.int32)
    slot_mapping = torch.zeros(4, device="cuda", dtype=torch.int64)

    def metadata(pages, lengths):
        count = len(pages)
        block_table.zero_()
        block_table[:count, 0] = torch.tensor(pages, device="cuda")
        query_start_loc.copy_(torch.arange(5, device="cuda").clamp_max(count))
        seq_lens.zero_()
        seq_lens[:count] = torch.tensor(lengths, device="cuda")
        slot_mapping.fill_(-1)
        slot_mapping[:count] = torch.tensor(
            [page * 16 + length - 1 for page, length in zip(pages, lengths)],
            device="cuda",
        )
        return builder.build(
            0,
            _common_metadata(
                num_tokens=4,
                block_table=block_table,
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                slot_mapping=slot_mapping,
            ),
        )

    query = torch.randn(4, 2, 32, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(4, 1, 32, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    blocks = torch.zeros(4, 1, device="cuda", dtype=torch.int32)
    context["graph"] = metadata([1, 2, 3, 4], [8, 9, 10, 11])

    def step():
        return cache.attend(query, key, value, blocks, scale=32**-0.5)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = step()
    context["graph"] = metadata([6, 1], [13, 7])
    expected = step()
    graph.replay()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.count_nonzero(actual[2:]) == 0
