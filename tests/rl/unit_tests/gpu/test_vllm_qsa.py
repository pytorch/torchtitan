# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cache addressing, causal tails and request isolation for Flash-Next generation."""

import random
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch
from torch.testing._internal.common_utils import DeterministicGuard
from vllm.v1.attention.backend import CommonAttentionMetadata

pytest.importorskip(
    "attn_gym.sparse.indexer.paged",
    reason="Paged QSA tests require the Attention Gym decode indexer",
)

from torchtitan.components.data.multimodal.mrope import get_mrope_positions
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.qwen3_8_flash_next import build_model_config
from torchtitan.models.qwen3_8_flash_next.attention import QSAIndexMetadata
from torchtitan.rl.generator.vllm.attention import qsa
from torchtitan.rl.generator.vllm.attention.qsa import (
    VLLMQSAIndexer,
    VLLMQSAInnerAttention,
)
from torchtitan.tools.utils import has_cuda_capability

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Paged QSA requires a GPU"
)


@pytest.fixture
def make_cache(monkeypatch):
    config = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=16),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            enforce_eager=False,
            max_model_len=64,
            uses_mrope=False,
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


@pytest.mark.skipif(
    not has_cuda_capability(10, 0),
    reason="gather_attn with grouped KV heads and separate values needs SM100",
)
def test_batch_invariant_attention_matches_trainer(make_cache, monkeypatch):
    """In batch-invariant mode, QSA attention matches the trainer's bitwise in a
    150-token prefill step, in 64-row chunks, and in one-token decode steps.

    paged_sparse_decode reads a row's selected blocks and its tail in separate tiles
    from position 64 of a prefill and in separate splits in decode, so outside
    batch-invariant mode rows with a tail differ from the trainer.
    """
    monkeypatch.setattr(qsa, "is_in_batch_invariant_mode", lambda: True)
    vllm_config = qsa.get_current_vllm_config()
    vllm_config.model_config.max_model_len = 256
    vllm_config.parallel_config.tensor_parallel_size = 1
    length, num_kv_heads, head_dim, topk, ratio = 150, 2, 64, 4, 4
    trainer_attention = (
        build_model_config("debugmodel", seq_len=256, parallelism=ParallelismConfig())
        .layers[3]
        .attention.inner_attention.build()
        .cuda()
        .eval()
    )
    attention = (
        VLLMQSAInnerAttention.Config(
            compress_ratio=ratio,
            chunk_size=64,
            layer_idx=3,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        .build()
        .cuda()
        .eval()
    )
    num_pages = 16
    attention.cache.bind_kv_cache(
        torch.zeros(
            num_pages,
            num_kv_heads,
            16,
            2 * head_dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
    )
    torch.manual_seed(0)
    block_table = torch.randperm(num_pages, device="cuda").int()[None]
    query = torch.randn(length + 1, 4, head_dim, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(
        length + 1, num_kv_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    value = torch.randn_like(key)
    # Up to topk random complete blocks per row, ascending, as the batch-invariant
    # indexer returns them, and the trainer's indices for the same selection: each
    # block's tokens, then the tail ending at the row.
    rng = random.Random(0)
    blocks, tokens = [], []
    for position in range(length):
        num_complete = (position + 1) // ratio
        chosen = sorted(rng.sample(range(num_complete), min(topk, num_complete)))
        chosen += [-1] * (topk - len(chosen))
        tail = list(range(num_complete * ratio, position + 1))
        blocks.append(chosen)
        tokens.append(
            [
                block * ratio + j if block >= 0 else -1
                for block in chosen
                for j in range(ratio)
            ]
            + tail
            + [-1] * (ratio - 1 - len(tail))
        )
    # The padding row selects nothing.
    blocks = torch.tensor(blocks + [[-1] * topk], device="cuda", dtype=torch.int32)
    tokens = torch.tensor(tokens, device="cuda", dtype=torch.int32)
    scale = head_dim**-0.5
    context = qsa.get_forward_context().attn_metadata
    builder = _metadata_builder(attention.cache)

    def step(start, num_rows):
        """Rows start..start + num_rows of one request, then a padding row."""
        positions = torch.arange(start, start + num_rows, device="cuda")
        context[attention.cache.prefix] = builder.build(
            0,
            _common_metadata(
                block_table=block_table,
                slot_mapping=(
                    block_table[0, positions // 16].long() * 16 + positions % 16
                ),
                query_start_loc=torch.tensor(
                    [0, num_rows], device="cuda", dtype=torch.int32
                ),
                seq_lens=torch.tensor(
                    [start + num_rows], device="cuda", dtype=torch.int32
                ),
                num_tokens=num_rows,
            ),
        )
        rows = torch.cat((positions, torch.tensor([length], device="cuda")))
        output = attention(
            query[rows],
            key[rows],
            value[rows],
            selected_indices_TJ=blocks[rows],
            scores_TB=query.new_empty(num_rows + 1, 0),
            scale=scale,
        )
        assert torch.count_nonzero(output[-1]) == 0
        return output[:-1]

    with torch.no_grad():
        expected = trainer_attention(
            query[:length],
            key[:length],
            value[:length],
            selected_indices_TJ=tokens,
            scores_TB=query.new_empty(length, 0),
            scale=scale,
        )
        prefill = step(0, length)
        decode = torch.cat([step(position, 1) for position in range(length)])
    torch.testing.assert_close(prefill, expected, atol=0, rtol=0)
    torch.testing.assert_close(decode, expected, atol=0, rtol=0)


@pytest.fixture
def deterministic_algorithms(monkeypatch):
    # cuBLAS needs a fixed workspace for deterministic algorithms.
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with DeterministicGuard(True):
        yield


@pytest.mark.parametrize("mrope", [False, True], ids=["1d", "mrope"])
def test_indexer_completed_blocks_across_prefill_and_decode(
    make_cache, deterministic_algorithms, mrope
):
    """Chunked cache pooling and selected blocks must match the trainer.

    With M-RoPE (image prompts) a completed block's key rotates at its first token's 3D
    position. Here an image's placeholders take positions 5-12, so the block of
    positions 12-15 starts on an image token that an earlier step held, and the ring
    must return its position.

    ReLU scores tie often. On SM100 both indexers default to the CuTe selector, which
    resolves ties toward earlier blocks only under deterministic algorithms.
    """
    torch.manual_seed(42)
    qsa.get_current_vllm_config().model_config.uses_mrope = mrope
    config = (
        build_model_config("debugmodel", seq_len=64, parallelism=ParallelismConfig())
        .layers[3]
        .attention.indexer
    )
    config.token_budget = 8
    indexer = (
        VLLMQSAIndexer.Config(
            **{field.name: getattr(config, field.name) for field in fields(config)},
            layer_idx=3,
        )
        .build()
        .cuda()
    )
    indexer.init_states()
    trainer_indexer = config.build().cuda()
    trainer_indexer.init_states()
    for module in (indexer, trainer_indexer):
        for parameter in module.parameters():
            parameter.data = parameter.data.bfloat16()
        module.requires_grad_(False)
    trainer_indexer.load_state_dict(indexer.state_dict())
    # Hybrid cache groups can pad the physical page stride without adding states.
    indexer.cache.bind_kv_cache(
        torch.zeros(3, 1, 8, config.head_dim, device="cuda", dtype=torch.bfloat16)[
            :, :, :4
        ]
    )
    # M-RoPE positions take three int32 coordinates, two 16-bit lanes each.
    ring_width = config.head_dim + (6 if mrope else 0)
    indexer.raw_cache.bind_kv_cache(
        torch.randn(2, 1, 16, ring_width, device="cuda", dtype=torch.bfloat16)[:, :, :4]
    )
    step_lengths = (3, 1, 9, 6, 1)
    total_length = sum(step_lengths)
    if mrope:
        # A 1 x 2 x 4 grid of merged patches at token 5: positions 5-12 are image tokens.
        rope_positions = get_mrope_positions(total_length, [(5, (1, 2, 4))]).cuda()
        assert rope_positions[12].tolist() == [5, 6, 8]
    else:
        rope_positions = torch.arange(total_length, device="cuda")
    context = qsa.get_forward_context().attn_metadata
    candidate_builder = _metadata_builder(indexer.cache)
    raw_builder = _metadata_builder(indexer.raw_cache)
    raw_keys = []
    inputs = []
    offset = 0
    block_table = torch.tensor([[2, 1]], device="cuda", dtype=torch.int32)
    for length in step_lengths:
        positions = torch.arange(offset, offset + length, device="cuda")
        # vLLM pads the step with a row at position 0.
        step_rope_positions = torch.cat(
            (
                rope_positions[offset : offset + length],
                torch.zeros_like(rope_positions[:1]),
            )
        )
        x = torch.randn(
            length + 1,
            config.index_qk_proj.in_features,
            device="cuda",
            dtype=torch.bfloat16,
        )
        inputs.append(x[:length])
        raw_keys.append(indexer.index_qk_proj(x)[:length, -config.head_dim :])
        common = _common_metadata(
            block_table=block_table,
            slot_mapping=(block_table[0, positions // 16].long() * 16 + positions % 16),
            query_start_loc=torch.tensor([0, length], device="cuda", dtype=torch.int32),
            seq_lens=torch.tensor([offset + length], device="cuda", dtype=torch.int32),
            num_tokens=length,
        )
        metadata = candidate_builder.build(0, common)
        context[indexer.cache.prefix] = metadata
        common.block_table_tensor = torch.ones(1, 1, device="cuda", dtype=torch.int32)
        context[indexer.raw_cache.prefix] = raw_builder.build(0, common)
        selected, _ = indexer(x, positions=step_rope_positions)
        offset += length
        count = offset // 4
        if count:
            pooled = (
                torch.cat(raw_keys)[: count * 4]
                .reshape(count, 4, -1)
                .float()
                .mean(1)
                .bfloat16()
            )
            expected = indexer._rotate(
                indexer.k_layernorm(pooled)[:, None],
                rope_positions[torch.arange(count, device="cuda") * 4],
            )[:, 0]
            (cache,) = indexer.cache.views()
            blocks = torch.arange(count, device="cuda")
            actual = cache[block_table[0, blocks // 4].long(), blocks % 4, 0]
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        prefix_positions = torch.arange(offset, device="cuda")
        token_indices, _ = trainer_indexer(
            torch.cat(inputs),
            positions=rope_positions[:offset],
            metadata=QSAIndexMetadata.from_inputs(
                prefix_positions,
                padding_mask_T=None,
                compress_ratio=trainer_indexer.compress_ratio,
            ),
        )
        expected_selection = torch.full_like(selected, -1)
        num_selected = min(indexer.block_topk, count)
        # The trainer expands selected blocks into tokens, followed by the tail.
        expected_selection[:length, :num_selected] = (
            token_indices[-length:, : num_selected * 4 : 4] // 4
        )
        torch.testing.assert_close(selected, expected_selection, atol=0, rtol=0)
        assert (selected[length:] == -1).all()
        for row, position in enumerate(positions.tolist()):
            assert ((selected[row] < (position + 1) // 4) | (selected[row] == -1)).all()


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
