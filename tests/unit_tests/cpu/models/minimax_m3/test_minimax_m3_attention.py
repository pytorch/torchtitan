# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import pytest
import torch
from torch.nn.attention.flex_attention import create_mask

from torchtitan.models.minimax_m3.attention import (
    build_msa_metadata,
    msa_block_mask,
    MSAIndexerLoss,
    MSAInnerAttention,
    select_msa_blocks,
)

_BLOCK_SIZE = 16
_TILE_SIZE = 32
_TOPK_BLOCKS = 4
# Packed documents; the later ones start inside a flex tile.
_DOCUMENT_LENGTHS = (150, 70, 36)
_NUM_GROUPS = 2
_HEADS_PER_GROUP = 2
_HEAD_DIM = 16
_INDEX_HEAD_DIM = 16


def _positions() -> torch.Tensor:
    return torch.cat([torch.arange(length) for length in _DOCUMENT_LENGTHS])


def _block_starts(positions: torch.Tensor) -> torch.Tensor:
    """Global first token of each token's document-local key block."""
    token = torch.arange(positions.numel())
    return token - positions + positions // _BLOCK_SIZE * _BLOCK_SIZE


def _reference_selection(index_q, index_k, positions) -> list[list[set[int]]]:
    """Top-k document-local blocks by causal max-pooled score, own block first."""
    num_tokens = positions.numel()
    doc_start = (torch.arange(num_tokens) - positions).tolist()
    block_start = _block_starts(positions).tolist()
    selections = []
    for query in range(num_tokens):
        per_group = []
        for group in range(index_q.shape[1]):
            scores: dict[int, float] = {}
            for key in range(doc_start[query], query + 1):
                score = float(index_q[query, group] @ index_k[key])
                scores[block_start[key]] = max(
                    scores.get(block_start[key], -math.inf), score
                )
            scores[block_start[query]] = math.inf
            per_group.append(
                set(sorted(scores, key=lambda block: -scores[block])[:_TOPK_BLOCKS])
            )
        selections.append(per_group)
    return selections


def _reference_mask(selections, positions) -> torch.Tensor:
    """``[H, T, T]``: a query attends the causal keys of its selected blocks."""
    num_tokens = positions.numel()
    block_start = _block_starts(positions)
    allowed = torch.zeros(
        _NUM_GROUPS * _HEADS_PER_GROUP, num_tokens, num_tokens, dtype=torch.bool
    )
    causal = torch.ones(num_tokens, num_tokens, dtype=torch.bool).tril()
    for query in range(num_tokens):
        for head in range(allowed.shape[0]):
            blocks = torch.tensor(sorted(selections[query][head // _HEADS_PER_GROUP]))
            allowed[head, query] = torch.isin(block_start, blocks) & causal[query]
    return allowed


@pytest.fixture(scope="module")
def msa_case():
    torch.manual_seed(0)
    positions = _positions()
    num_tokens = positions.numel()
    index_q = torch.randn(num_tokens, _NUM_GROUPS, _INDEX_HEAD_DIM)
    index_k = torch.randn(num_tokens, _INDEX_HEAD_DIM)
    metadata = build_msa_metadata(
        positions, block_size=_BLOCK_SIZE, tile_size=_TILE_SIZE
    )
    block_indices = select_msa_blocks(
        index_q, index_k, metadata, topk_blocks=_TOPK_BLOCKS
    )
    selections = _reference_selection(index_q, index_k, positions)
    return positions, index_q, index_k, metadata, block_indices, selections


def test_selection_matches_reference(msa_case) -> None:
    positions, _, _, metadata, block_indices, selections = msa_case
    for query in range(positions.numel()):
        for group in range(_NUM_GROUPS):
            selected = block_indices[query, group]
            starts = metadata.block_start_N[selected[selected >= 0]].tolist()
            assert set(starts) == selections[query][group], (query, group)


def test_block_mask_attends_selected_causal_keys(msa_case) -> None:
    positions, _, _, metadata, block_indices, selections = msa_case
    num_heads = _NUM_GROUPS * _HEADS_PER_GROUP
    block_mask, _ = msa_block_mask(block_indices, metadata, num_heads=num_heads)
    num_tokens = positions.numel()
    dense = create_mask(
        block_mask.mask_mod, 1, num_heads, num_tokens, num_tokens, device="cpu"
    )[0]
    expected = _reference_mask(selections, positions)
    assert torch.equal(dense, expected)

    # Every attended key lies in a tile the block mask visits.
    num_blocks, indices = block_mask.kv_num_blocks[0], block_mask.kv_indices[0]
    for head in range(num_heads):
        for query_tile in range(num_blocks.shape[1]):
            rows = expected[
                head, query_tile * _TILE_SIZE : (query_tile + 1) * _TILE_SIZE
            ]
            needed = set((rows.any(dim=0).nonzero().flatten() // _TILE_SIZE).tolist())
            visited = indices[head, query_tile, : num_blocks[head, query_tile]]
            assert needed <= set(visited.tolist()), (head, query_tile)


def test_indexer_loss_matches_reference_kl(msa_case) -> None:
    positions, index_q, index_k, metadata, block_indices, selections = msa_case
    num_tokens = positions.numel()
    num_heads = _NUM_GROUPS * _HEADS_PER_GROUP
    _, occupied = msa_block_mask(block_indices, metadata, num_heads=num_heads)
    allowed = _reference_mask(selections, positions)
    q = torch.randn(num_tokens, num_heads, _HEAD_DIM, requires_grad=True)
    k = torch.randn(num_tokens, _NUM_GROUPS, _HEAD_DIM, requires_grad=True)
    logits = torch.einsum(
        "thd,uhd->htu", q, k.repeat_interleave(_HEADS_PER_GROUP, dim=1)
    ) * (_HEAD_DIM**-0.5)
    logits = logits.masked_fill(~allowed, -math.inf)
    lse = torch.logsumexp(logits, dim=-1).transpose(0, 1).detach()

    loss = MSAIndexerLoss.Config(
        coeff=1.0,
        num_groups=_NUM_GROUPS,
        softmax_scale=_HEAD_DIM**-0.5,
        index_softmax_scale=_INDEX_HEAD_DIM**-0.5,
    ).build()
    index_q = index_q.clone().requires_grad_()
    index_k = index_k.clone().requires_grad_()
    carrier = torch.zeros(num_tokens, num_heads, _HEAD_DIM, requires_grad=True)
    out = loss(
        q,
        k,
        index_q,
        index_k,
        lse,
        metadata,
        block_indices_TGS=block_indices,
        occupied_XGX=occupied,
        carrier=carrier,
        denominator=torch.tensor(1.0),
    )
    out.sum().backward()

    teacher = (
        logits.softmax(dim=-1)
        .view(_NUM_GROUPS, _HEADS_PER_GROUP, num_tokens, num_tokens)
        .mean(dim=1)
    )
    group_allowed = allowed.view(_NUM_GROUPS, _HEADS_PER_GROUP, num_tokens, -1)[:, 0]
    index_logits = torch.einsum("tgi,ui->gtu", index_q, index_k) * (
        _INDEX_HEAD_DIM**-0.5
    )
    log_student = (
        index_logits.masked_fill(~group_allowed, -math.inf)
        .log_softmax(dim=-1)
        .masked_fill(~group_allowed, 0.0)
    )
    expected = (
        torch.special.xlogy(teacher, teacher) - teacher * log_student
    ).sum() / _NUM_GROUPS
    torch.testing.assert_close(loss.instance_acc, expected.detach())

    assert torch.equal(out, carrier)
    assert carrier.grad is not None
    assert torch.equal(carrier.grad, torch.ones_like(carrier))
    assert index_q.grad is not None and index_q.grad.abs().sum() > 0
    assert index_k.grad is not None and index_k.grad.abs().sum() > 0
    assert q.grad is None and k.grad is None


def test_dense_mode_keeps_the_document_causal_mask() -> None:
    config = MSAInnerAttention.Config(
        index_block_size=_BLOCK_SIZE, topk_blocks=_TOPK_BLOCKS, sparse=False
    )
    assert config.is_full_attention
    positions = _positions()
    num_tokens = positions.numel()
    metadata = config.build_attention_metadata(positions)
    assert metadata.dense_mask is not None
    dense = create_mask(
        metadata.dense_mask.mask_mod, 1, 1, num_tokens, num_tokens, device="cpu"
    )[0, 0]
    doc_start = torch.arange(num_tokens) - positions
    token = torch.arange(num_tokens)
    expected = (token[None, :] <= token[:, None]) & (
        token[None, :] >= doc_start[:, None]
    )
    assert torch.equal(dense, expected)


def test_rejects_index_blocks_larger_than_the_flex_tile() -> None:
    with pytest.raises(ValueError, match="index_block_size"):
        MSAInnerAttention.Config(index_block_size=256, block_size=128)


def test_indexer_loss_rejects_groups_that_do_not_tile_tp(msa_case) -> None:
    positions, index_q, index_k, metadata, block_indices, _ = msa_case
    num_tokens = positions.numel()
    loss = MSAIndexerLoss.Config(
        coeff=1.0,
        num_groups=2 * _NUM_GROUPS,
        softmax_scale=1.0,
        index_softmax_scale=1.0,
    ).build()
    with pytest.raises(ValueError, match="do not tile"):
        loss(
            torch.randn(num_tokens, _NUM_GROUPS * _HEADS_PER_GROUP, _HEAD_DIM),
            torch.randn(num_tokens, _NUM_GROUPS, _HEAD_DIM),
            index_q,
            index_k,
            torch.zeros(num_tokens, _NUM_GROUPS * _HEADS_PER_GROUP),
            metadata,
            block_indices_TGS=block_indices,
            occupied_XGX=None,
            carrier=torch.zeros(1),
            denominator=torch.tensor(1.0),
        )
