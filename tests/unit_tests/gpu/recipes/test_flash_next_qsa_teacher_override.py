# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.models.common.attn_gym import AuxRequest, gather_attn
from torchtitan.models.qwen3_8_flash_next.attention import QSAInnerAttention
from torchtitan.tools.utils import has_cuda_capability
from torchtitan_recipes.overrides.flash_next_qsa_teacher import qsa_teacher

# 125B QSA head layout; a short row of 256-token documents keeps the test cheap.
_NUM_HEADS, _NUM_KV_HEADS, _HEAD_DIM = 24, 2, 256
# 20 blocks = 80 block tokens: one full 64-token kernel tile and a partial one.
_NUM_TOKENS, _DOC_LEN, _RATIO, _NUM_BLOCKS = 1024, 256, 4, 20
# The fused target may round differently, but its error against an fp64 golden
# must stay within this factor of the current teacher's error (plus fp32 rounding).
_FUDGE_FACTOR = 2.0


def _selected_indices(generator: torch.Generator) -> torch.Tensor:
    """Up to _NUM_BLOCKS random complete blocks of each query's document, plus its tail."""
    # Mixed lengths straddle the 64-query tiles; _DOC_LEN is the longest.
    lengths = (37, 200, 256, 131, 256, 144)
    positions = torch.cat([torch.arange(n) for n in lengths])
    doc_start = torch.arange(_NUM_TOKENS) - positions
    blocks = (
        torch.rand(_NUM_TOKENS, _DOC_LEN // _RATIO, generator=generator)
        .argsort(-1)[:, :_NUM_BLOCKS]
        .sort(-1)
        .values
    )
    available = blocks < ((positions + 1) // _RATIO)[:, None]
    tokens = (
        doc_start[:, None, None] + blocks[..., None] * _RATIO + torch.arange(_RATIO)
    )
    tokens = tokens.masked_fill(~available[..., None], -1).flatten(1)
    tail_count = (positions + 1) % _RATIO
    tail = (
        torch.arange(_NUM_TOKENS)[:, None]
        - tail_count[:, None]
        + 1
        + torch.arange(_RATIO - 1)
    ).masked_fill(torch.arange(_RATIO - 1) >= tail_count[:, None], -1)
    selected = torch.cat((tokens, tail), -1)
    selected[-7:] = -1  # padding rows select nothing
    return selected.int().cuda()


def _golden(query, key, selected, scale):
    group = _NUM_HEADS // _NUM_KV_HEADS
    valid = selected >= 0
    keys = key.double()[selected.long().clamp_min(0)]
    logits = torch.einsum(
        "tgrk,tjgk->tgrj", query.double().unflatten(1, (_NUM_KV_HEADS, group)), keys
    )
    logits = (logits * scale).masked_fill(~valid[:, None, None], -torch.inf)
    probs = logits.softmax(-1).nan_to_num(0.0).sum((1, 2)) / _NUM_HEADS
    return (
        probs[:, : _NUM_BLOCKS * _RATIO].unflatten(-1, (_NUM_BLOCKS, _RATIO)).amax(-1)
    )


def _relative_error(value, golden):
    return ((value.double() - golden).norm() / golden.norm()).item()


@unittest.skipUnless(
    torch.cuda.is_available() and has_cuda_capability(10, 0),
    "the fused gather_attn with grouped KV heads and separate values requires SM100",
)
class TestFlashNextQSATeacherOverride(unittest.TestCase):
    def test_teacher_matches_golden_with_and_without_tp_split(self) -> None:
        generator = torch.Generator().manual_seed(0)
        torch.manual_seed(0)
        query = torch.randn(
            _NUM_TOKENS, _NUM_HEADS, _HEAD_DIM, device="cuda"
        ).bfloat16()
        key = torch.randn(
            _NUM_TOKENS, _NUM_KV_HEADS, _HEAD_DIM, device="cuda"
        ).bfloat16()
        value = torch.randn_like(key)
        selected = _selected_indices(generator)
        scale = _HEAD_DIM**-0.5
        _, aux = gather_attn(
            query.transpose(0, 1)[None],
            None,
            key.transpose(0, 1)[None],
            selected[None],
            sliding_window_size=0,
            scale=scale,
            impl="fused",
            sparse_v=value.transpose(0, 1)[None],
            return_aux=AuxRequest(lse=True),
        )
        lse = aux.lse.squeeze(0)
        common = dict(
            num_blocks=_NUM_BLOCKS,
            compress_ratio=_RATIO,
            scale=scale,
            max_context_length=_DOC_LEN,
            num_global_heads=_NUM_HEADS,
        )

        def pool(probs_TJ):
            return probs_TJ.unflatten(-1, (_NUM_BLOCKS, _RATIO)).amax(-1)

        fused = pool(qsa_teacher(query, key, selected, lse, **common))
        # TP2: each rank holds one KV head, its query heads and their LSE; the
        # unpooled sums are reduced before max-pooling.
        group = _NUM_HEADS // _NUM_KV_HEADS
        shard_sums = sum(
            qsa_teacher(
                query[:, g * group : (g + 1) * group],
                key[:, g : g + 1],
                selected,
                lse[g * group : (g + 1) * group],
                **common,
            )
            for g in range(_NUM_KV_HEADS)
        )
        tp_split = pool(shard_sums)
        reference = (
            QSAInnerAttention.Config(compress_ratio=_RATIO)
            .build()
            ._teacher(query, key, selected, _NUM_BLOCKS, scale=scale)
        )
        golden = _golden(query, key, selected, scale)

        floor = _relative_error(golden.float(), golden)
        bound = _FUDGE_FACTOR * _relative_error(reference, golden) + floor
        self.assertLessEqual(_relative_error(fused, golden), bound)
        self.assertLessEqual(_relative_error(tp_split, golden), bound)
        self.assertTrue((fused[-7:] == 0).all() and (tp_split[-7:] == 0).all())


if __name__ == "__main__":
    unittest.main()
