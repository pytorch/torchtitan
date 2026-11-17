# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import create_block_mask
from torchtitan.distributed.spmd_types import spmd_dense_sp_enabled

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.vision_encoder import (
    VisionAttention,
    VisionTransformerBlock,
)


class _IdentityAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_shape: torch.Size | None = None

    def forward(self, q_THK, k_THK, v_THV, *, attention_metadata):
        self.input_shape = q_THK.shape
        return q_THK


class TestVisionAttention(unittest.TestCase):
    def test_td_layout(self) -> None:
        dim, num_heads, num_tokens = 16, 4, 7
        attention = VisionAttention(
            VisionAttention.Config(
                dim=dim,
                num_heads=num_heads,
                wq=Linear.Config(in_features=dim, out_features=dim),
                wk=Linear.Config(in_features=dim, out_features=dim),
                wv=Linear.Config(in_features=dim, out_features=dim),
                proj=Linear.Config(in_features=dim, out_features=dim),
            )
        )
        identity_attention = _IdentityAttention()
        attention.inner_attention = identity_attention

        x_TD = torch.randn(num_tokens, dim)
        attention_mask = create_block_mask(
            lambda b, h, q_idx, kv_idx: torch.tensor(True),
            B=1,
            H=None,
            Q_LEN=num_tokens,
            KV_LEN=num_tokens,
            device="cpu",
        )

        out_TD = attention(
            x_TD,
            rope_cache=torch.empty(0),
            rope_apply=lambda q, k, cache: (q, k),
            attention_mask=attention_mask,
        )

        self.assertEqual(identity_attention.input_shape, (num_tokens, num_heads, 4))
        self.assertEqual(out_TD.shape, (num_tokens, dim))


if __name__ == "__main__":
    unittest.main()


class _RecordDenseSP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[bool] = []

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        self.seen.append(spmd_dense_sp_enabled())
        return x


def test_vision_block_disables_dense_sp_outside_the_encoder_scope() -> None:
    """Activation remat replays a vision block during backward, after the
    encoder forward's disable_dense_sp scope has exited. The block must enter
    the scope itself so the replay picks the same TP layout as the forward."""
    import torchtitan.distributed.spmd_types as spmd_types_module

    block = VisionTransformerBlock.__new__(VisionTransformerBlock)
    torch.nn.Module.__init__(block)
    block.norm1 = torch.nn.Identity()
    block.norm2 = torch.nn.Identity()
    block.attn = _RecordDenseSP()
    block.mlp = _RecordDenseSP()

    with patch.object(
        spmd_types_module._MESH_TLS, "dense_sp_enabled", True, create=True
    ):
        # The decoder registered SP; replay runs outside the encoder forward.
        block(torch.zeros(2, 3), rope_cache=None, rope_apply=None, attention_mask=None)
        assert spmd_dense_sp_enabled()

    assert block.attn.seen == [False]
    assert block.mlp.seen == [False]
