# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from typing import Callable, ClassVar, Optional

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention,
)


class SDPA(torch.nn.Module):
    # We registered flex_attention related attributes as class variables as we
    # need to amortize the cost of compilation. Enabling per-instance flex_attention
    # is not supported.
    block_mask: ClassVar[Optional[BlockMask]] = None
    use_flex_attn: ClassVar[bool] = False
    flex_attn: ClassVar[Optional[Callable]] = None

    def __init__(self, use_flex_attn) -> None:
        super().__init__()
        self.use_flex_attn = use_flex_attn

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        if self.use_flex_attn:
            _, _, seqlen, _ = q.shape
            self._init_flex_attn(seqlen=seqlen)
            return self.flex_attn(q, k, v, block_mask=self.block_mask)
        else:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    @torch.no_grad()
    def _init_flex_attn(self, seqlen: int) -> None:
        if self.block_mask is not None:
            return

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        compiled_create_block_mask = torch.compile(create_block_mask)
        self.block_mask = compiled_create_block_mask(
            causal_mask, None, None, seqlen, seqlen
        )
        self.flex_attn = torch.compile(
            flex_attention, mode="max-autotune-no-cudagraphs"
        )
