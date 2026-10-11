# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SDPA support for graph-trainer tests and numerical comparison scripts."""

# Shape suffix legend:
#   B = batch, L = sequence length, H = attention heads,
#   K = query/key head dimension, V = value head dimension.

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

from torchtitan.models.common.attention import InnerAttention


class ScaledDotProductInnerAttention(InnerAttention):
    """Inner attention using ``F.scaled_dot_product_attention`` with CP support.

    ``forward()`` adapts Q/K from ``(B, L, H, K)`` to ``(B, H, L, K)`` and V
    from ``(B, L, H, V)`` to ``(B, H, L, V)``, then converts the result back to
    ``(B, L, H, V)``.

    Note:
        The forward function must have q, k, v as the first three arguments to be
        compatible with _ContextParallel.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        pass

    sdpa_backends: list[SDPBackend] = []

    def __init__(self, config: Config) -> None:
        if config is None:
            config = ScaledDotProductInnerAttention.Config()
        super().__init__()
        if not self.sdpa_backends:
            self.sdpa_backends = [
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.MATH,
            ]

    def forward(
        self,
        q_BLHK: torch.Tensor,
        k_BLHK: torch.Tensor,
        v_BLHV: torch.Tensor,
        *,
        attention_metadata: None = None,
        scale: float | None = None,
        enable_gqa: bool = False,
        is_causal: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        if attention_metadata is not None:
            raise ValueError(
                "ScaledDotProductInnerAttention does not support attention_metadata; it "
                "only supports causal/non-causal attention via is_causal."
            )
        q_BHLK, k_BHLK, v_BHLV = (
            q_BLHK.transpose(1, 2),
            k_BLHK.transpose(1, 2),
            v_BLHV.transpose(1, 2),
        )
        with sdpa_kernel(self.sdpa_backends, set_priority=True):
            out_BHLV = F.scaled_dot_product_attention(
                q_BHLK,
                k_BHLK,
                v_BHLV,
                scale=scale,
                is_causal=is_causal,
                enable_gqa=enable_gqa,
            )
        return out_BHLV.transpose(1, 2)
