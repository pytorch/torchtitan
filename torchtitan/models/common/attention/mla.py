# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inner attention backends for compact multi-head latent attention inputs."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from .attention import FlexInnerAttention, InnerAttention, VarlenInnerAttention

__all__ = [
    "MLAFlexInnerAttention",
    "MLAInnerAttention",
    "MLAVarlenInnerAttention",
    "materialize_mla_kv",
]

# Shape suffixes:
# T = packed token count (num_tokens)
# H = attention head count (n_heads)
# N = per-head non-positional key dimension (qk_nope_head_dim)
# R = head-shared key dimension (qk_rope_head_dim)
# V = per-head value dimension (v_head_dim)
# K = full per-head query/key dimension (N + R)
# P = packed per-head KV channels (N + V)


def materialize_mla_kv(
    q_THK: torch.Tensor,
    kv_THP: torch.Tensor,
    k_shared_TR: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize per-head K/V from packed KV and a shared key."""
    nope_head_dim = q_THK.shape[-1] - k_shared_TR.shape[-1]
    value_head_dim = kv_THP.shape[-1] - nope_head_dim
    k_nope_THN, v_THV = torch.split(kv_THP, [nope_head_dim, value_head_dim], dim=-1)
    k_shared_THR = k_shared_TR.unsqueeze(1).expand(-1, k_nope_THN.shape[1], -1)
    k_THK = torch.cat((k_nope_THN, k_shared_THR), dim=-1)
    return k_THK, v_THV


class MLAInnerAttention(InnerAttention, ABC):
    """Inner attention accepting packed KV and a head-shared key."""

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        pass

    @abstractmethod
    def forward(
        self,
        q_THK: torch.Tensor,
        kv_THP: torch.Tensor,
        k_shared_TR: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Run attention from compact Q/KV inputs."""


class MLAFlexInnerAttention(MLAInnerAttention, FlexInnerAttention):
    """Flex attention accepting compact MLA inputs."""

    @dataclass(kw_only=True, slots=True)
    class Config(MLAInnerAttention.Config, FlexInnerAttention.Config):
        pass

    def forward(  # pyrefly: ignore[bad-param-name-override]
        self,
        q_THK: torch.Tensor,
        kv_THP: torch.Tensor,
        k_shared_TR: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        k_THK, v_THV = materialize_mla_kv(q_THK, kv_THP, k_shared_TR)
        return FlexInnerAttention.forward(self, q_THK, k_THK, v_THV, **kwargs)


class MLAVarlenInnerAttention(MLAInnerAttention, VarlenInnerAttention):
    """Variable-length attention accepting compact MLA inputs."""

    @dataclass(kw_only=True, slots=True)
    class Config(MLAInnerAttention.Config, VarlenInnerAttention.Config):
        pass

    def forward(  # pyrefly: ignore[bad-param-name-override]
        self,
        q_THK: torch.Tensor,
        kv_THP: torch.Tensor,
        k_shared_TR: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        k_THK, v_THV = materialize_mla_kv(q_THK, kv_THP, k_shared_TR)
        return VarlenInnerAttention.forward(self, q_THK, k_THK, v_THV, **kwargs)
