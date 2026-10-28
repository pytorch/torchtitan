# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Attention backend for compact multi-head latent attention inputs."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import spmd_types as spmd
import torch
import torch_remat as remat

from torchtitan.models.common.linear import Linear

from .attention import InnerAttention

if TYPE_CHECKING:
    from . import AttentionMetadata

__all__ = [
    "MLAAttention",
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


class MLAAttention(InnerAttention):
    """Project compact MLA inputs and delegate materialized Q/K/V attention."""

    inner_attention: InnerAttention

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        wkv_b: Linear.Config
        packed_kv_head_dim: int
        inner_attention: InnerAttention.Config

        @property
        def is_full_attention(self) -> bool:
            return self.inner_attention.is_full_attention

        def build_attention_metadata(
            self,
            positions: torch.Tensor,
            *,
            padding_mask: torch.Tensor | None = None,
            max_num_documents: int | None = None,
            max_context_length: int | None = None,
        ) -> "AttentionMetadata | None":
            return self.inner_attention.build_attention_metadata(
                positions,
                padding_mask=padding_mask,
                max_num_documents=max_num_documents,
                max_context_length=max_context_length,
            )

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.wkv_b = config.wkv_b.build()
        self.packed_kv_head_dim = config.packed_kv_head_dim
        self.inner_attention = config.inner_attention.build()

    def remat_region_name(self, local_name: str) -> str:
        # Preserve the outer attention's kernel policy name inside the MLA backend.
        if local_name == "inner_attention":
            return self._remat_module_fqn or local_name
        return super().remat_region_name(local_name)

    def project_mla_kv(
        self,
        kv_c_normed_TL: torch.Tensor,
    ) -> torch.Tensor:
        """Up-project a normalized MLA latent into packed per-head K/V."""
        projected_kv_TD = self.wkv_b(kv_c_normed_TL)
        # Splitting local heads preserves the backend's current CP placement.
        with spmd.local():
            return projected_kv_TD.view(
                projected_kv_TD.shape[0], -1, self.packed_kv_head_dim
            )

    def forward(
        self,
        q_THK: torch.Tensor,
        kv_c_normed_TL: torch.Tensor,
        k_shared_TR: torch.Tensor,
        *,
        materialize_kv=materialize_mla_kv,
        **kwargs,
    ) -> torch.Tensor:
        """Project and run attention from compact Q/KV inputs."""
        kv_THP = self.project_mla_kv(kv_c_normed_TL)
        # Materialization reads saved projection/communication outputs outside a region.
        remat.recompute_needs_tensor(kv_THP, k_shared_TR)
        k_THK, v_THV = materialize_kv(q_THK, kv_THP, k_shared_TR)
        return remat.region(
            self.inner_attention,
            self.remat_region_name("inner_attention"),
            recompute=self.remat_should_recompute("inner_attention"),
        )(q_THK, k_THK, v_THV, **kwargs)
