# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Attention backend for compact multi-head latent attention inputs."""

from dataclasses import dataclass

import torch

from torchtitan.models.common.linear import Linear

from .attention import AttentionBackend, InnerAttention, local_head_split

__all__ = [
    "MLAAttention",
    "materialize_mla_kv",
    "register_mla_wkv_b_state_dict_hooks",
]

# Shape suffixes:
# T = packed token count (num_tokens)
# H = attention head count (n_heads)
# N = per-head non-positional key dimension (qk_nope_head_dim)
# R = head-shared key dimension (qk_rope_head_dim)
# V = per-head value dimension (v_head_dim)
# K = full per-head query/key dimension (N + R)
# P = packed per-head KV channels (N + V)


def register_mla_wkv_b_state_dict_hooks(module: torch.nn.Module) -> None:
    """Preserve the outer MLA checkpoint key for the backend-owned projection."""

    def state_dict_post_hook(module, state_dict, prefix, local_metadata):
        del module, local_metadata
        nested_prefix = f"{prefix}mla_attention.wkv_b."
        outer_prefix = f"{prefix}wkv_b."
        for key in tuple(state_dict):
            if key.startswith(nested_prefix):
                suffix = key.removeprefix(nested_prefix)
                state_dict[f"{outer_prefix}{suffix}"] = state_dict.pop(key)

    def load_state_dict_pre_hook(
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        del module
        del local_metadata
        del strict
        del missing_keys
        del unexpected_keys
        del error_msgs
        outer_prefix = f"{prefix}wkv_b."
        nested_prefix = f"{prefix}mla_attention.wkv_b."
        for key in tuple(state_dict):
            if key.startswith(outer_prefix):
                suffix = key.removeprefix(outer_prefix)
                state_dict[f"{nested_prefix}{suffix}"] = state_dict.pop(key)

    module.register_state_dict_post_hook(state_dict_post_hook)
    module.register_load_state_dict_pre_hook(load_state_dict_pre_hook)


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


class MLAAttention(AttentionBackend):
    """Project compact MLA inputs and delegate materialized Q/K/V attention."""

    @dataclass(kw_only=True, slots=True)
    class Config(AttentionBackend.Config):
        wkv_b: Linear.Config
        packed_kv_head_dim: int
        inner_attention: InnerAttention.Config

        @property
        def inner_attention_config(self) -> InnerAttention.Config:
            """Return the composed Q/K/V attention config."""
            return self.inner_attention

        def build_attention_metadata(self, *args, **kwargs):
            """Build metadata for the composed Q/K/V attention kernel."""
            return self.inner_attention.build_attention_metadata(*args, **kwargs)

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.wkv_b = config.wkv_b.build()
        self.packed_kv_head_dim = config.packed_kv_head_dim
        self.inner_attention = config.inner_attention.build()
        self.attention_metadata_key = self.inner_attention.attention_metadata_key

    def project_mla_kv(
        self,
        kv_c_normed_TL: torch.Tensor,
    ) -> torch.Tensor:
        """Up-project a normalized MLA latent into packed per-head K/V."""
        projected_kv_TD = self.wkv_b(kv_c_normed_TL)
        return local_head_split(
            projected_kv_TD,
            self.packed_kv_head_dim,
            cp_shard_dim=0,
        )

    def forward(
        self,
        q_THK: torch.Tensor,
        kv_c_normed_TL: torch.Tensor,
        k_shared_TR: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Project and run attention from compact Q/KV inputs."""
        kv_THP = self.project_mla_kv(kv_c_normed_TL)
        k_THK, v_THV = materialize_mla_kv(q_THK, kv_THP, k_shared_TR)
        return self.inner_attention(q_THK, k_THK, v_THV, **kwargs)
