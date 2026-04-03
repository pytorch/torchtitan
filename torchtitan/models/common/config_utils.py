# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared config builder helpers for model registries.

These helpers construct fully-specified sub-configs with all dimensional
fields set at config creation time.
"""

from collections.abc import Callable
from typing import Literal

from torchtitan.models.common.attention import GQAttention, LocalMapInnerAttention
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import GroupedExperts, MoE, TokenChoiceTopKRouter
from torchtitan.models.common.rmsnorm import RMSNorm


def make_gqa_config(
    *,
    dim: int,
    n_heads: int,
    wqkv_param_init: dict[str, Callable],
    wo_param_init: dict[str, Callable],
    inner_attention: LocalMapInnerAttention.Config,
    n_kv_heads: int | None = None,
    head_dim: int | None = None,
    use_rope: bool = True,
    mask_type: str = "causal",
    rope_backend: str = "complex",
    q_norm: RMSNorm.Config | None = None,
    k_norm: RMSNorm.Config | None = None,
) -> GQAttention.Config:
    """Build a fully-specified GQAttention.Config."""
    n_kv = n_kv_heads if n_kv_heads is not None else n_heads
    hd = head_dim if head_dim is not None else dim // n_heads
    return GQAttention.Config(
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        dim=dim,
        wq=Linear.Config(
            in_features=dim, out_features=n_heads * hd, param_init=wqkv_param_init
        ),
        wkv=Linear.Config(
            in_features=dim, out_features=n_kv * hd, param_init=wqkv_param_init
        ),
        wo=Linear.Config(
            in_features=n_heads * hd, out_features=dim, param_init=wo_param_init
        ),
        q_norm=q_norm,
        k_norm=k_norm,
        use_rope=use_rope,
        inner_attention=inner_attention,
        mask_type=mask_type,
        rope_backend=rope_backend,
    )


def make_ffn_config(
    *,
    dim: int,
    hidden_dim: int,
    w1_param_init: dict[str, Callable],
    w2w3_param_init: dict[str, Callable],
) -> FeedForward.Config:
    """Build a fully-specified FeedForward.Config."""
    return FeedForward.Config(
        w1=Linear.Config(
            in_features=dim, out_features=hidden_dim, param_init=w1_param_init
        ),
        w2=Linear.Config(
            in_features=hidden_dim, out_features=dim, param_init=w2w3_param_init
        ),
        w3=Linear.Config(
            in_features=dim, out_features=hidden_dim, param_init=w2w3_param_init
        ),
    )


def make_moe_config(
    *,
    num_experts: int = 8,
    router: TokenChoiceTopKRouter.Config,
    experts: GroupedExperts.Config,
    shared_experts: FeedForward.Config | None = None,
    score_before_experts: bool = True,
    load_balance_coeff: float | None = 1e-3,
) -> MoE.Config:
    """Build a fully-specified MoE.Config."""
    return MoE.Config(
        num_experts=num_experts,
        score_before_experts=score_before_experts,
        load_balance_coeff=load_balance_coeff,
        router=router,
        experts=experts,
        shared_experts=shared_experts,
    )


def make_router_config(
    *,
    dim: int,
    num_experts: int,
    gate_param_init: dict[str, Callable],
    top_k: int = 1,
    score_func: Literal["sigmoid", "softmax"] = "sigmoid",
    route_norm: bool = False,
    route_scale: float = 1.0,
    num_expert_groups: int | None = None,
    num_limited_groups: int | None = None,
    bias: bool = False,
) -> TokenChoiceTopKRouter.Config:
    """Build a fully-specified TokenChoiceTopKRouter.Config."""
    return TokenChoiceTopKRouter.Config(
        num_experts=num_experts,
        gate=Linear.Config(
            in_features=dim,
            out_features=num_experts,
            bias=bias,
            param_init=gate_param_init,
        ),
        top_k=top_k,
        score_func=score_func,
        route_norm=route_norm,
        route_scale=route_scale,
        num_expert_groups=num_expert_groups,
        num_limited_groups=num_limited_groups,
    )


def make_experts_config(
    *,
    dim: int,
    hidden_dim: int,
    num_experts: int,
    param_init: dict[str, Callable],
    use_grouped_mm: bool = True,
) -> GroupedExperts.Config:
    """Build a fully-specified GroupedExperts.Config."""
    return GroupedExperts.Config(
        dim=dim,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        use_grouped_mm=use_grouped_mm,
        param_init=param_init,
    )
