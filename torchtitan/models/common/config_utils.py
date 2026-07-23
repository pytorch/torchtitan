# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared config builder helpers for model registries.

These helpers construct fully-specified sub-configs with all dimensional
fields set at config creation time.
"""

import dataclasses
from collections.abc import Callable
from typing import Literal

import torch
from torch.distributed.tensor import DTensor

from torchtitan.distributed.spmd_types import current_spmd_mesh, spmd_mesh_size
from torchtitan.models.common.attention import (
    FlexAttention,
    FusedQKVLinear,
    GQAttention,
    QKVLinear,
    VarlenAttention,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import (
    GroupedExperts,
    MoE,
    RoutedExperts,
    TokenChoiceTopKRouter,
)
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.models.common.token_dispatcher import (
    AllToAllTokenDispatcher,
    DeepEPTokenDispatcher,
    HybridEPTokenDispatcher,
    LocalTokenDispatcher,
    MinimalAsyncEPTokenDispatcher,
)
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.protocols.module import Module


def decoder_vocab_size(model_spec: ModelSpec) -> int:
    """Assert Decoder.Config type so lint is not annoyed."""
    model_config = model_spec.model
    assert isinstance(model_config, Decoder.Config)
    return model_config.vocab_size


def get_attention_config(
    backend: str,
) -> Module.Config:
    """Map backend string to an inner_attention config.

    Language models always use block_causal masking (the dataloaders always
    emit per-document positions), so every backend here is a masked attention
    backend. ``ScaledDotProductAttention`` only supports a boolean ``is_causal``
    flag and cannot consume per-document positions, so it is not a valid
    language-model backend (it remains available for Flux, which builds it
    directly).
    """
    if backend == "flex":
        return FlexAttention.Config()
    elif backend == "flex_flash":
        from torchtitan.tools.utils import has_cuda_capability

        if not has_cuda_capability(9, 0):
            raise ValueError(
                "Flash backend of FlexAttention is only supported on Hopper or Blackwell"
            )
        return FlexAttention.Config(
            block_size=(256, 128), kernel_options={"BACKEND": "FLASH"}
        )
    elif backend == "varlen":
        return VarlenAttention.Config()
    elif backend == "sdpa":
        raise ValueError(
            "sdpa is no longer supported for language models; positions are "
            "always available so use flex, flex_flash, or varlen."
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def _fused_qkv_param_init(
    base_param_init: dict[str, Callable],
    *,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> dict[str, Callable]:
    """Init for the fused ``wqkv`` that is bit-identical to the stock separate
    ``wq``/``wk``/``wv`` and independent of the sharding degree.

    The non-fused module initializes ``wq``/``wk``/``wv`` as three separate
    contiguous parameters. This reproduces those exact draws: it initializes q,
    k, v as three contiguous tensors of the stock-weight shapes (in
    ``QKVLinear``'s ``wq``/``wk``/``wv`` build order), then assembles them into
    the fused ``(n_kv_heads, R, head_dim, dim)`` layout (``R = heads_per_kv + 2``)
    -- the same concatenation ``_merge_qkv_on_load`` uses -- and copies into the
    buffer.

    Parallelism-agnostic RNG: at init ``t`` is the (possibly sharded) param --
    e.g. a ``Shard(0)`` DTensor for the colwise wqkv. ``t.new_empty(...)``
    returns ``Replicate`` DTensors, so each ``base_init`` runs on the full tensor
    and draws the same values on every rank (the weights do not depend on the
    TP/FSDP degree). ``cat`` of ``Replicate`` stays ``Replicate``, and the final
    ``copy_`` scatters it into the sharded ``t`` (each rank keeps its shard). So
    the path is "init replicated, then shard," which both matches the non-fused
    module and keeps RNG independent of the parallelism.
    """
    heads_per_kv = n_heads // n_kv_heads

    def _make_init(base_init: Callable) -> Callable:
        # ``tail`` is the per-row shape: () for bias, (in_features,) for weight.
        # Building q/k/v with the exact stock shapes and drawing them in
        # wq/wk/wv order keeps the RNG sequence identical to the non-fused module.
        def _init(t):
            tail = t.shape[1:]
            # If t is a sharded DTensor, new_empty (with the full stock shape)
            # returns Replicate DTensors, so base_init runs replicated and draws
            # the same values on every rank (parallelism-agnostic RNG).
            q = t.new_empty(n_heads * head_dim, *tail)
            k = t.new_empty(n_kv_heads * head_dim, *tail)
            v = t.new_empty(n_kv_heads * head_dim, *tail)
            base_init(q)
            base_init(k)
            base_init(v)
            fused = torch.cat(
                [
                    q.view(n_kv_heads, heads_per_kv, head_dim, *tail),
                    k.view(n_kv_heads, 1, head_dim, *tail),
                    v.view(n_kv_heads, 1, head_dim, *tail),
                ],
                dim=1,
            ).view(-1, *tail)
            with torch.no_grad():
                # fused is Replicate (cat of Replicates). Flatten it to t's native
                # 2D [(n_kv_heads*r_dim*head_dim), *tail] shape and copy_ into t,
                # which scatters each rank's own shard. Reshaping the Replicate
                # source (instead of t.view(n_kv_heads, ...) on the sharded t)
                # avoids the "unflatten unevenly sharded" error when dp_shard*tp
                # does not divide n_kv_heads (e.g. dp_shard=8, n_kv_heads=4); no
                # gather, since fused is already replicated.
                if not isinstance(t, DTensor) and (tp_size := spmd_mesh_size("tp")) > 1:
                    # RL generator init_weights() only needs non-persistent
                    # buffers; weights come from trainer state dict. Until it has
                    # a DTensor static state dict path, copy this TP shard here.
                    # TODO: Remove once RL can init buffers without weight init.
                    mesh = current_spmd_mesh()
                    assert mesh is not None
                    tp_rank = mesh.get_local_rank("tp")
                    fused = fused.chunk(tp_size, dim=0)[tp_rank]
                t.copy_(fused)

        return _init

    out: dict[str, Callable] = {}
    for param in ("weight", "bias"):
        base_init = base_param_init.get(param)
        if base_init is not None:
            out[param] = _make_init(base_init)
    return out


def make_gqa_config(
    *,
    dim: int,
    n_heads: int,
    wqkv_param_init: dict[str, Callable],
    wo_param_init: dict[str, Callable],
    inner_attention: Module.Config,
    rope: RoPE.Config,
    n_kv_heads: int | None = None,
    head_dim: int | None = None,
    fuse_qkv: bool = False,
    qk_norm: RMSNorm.Config | None = None,
) -> GQAttention.Config:
    """Build a fully-specified GQAttention.Config."""
    n_kv = n_kv_heads if n_kv_heads is not None else n_heads
    per_head_dim = head_dim if head_dim is not None else dim // n_heads
    rope = dataclasses.replace(rope)

    if fuse_qkv:
        qkv = FusedQKVLinear.Config(
            head_dim=per_head_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv,
            wqkv=Linear.Config(
                in_features=dim,
                out_features=(n_heads + 2 * n_kv) * per_head_dim,
                # Per-slice init so the fused wqkv is bit-identical to the stock
                # separate wq/wk/wv (see _fused_qkv_param_init).
                param_init=_fused_qkv_param_init(
                    wqkv_param_init,
                    n_heads=n_heads,
                    n_kv_heads=n_kv,
                    head_dim=per_head_dim,
                ),
            ),
        )
    else:
        qkv = QKVLinear.Config(
            head_dim=per_head_dim,
            wq=Linear.Config(
                in_features=dim,
                out_features=n_heads * per_head_dim,
                param_init=wqkv_param_init,
            ),
            wkv=Linear.Config(
                in_features=dim,
                out_features=n_kv * per_head_dim,
                param_init=wqkv_param_init,
            ),
        )

    return GQAttention.Config(
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        dim=dim,
        qkv_linear=qkv,
        wo=Linear.Config(
            in_features=n_heads * per_head_dim,
            out_features=dim,
            param_init=wo_param_init,
        ),
        qk_norm=qk_norm,
        inner_attention=inner_attention,
        rope=rope,
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
    routed_experts: RoutedExperts.Config,
    shared_experts: FeedForward.Config | None = None,
    load_balance_coeff: float | None = 1e-3,
) -> MoE.Config:
    """Build a fully-specified MoE.Config."""
    return MoE.Config(
        num_experts=num_experts,
        load_balance_coeff=load_balance_coeff,
        router=router,
        routed_experts=routed_experts,
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


def make_token_dispatcher_config(
    *,
    num_experts: int,
    top_k: int,
    comm_backend: str,
    non_blocking_capacity_factor: float | None = None,
    hidden_dim: int | None = None,
    num_max_tokens_per_rank: int | None = None,
    cudagraphable: bool = False,
) -> LocalTokenDispatcher.Config:
    """Build the appropriate token dispatcher config.

    Returns the right Config subclass based on comm_backend:
    - "standard": Uses PyTorch all-to-all collectives (falls back to local
      dispatch when EP=1, i.e. ep_mesh is None at runtime)
    - "deepep": Uses DeepEP custom kernels for H100/NVLink Switch
    - "hybridep": Uses HybridEP with TMA optimization for GB200/NVLink72
    - "minimal_async_ep": Uses MinimalAsyncEP for constrained DP>=EP

    DeepEP/HybridEP requires installation:
    https://github.com/deepseek-ai/DeepEP

    For HybridEP, SM configuration can be set via environment variables:
    - HYBRIDEP_NUM_SMS_DISPATCH (default: 16)
    - HYBRIDEP_NUM_SMS_COMBINE (default: 16)
    """
    # TODO(unify-ep-dispatch-knobs): unify the per-backend static-shape/cudagraph knobs --
    # HybridEP non_blocking_capacity_factor vs DeepEP cudagraphable + num_max_tokens_per_rank.
    if comm_backend == "deepep":
        # DeepEP v2: a single ElasticBuffer handles training and inference. ``hidden_dim``
        # (model dim) sizes the buffer; wire_meshes creates it eagerly. ``cudagraphable``
        # selects the static no-host-sync expand layout (set on the generator by the
        # deepep_override). ``num_max_tokens_per_rank`` is the per-rank EXPAND
        # capacity: training infers it (the compact path auto-sizes), inference must set it
        # >= the largest per-rank token count for droplessness.
        return DeepEPTokenDispatcher.Config(
            num_experts=num_experts,
            top_k=top_k,
            hidden_dim=hidden_dim,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            cudagraphable=cudagraphable,
        )
    elif comm_backend == "hybridep":
        return HybridEPTokenDispatcher.Config(
            num_experts=num_experts,
            top_k=top_k,
            non_blocking_capacity_factor=non_blocking_capacity_factor,
        )
    elif comm_backend == "minimal_async_ep":
        return MinimalAsyncEPTokenDispatcher.Config(
            num_experts=num_experts,
            top_k=top_k,
        )
    elif comm_backend == "standard":
        return AllToAllTokenDispatcher.Config(
            num_experts=num_experts,
            top_k=top_k,
        )
    else:
        raise ValueError(
            f"Unknown comm_backend: '{comm_backend}'. "
            "Must be one of 'standard', 'deepep', 'hybridep', 'minimal_async_ep'."
        )


def make_routed_experts_config(
    *,
    dim: int,
    hidden_dim: int,
    num_experts: int,
    top_k: int,
    param_init: dict[str, Callable],
    comm_backend: str,
    non_blocking_capacity_factor: float | None = None,
    num_max_tokens_per_rank: int | None = None,
    cudagraphable: bool = False,
) -> RoutedExperts.Config:
    """Build a fully-specified RoutedExperts.Config (inner_experts + token_dispatcher)."""
    return RoutedExperts.Config(
        inner_experts=GroupedExperts.Config(
            dim=dim,
            hidden_dim=hidden_dim,
            num_experts=num_experts,
            param_init=param_init,
        ),
        token_dispatcher=make_token_dispatcher_config(
            num_experts=num_experts,
            top_k=top_k,
            comm_backend=comm_backend,
            non_blocking_capacity_factor=non_blocking_capacity_factor,
            hidden_dim=dim,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            cudagraphable=cudagraphable,
        ),
    )
