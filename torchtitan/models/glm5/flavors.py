# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GLM-5 model flavors.

``model.py`` defines modules, while this file wires their nested ``Config``
objects, initialization policies, and the dense/MoE layer schedule.
"""

import copy
import dataclasses
from collections.abc import Callable
from functools import partial

import torch
import torch.nn as nn

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common import (
    ComplexRoPE,
    Embedding,
    HiMidLoLinear,
    LayerNorm,
    Linear,
    RMSNorm,
    Sigmoid,
    TransformerBlock,
)
from torchtitan.models.common.moe import TokenChoiceTopKRouter
from torchtitan.models.common.moe.sharding import (
    router_gate_sharding_config,
    router_sharding_config,
)
from torchtitan.models.common.config_utils import (
    DEFAULT_DEBUG_MODEL_SEQ_LEN,
    make_ffn_config,
    make_moe_config,
    make_routed_experts_config,
    make_shared_expert_ffn_config,
)
from torchtitan.models.common.decoder_sharding import (
    lm_head_sharding_config,
    norm_sharding_config,
    pre_lm_head_norm_sharding_config,
    replicated_state_sharding_config,
    rope_cache_sharding_config,
    token_embedding_sharding_config,
)
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.models.deepseek_v3.sharding import (
    mtp_block_sharding_config,
    mtp_projection_sharding_config,
)

from .model import (
    Glm5Attention,
    Glm5Model,
    Glm5TransformerBlock,
)
from .mtp import Glm5MTPTransformerBlock
from .dsa import (
    DSAIndexerTopK,
    Glm5DsaIndexer,
    Glm5FlexAttention,
    Glm5AllGatherKVFlexAttention,
)
from .sharding import (
    glm5_attention_sharding_config,
    glm5_indexer_sharding_config,
    glm5_indexer_topk_sharding_config,
    glm5_inner_attention_sharding_config,
)


DSAInnerAttention = Glm5FlexAttention

__all__ = [
    "DSAIndexerTopK",
    "DSAInnerAttention",
    "Glm5Attention",
    "Glm5DsaIndexer",
    "Glm5Model",
    "Glm5TransformerBlock",
    "Glm5MTPTransformerBlock",
    "build_glm5_layers",
    "MODEL_FLAVORS",
    "build_model_config",
    "make_glm5_attention_config",
]


def build_glm5_mtp_layers(
    inner_cfg: Glm5TransformerBlock.Config,
    *,
    dim: int,
    num_mtp_layers: int,
    norm_eps: float,
    enable_sp: bool,
) -> list[Glm5MTPTransformerBlock.Config]:
    """Build GLM NextN blocks from the released decoder-block structure."""
    norm_sharding = norm_sharding_config(enable_sp=enable_sp)
    return [
        Glm5MTPTransformerBlock.Config(
            attention=copy.deepcopy(inner_cfg.attention),
            feed_forward=copy.deepcopy(inner_cfg.feed_forward),
            attention_norm=copy.deepcopy(inner_cfg.attention_norm),
            ffn_norm=copy.deepcopy(inner_cfg.ffn_norm),
            enorm=RMSNorm.Config(
                normalized_shape=dim,
                eps=norm_eps,
                param_init=_RMS_NORM_INIT,
                sharding_config=norm_sharding,
            ),
            hnorm=RMSNorm.Config(
                normalized_shape=dim,
                eps=norm_eps,
                param_init=_RMS_NORM_INIT,
                sharding_config=norm_sharding,
            ),
            eh_proj=Linear.Config(
                in_features=2 * dim,
                out_features=dim,
                bias=False,
                param_init=_LINEAR_INIT,
                sharding_config=mtp_projection_sharding_config(enable_sp=enable_sp),
            ),
            mtp_norm=RMSNorm.Config(
                normalized_shape=dim,
                eps=norm_eps,
                param_init=_RMS_NORM_INIT,
                sharding_config=pre_lm_head_norm_sharding_config(enable_sp=enable_sp),
            ),
            sharding_config=mtp_block_sharding_config(enable_sp=enable_sp),
        )
        for _ in range(num_mtp_layers)
    ]


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_LAYER_NORM_INIT = {"weight": nn.init.ones_, "bias": nn.init.zeros_}
_RMS_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}


def _output_linear_init(dim: int) -> dict[str, Callable]:
    std = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=std, a=-3 * std, b=3 * std),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict[str, Callable]:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def _depth_experts_init(layer_id: int) -> dict[str, Callable]:
    return {
        "w1_EFD": partial(nn.init.trunc_normal_, std=0.02),
        "w2_EDF": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "w3_EFD": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
    }


class Glm5Router(TokenChoiceTopKRouter):
    """GLM group-limited sigmoid router."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(TokenChoiceTopKRouter.Config):
        num_expert_groups: int
        num_limited_groups: int

    def __init__(self, config: Config):
        super().__init__(config)
        self.num_expert_groups = config.num_expert_groups
        self.num_limited_groups = config.num_limited_groups

    def _select_experts(
        self,
        scores_TE: torch.Tensor,
        expert_bias_E: torch.Tensor | None = None,
        **router_kwargs,
    ) -> torch.Tensor:
        del router_kwargs
        experts_per_group = self.num_experts // self.num_expert_groups
        choice_scores_TE = (
            scores_TE if expert_bias_E is None else scores_TE + expert_bias_E
        )
        choice_scores_TGP = choice_scores_TE.unflatten(
            -1, (self.num_expert_groups, experts_per_group)
        )
        group_scores_TG = choice_scores_TGP.topk(2, dim=-1).values.sum(dim=-1)
        selected_groups_TL = group_scores_TG.topk(
            self.num_limited_groups, dim=-1, sorted=False
        ).indices
        excluded_groups_TG = torch.ones_like(group_scores_TG, dtype=torch.bool)
        excluded_groups_TG.scatter_(-1, selected_groups_TL, False)
        choice_scores_TE = choice_scores_TGP.masked_fill(
            excluded_groups_TG.unsqueeze(-1), float("-inf")
        ).flatten(-2)
        return choice_scores_TE.topk(self.top_k, dim=-1, sorted=False).indices


def make_glm5_router_config(
    *,
    dim: int,
    num_experts: int,
    top_k: int,
    num_expert_groups: int,
    num_limited_groups: int,
    route_scale: float,
    gate_param_init: dict[str, Callable],
) -> Glm5Router.Config:
    """Build GLM's sigmoid group-limited router using shared components."""
    return Glm5Router.Config(
        num_experts=num_experts,
        gate=HiMidLoLinear.Config(
            in_features=dim,
            out_features=num_experts,
            backward_mode="hi_mid_lo",
            bias=False,
            param_init=gate_param_init,
            sharding_config=router_gate_sharding_config(),
        ),
        top_k=top_k,
        score_func=Sigmoid.Config(),
        route_norm=True,
        route_scale=route_scale,
        num_expert_groups=num_expert_groups,
        num_limited_groups=num_limited_groups,
        sharding_config=router_sharding_config(),
    )


def make_glm5_attention_config(
    *,
    layer_id: int,
    dim: int,
    n_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    attention_dropout: float,
    mla_norm_eps: float,
    rope: ComplexRoPE.Config,
    enable_sp: bool,
    enable_cp: bool,
) -> Glm5Attention.Config:
    """Build a fully specified GLM-5 MLA plus DSA attention config.

    Configs are nested instead of constructing modules directly so TorchTitan
    can validate, shard, pipeline-partition, and initialize the model before
    materializing parameters on the target device.
    """
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    return Glm5Attention.Config(
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        wq_a=Linear.Config(
            in_features=dim,
            out_features=q_lora_rank,
            param_init=_LINEAR_INIT,
            sharding_config=replicated_state_sharding_config("weight", "bias"),
        ),
        q_norm=RMSNorm.Config(
            normalized_shape=q_lora_rank,
            eps=mla_norm_eps,
            param_init=_RMS_NORM_INIT,
            sharding_config=replicated_state_sharding_config("weight"),
        ),
        wq_b=Linear.Config(
            in_features=q_lora_rank,
            out_features=n_heads * qk_head_dim,
            param_init=_LINEAR_INIT,
            sharding_config=replicated_state_sharding_config("weight", "bias"),
        ),
        wkv_a=Linear.Config(
            in_features=dim,
            out_features=kv_lora_rank + qk_rope_head_dim,
            param_init=_LINEAR_INIT,
            sharding_config=replicated_state_sharding_config("weight", "bias"),
        ),
        kv_norm=RMSNorm.Config(
            normalized_shape=kv_lora_rank,
            eps=mla_norm_eps,
            param_init=_RMS_NORM_INIT,
            sharding_config=replicated_state_sharding_config("weight"),
        ),
        wkv_b=Linear.Config(
            in_features=kv_lora_rank,
            out_features=n_heads * (qk_nope_head_dim + v_head_dim),
            param_init=_LINEAR_INIT,
            sharding_config=replicated_state_sharding_config("weight", "bias"),
        ),
        wo=Linear.Config(
            in_features=n_heads * v_head_dim,
            out_features=dim,
            param_init=_depth_init(layer_id),
            sharding_config=replicated_state_sharding_config("weight", "bias"),
        ),
        rope=dataclasses.replace(rope, sharding_config=rope_cache_sharding_config()),
        indexer=Glm5DsaIndexer.Config(
            dim=dim,
            q_lora_rank=q_lora_rank,
            n_heads=index_n_heads,
            head_dim=index_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            index_topk=index_topk,
            wq_b=Linear.Config(
                in_features=q_lora_rank,
                out_features=index_n_heads * index_head_dim,
                param_init=_LINEAR_INIT,
                sharding_config=replicated_state_sharding_config("weight", "bias"),
            ),
            wk=Linear.Config(
                in_features=dim,
                out_features=index_head_dim,
                param_init=_LINEAR_INIT,
                sharding_config=replicated_state_sharding_config("weight", "bias"),
            ),
            k_norm=LayerNorm.Config(
                normalized_shape=index_head_dim,
                eps=1e-6,
                param_init=_LAYER_NORM_INIT,
                sharding_config=replicated_state_sharding_config("weight", "bias"),
            ),
            weights_proj=Linear.Config(
                in_features=dim,
                out_features=index_n_heads,
                param_init=_LINEAR_INIT,
                sharding_config=replicated_state_sharding_config("weight", "bias"),
            ),
            rope=dataclasses.replace(
                rope,
                sharding_config=rope_cache_sharding_config(),
            ),
            topk=DSAIndexerTopK.Config(
                index_topk=index_topk,
                softmax_scale=index_head_dim**-0.5,
                sharding_config=glm5_indexer_topk_sharding_config(),
            ),
            sharding_config=glm5_indexer_sharding_config(),
        ),
        inner_attention=(
            Glm5AllGatherKVFlexAttention.Config
            if enable_cp
            else Glm5FlexAttention.Config
        )(
            attention_dropout=attention_dropout,
            sharding_config=glm5_inner_attention_sharding_config(),
        ),
        sharding_config=glm5_attention_sharding_config(enable_sp=enable_sp),
    )


def build_glm5_layers(
    *,
    n_layers: int,
    n_dense_layers: int,
    dim: int,
    n_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    dense_hidden_dim: int,
    moe_hidden_dim: int,
    num_experts: int,
    num_shared_experts: int,
    router_top_k: int,
    router_num_expert_groups: int,
    router_num_limited_groups: int,
    router_route_scale: float,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    attention_dropout: float,
    mla_norm_eps: float = 1e-6,
    norm_eps: float = 1e-5,
    rope: ComplexRoPE.Config,
    enable_sp: bool = False,
    enable_cp: bool = False,
    enable_ep: bool = False,
) -> list[TransformerBlock.Config]:
    """Build GLM-5 dense and MoE block configurations for every layer."""
    if not 0 <= n_dense_layers <= n_layers:
        raise ValueError("n_dense_layers must be in [0, n_layers].")
    if router_num_expert_groups <= 0:
        raise ValueError("router_num_expert_groups must be > 0.")
    if num_experts % router_num_expert_groups != 0:
        raise ValueError("num_experts must divide evenly into router expert groups.")
    experts_per_group = num_experts // router_num_expert_groups
    if experts_per_group < 2:
        raise ValueError("experts_per_group must be >= 2.")
    if not 0 < router_num_limited_groups <= router_num_expert_groups:
        raise ValueError(
            "router_num_limited_groups must be in [1, router_num_expert_groups]."
        )
    if not 0 < router_top_k <= num_experts:
        raise ValueError("router_top_k must be in [1, num_experts].")
    if router_top_k > experts_per_group * router_num_limited_groups:
        raise ValueError("router_top_k exceeds the experts in limited groups.")

    layers: list[TransformerBlock.Config] = []
    for layer_id in range(n_layers):
        # Every block uses MLA+DSA attention. Only the FFN family changes:
        # early blocks are dense and the remaining blocks use routed+shared MoE.
        attention = make_glm5_attention_config(
            layer_id=layer_id,
            dim=dim,
            n_heads=n_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            attention_dropout=attention_dropout,
            mla_norm_eps=mla_norm_eps,
            rope=rope,
            enable_sp=enable_sp,
            enable_cp=enable_cp,
        )
        if layer_id < n_dense_layers:
            feed_forward = make_ffn_config(
                dim=dim,
                hidden_dim=dense_hidden_dim,
                w13_param_init=_LINEAR_INIT,
                w2_param_init=_depth_init(layer_id),
                enable_sp=enable_sp,
            )
        else:
            feed_forward = make_moe_config(
                num_experts=num_experts,
                router=make_glm5_router_config(
                    dim=dim,
                    num_experts=num_experts,
                    top_k=router_top_k,
                    num_expert_groups=router_num_expert_groups,
                    num_limited_groups=router_num_limited_groups,
                    route_scale=router_route_scale,
                    gate_param_init=_depth_init(layer_id),
                ),
                routed_experts=make_routed_experts_config(
                    dim=dim,
                    hidden_dim=moe_hidden_dim,
                    num_experts=num_experts,
                    top_k=router_top_k,
                    param_init=_depth_experts_init(layer_id),
                    enable_ep=enable_ep,
                ),
                shared_experts=make_shared_expert_ffn_config(
                    dim=dim,
                    hidden_dim=moe_hidden_dim * num_shared_experts,
                    w13_param_init=_LINEAR_INIT,
                    w2_param_init=_depth_init(layer_id),
                    enable_sp=enable_sp,
                ),
                load_balance_coeff=1e-3,
                enable_sp=enable_sp,
            )
        layers.append(
            Glm5TransformerBlock.Config(
                attention=attention,
                attention_norm=RMSNorm.Config(
                    normalized_shape=dim,
                    eps=norm_eps,
                    param_init=_RMS_NORM_INIT,
                    sharding_config=norm_sharding_config(enable_sp=enable_sp),
                ),
                ffn_norm=RMSNorm.Config(
                    normalized_shape=dim,
                    eps=norm_eps,
                    param_init=_RMS_NORM_INIT,
                    sharding_config=norm_sharding_config(enable_sp=enable_sp),
                ),
                feed_forward=feed_forward,
            )
        )
    return layers


def _debugmodel(
    *,
    n_dense_layers: int = 1,
    index_sources: tuple[int, ...] = (),
    seq_len: int = 128,
    parallelism: ParallelismConfig | None = None,
) -> Glm5Model.Config:
    dim = 256
    vocab_size = 2048
    rope = ComplexRoPE.Config(
        dim=32,
        max_context_length=seq_len,
        theta=1_000_000,
        scaling="none",
    )
    return Glm5Model.Config(
        max_context_length=seq_len,
        vocab_size=vocab_size,
        dim=dim,
        index_sources=index_sources,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(
                enable_sp=bool(parallelism and parallelism.enable_sequence_parallel)
            ),
        ),
        norm=RMSNorm.Config(
            normalized_shape=dim,
            param_init=_RMS_NORM_INIT,
            sharding_config=pre_lm_head_norm_sharding_config(
                enable_sp=bool(parallelism and parallelism.enable_sequence_parallel)
            ),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=build_glm5_layers(
            n_layers=8,
            n_dense_layers=n_dense_layers,
            dim=dim,
            n_heads=8,
            q_lora_rank=128,
            kv_lora_rank=64,
            qk_nope_head_dim=32,
            qk_rope_head_dim=32,
            v_head_dim=64,
            dense_hidden_dim=1024,
            moe_hidden_dim=256,
            num_experts=8,
            num_shared_experts=1,
            router_top_k=2,
            router_num_expert_groups=1,
            router_num_limited_groups=1,
            router_route_scale=2.5,
            index_n_heads=4,
            index_head_dim=64,
            index_topk=8,
            attention_dropout=0.0,
            rope=rope,
            enable_sp=bool(parallelism and parallelism.enable_sequence_parallel),
            enable_cp=bool(parallelism and parallelism.context_parallel_degree > 1),
            enable_ep=bool(parallelism and parallelism.expert_parallel_degree > 1),
        ),
    )


def _shared_dsa_debugmodel(
    *,
    seq_len: int = 128,
    parallelism: ParallelismConfig | None = None,
) -> Glm5Model.Config:
    return _debugmodel(
        index_sources=(0, 0, 2, 2, 4, 4, 6, 6),
        seq_len=seq_len,
        parallelism=parallelism,
    )


def _glm5_2_debugmodel(
    *,
    seq_len: int = 128,
    parallelism: ParallelismConfig | None = None,
) -> Glm5Model.Config:
    """Build a small model that preserves GLM-5.2 layer-type transitions."""
    return _debugmodel(
        n_dense_layers=3,
        index_sources=(0, 1, 2, 2, 2, 2, 6, 6),
        seq_len=seq_len,
        parallelism=parallelism,
    )


def _glm5_2_debugmodel_mtp(
    *,
    seq_len: int = 128,
    parallelism: ParallelismConfig | None = None,
) -> Glm5Model.Config:
    """Small GLM-5.2 training model with one faithful NextN/MTP block."""
    config = _glm5_2_debugmodel(seq_len=seq_len, parallelism=parallelism)
    enable_sp = bool(parallelism and parallelism.enable_sequence_parallel)
    layers = [
        dataclasses.replace(layer, preserve_output_for_mtp=True)
        for layer in config.layers
    ]
    return dataclasses.replace(
        config,
        layers=layers,
        mtp_layers=build_glm5_mtp_layers(
            layers[-1],
            dim=config.dim,
            num_mtp_layers=1,
            norm_eps=1e-5,
            enable_sp=enable_sp,
        ),
    )


def _glm5_2_scaled_debugmodel(
    *,
    seq_len: int = 4096,
    parallelism: ParallelismConfig | None = None,
) -> Glm5Model.Config:
    """Build a larger GLM-5.2 validation model with the released structure.

    This flavor is sized for single-device training on the smaller validation
    accelerator while retaining enough depth for every eight-way topology.
    Width, vocabulary, and routed-expert count are reduced because one released
    256-expert MoE layer alone cannot fit the single-device validation contract.
    """
    dim = 768
    vocab_size = 32768
    num_layers = 16
    index_sources = tuple(
        layer_id
        if layer_id < 3 or (layer_id - 2) % 4 == 0
        else max(
            source
            for source in range(layer_id)
            if source < 3 or (source - 2) % 4 == 0
        )
        for layer_id in range(num_layers)
    )
    rope = ComplexRoPE.Config(
        dim=64,
        max_context_length=seq_len,
        theta=8_000_000,
        scaling="none",
    )
    return Glm5Model.Config(
        max_context_length=seq_len,
        vocab_size=vocab_size,
        dim=dim,
        index_sources=index_sources,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(
                enable_sp=bool(parallelism and parallelism.enable_sequence_parallel)
            ),
        ),
        norm=RMSNorm.Config(
            normalized_shape=dim,
            eps=1e-5,
            param_init=_RMS_NORM_INIT,
            sharding_config=pre_lm_head_norm_sharding_config(
                enable_sp=bool(parallelism and parallelism.enable_sequence_parallel)
            ),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=build_glm5_layers(
            n_layers=num_layers,
            n_dense_layers=3,
            dim=dim,
            n_heads=8,
            q_lora_rank=384,
            kv_lora_rank=192,
            qk_nope_head_dim=192,
            qk_rope_head_dim=64,
            v_head_dim=256,
            dense_hidden_dim=1536,
            moe_hidden_dim=384,
            num_experts=24,
            num_shared_experts=1,
            router_top_k=8,
            router_num_expert_groups=1,
            router_num_limited_groups=1,
            router_route_scale=2.5,
            index_n_heads=8,
            index_head_dim=128,
            index_topk=512,
            attention_dropout=0.0,
            mla_norm_eps=1e-5,
            norm_eps=1e-5,
            rope=rope,
            enable_sp=bool(parallelism and parallelism.enable_sequence_parallel),
            enable_cp=bool(parallelism and parallelism.context_parallel_degree > 1),
            enable_ep=bool(parallelism and parallelism.expert_parallel_degree > 1),
        ),
    )


def _glm5_2_744b_a40b(
    *,
    seq_len: int = 1_048_576,
    parallelism: ParallelismConfig | None = None,
) -> Glm5Model.Config:
    """Build the released GLM-5.2 744B-A40B decoder architecture.

    The values mirror zai-org/GLM-5.2 ``config.json``: 78 main decoder layers,
    the released cross-layer index-sharing schedule, and one full NextN/MTP
    block that shares token embeddings and the language-model head.
    """
    dim = 6144
    vocab_size = 154880
    num_layers = 78
    index_sources = tuple(
        layer_id
        if max(layer_id - 2, 0) % 4 == 0
        else max(source for source in range(layer_id) if max(source - 2, 0) % 4 == 0)
        for layer_id in range(num_layers)
    )
    rope = ComplexRoPE.Config(
        dim=64,
        max_context_length=seq_len,
        theta=8_000_000,
        scaling="none",
    )
    layers = build_glm5_layers(
        n_layers=num_layers,
        n_dense_layers=3,
        dim=dim,
        n_heads=64,
        q_lora_rank=2048,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        dense_hidden_dim=12288,
        moe_hidden_dim=2048,
        num_experts=256,
        num_shared_experts=1,
        router_top_k=8,
        router_num_expert_groups=1,
        router_num_limited_groups=1,
        router_route_scale=2.5,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=2048,
        attention_dropout=0.0,
        mla_norm_eps=1e-5,
        norm_eps=1e-5,
        rope=rope,
        enable_sp=bool(parallelism and parallelism.enable_sequence_parallel),
        enable_cp=bool(parallelism and parallelism.context_parallel_degree > 1),
        enable_ep=bool(parallelism and parallelism.expert_parallel_degree > 1),
    )
    layers = [
        dataclasses.replace(layer, preserve_output_for_mtp=True) for layer in layers
    ]
    return Glm5Model.Config(
        max_context_length=seq_len,
        vocab_size=vocab_size,
        dim=dim,
        index_sources=index_sources,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(
                enable_sp=bool(parallelism and parallelism.enable_sequence_parallel)
            ),
        ),
        norm=RMSNorm.Config(
            normalized_shape=dim,
            eps=1e-5,
            param_init=_RMS_NORM_INIT,
            sharding_config=pre_lm_head_norm_sharding_config(
                enable_sp=bool(parallelism and parallelism.enable_sequence_parallel)
            ),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=layers,
        mtp_layers=build_glm5_mtp_layers(
            layers[-1],
            dim=dim,
            num_mtp_layers=1,
            norm_eps=1e-5,
            enable_sp=bool(parallelism and parallelism.enable_sequence_parallel),
        ),
    )


MODEL_FLAVORS = {
    "744B-A40B": _glm5_2_744b_a40b,
    "debugmodel": _debugmodel,
    "dsa_debugmodel": _debugmodel,
    "glm5_2_debugmodel": _glm5_2_debugmodel,
    "glm5_2_debugmodel_mtp": _glm5_2_debugmodel_mtp,
    "glm5_2_scaled_debugmodel": _glm5_2_scaled_debugmodel,
    "shared_dsa_debugmodel": _shared_dsa_debugmodel,
}


def build_model_config(
    flavor: str,
    *,
    parallelism: ParallelismConfig,
    seq_len: int | None = None,
) -> Glm5Model.Config:
    """Build one GLM-5 flavor using TorchTitan's current model contract."""
    if flavor not in MODEL_FLAVORS:
        raise ValueError(
            f"Unknown glm5 flavor: {flavor}. Available: {list(MODEL_FLAVORS)}"
        )
    get_config = MODEL_FLAVORS[flavor]
    max_context_len = {
        "744B-A40B": 1_048_576,
        "glm5_2_scaled_debugmodel": 4096,
    }.get(flavor, DEFAULT_DEBUG_MODEL_SEQ_LEN)
    context_len = seq_len or max_context_len
    if context_len > max_context_len:
        raise ValueError(
            f"Requested seq_len {context_len} exceeds max context length "
            f"{max_context_len} for flavor {flavor}"
        )
    return get_config(
        seq_len=context_len,
        parallelism=parallelism,
    )
