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
from typing import Literal

import torch.nn as nn

from torchtitan.config import derive
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform.context_parallel import ContextParallelTransform
from torchtitan.models.common import (
    ComplexRoPE,
    Embedding,
    LayerNorm,
    Linear,
    RMSNorm,
    Sigmoid,
)
from torchtitan.models.common.attention import MLAAttention
from torchtitan.models.common.config_utils import DEFAULT_DEBUG_MODEL_SEQ_LEN
from torchtitan.models.common.decoder_sharding import (
    lm_head_sharding_config,
    pre_lm_head_norm_sharding_config,
    replicated_state_sharding_config,
    rope_cache_sharding_config,
    token_embedding_sharding_config,
)
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.models.deepseek_v3.attention import DeepSeekMLAAttention
from torchtitan.models.deepseek_v3.flavors import (
    _build_mtp_layers,
    build_mla_moe_layers,
    make_deepseek_v3_router_config,
)
from .dsa import (
    DSAIndexerTopK,
    Glm5CPMLAAttention,
    Glm5DsaIndexer,
    Glm5IndexerLoss,
    Glm5SparseAttention,
)

from .model import Glm5Attention, Glm5Model, Glm5TransformerBlock
from .mtp import Glm5MTPTransformerBlock
from .sharding import (
    glm5_attention_sharding_config,
    glm5_indexer_sharding_config,
    glm5_indexer_topk_sharding_config,
    glm5_inner_attention_sharding_config,
)


__all__ = [
    "DSAIndexerTopK",
    "Glm5Attention",
    "Glm5DsaIndexer",
    "Glm5Model",
    "Glm5TransformerBlock",
    "Glm5MTPTransformerBlock",
    "build_glm5_layers",
    "MODEL_FLAVORS",
    "build_model_config",
]


def build_glm5_mtp_layers(
    inner_cfg: Glm5TransformerBlock.Config,
    *,
    indexer: Glm5DsaIndexer.Config,
    dim: int,
    num_mtp_layers: int,
    norm_eps: float,
    enable_sp: bool,
) -> list[Glm5MTPTransformerBlock.Config]:
    """Build GLM NextN blocks from the released decoder-block structure."""
    layers = _build_mtp_layers(
        inner_cfg, dim=dim, num_mtp_layers=num_mtp_layers, enable_sp=enable_sp
    )
    for layer in layers:
        assert isinstance(layer.attention, Glm5Attention.Config)
        layer.attention.indexer = copy.deepcopy(indexer)
        for norm in (layer.enorm, layer.hnorm, layer.mtp_norm):
            norm.eps = norm_eps
            norm.param_init = _RMS_NORM_INIT
        layer.eh_proj.param_init = _LINEAR_INIT
    return [derive(layer, Glm5MTPTransformerBlock.Config) for layer in layers]


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
        "w3_EFD": partial(nn.init.trunc_normal_, std=0.02),
    }


def _make_glm5_attention_config(
    base: DeepSeekMLAAttention.Config,
    *,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    mla_norm_eps: float,
    enable_sp: bool,
) -> Glm5Attention.Config:
    """Add GLM indexing to the MLA config constructed by the shared layer builder."""
    base.q_norm.eps = mla_norm_eps
    base.kv_norm.eps = mla_norm_eps
    mla = base.inner_attention
    mla.sharding_config = glm5_inner_attention_sharding_config()
    base.inner_attention = mla
    base.sharding_config = glm5_attention_sharding_config(enable_sp=enable_sp)
    return Glm5Attention.Config(
        **{field.name: getattr(base, field.name) for field in dataclasses.fields(base)},
        index_topk=index_topk,
        indexer=Glm5DsaIndexer.Config(
            n_heads=index_n_heads,
            head_dim=index_head_dim,
            qk_rope_head_dim=base.qk_rope_head_dim,
            wq_b=Linear.Config(
                in_features=base.q_lora_rank,
                out_features=index_n_heads * index_head_dim,
                param_init=_LINEAR_INIT,
                sharding_config=replicated_state_sharding_config("weight", "bias"),
            ),
            wk=Linear.Config(
                in_features=base.dim,
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
                in_features=base.dim,
                out_features=index_n_heads,
                param_init=_LINEAR_INIT,
                preserve_dtype_params=("weight",),
                sharding_config=replicated_state_sharding_config("weight", "bias"),
            ),
            rope=dataclasses.replace(
                base.rope,
                sharding_config=rope_cache_sharding_config(),
            ),
            topk=DSAIndexerTopK.Config(
                index_topk=index_topk,
                sharding_config=glm5_indexer_topk_sharding_config(),
            ),
            sharding_config=glm5_indexer_sharding_config(),
        ),
    )


def build_glm5_layers(
    *,
    num_layers: int,
    index_sources: tuple[int, ...],
    num_dense_layers: int,
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
    mla_norm_eps: float = 1e-6,
    norm_eps: float = 1e-5,
    rope: ComplexRoPE.Config,
    enable_sp: bool = False,
    enable_ep: bool = False,
) -> list[Glm5TransformerBlock.Config]:
    """Build GLM-5 dense and MoE block configurations for every layer."""
    if not 0 <= num_dense_layers <= num_layers:
        raise ValueError("num_dense_layers must be in [0, num_layers].")
    if len(index_sources) != num_layers:
        raise ValueError("index_sources must contain one source per layer")
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

    base_layers = build_mla_moe_layers(
        n_layers=num_layers,
        n_dense_layers=num_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        mscale=1.0,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=router_top_k,
        router_score_func=Sigmoid.Config(),
        router_route_scale=router_route_scale,
        router_route_norm=True,
        attn_backend="flex",
        attention_config_factory=lambda _: Glm5SparseAttention.Config(),
        linear_init=_LINEAR_INIT,
        norm_init=_RMS_NORM_INIT,
        depth_init=_depth_init,
        depth_experts_init=_depth_experts_init,
        rope=rope,
        router_config_factory=partial(
            make_deepseek_v3_router_config,
            num_expert_groups=router_num_expert_groups,
            num_limited_groups=router_num_limited_groups,
        ),
        norm_eps=norm_eps,
        enable_sp=enable_sp,
        enable_ep=enable_ep,
    )
    layers = []
    for layer_id, base in enumerate(base_layers):
        assert isinstance(base.attention, DeepSeekMLAAttention.Config)
        base.attention = _make_glm5_attention_config(
            base.attention,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            mla_norm_eps=mla_norm_eps,
            enable_sp=enable_sp,
        )
        if index_sources[layer_id] != layer_id:
            base.attention.indexer = None
        layers.append(
            Glm5TransformerBlock.Config(
                **{
                    field.name: getattr(base, field.name)
                    for field in dataclasses.fields(base)
                }
            )
        )
    return layers


def _debugmodel(
    *,
    num_dense_layers: int = 1,
    index_sources: tuple[int, ...] = (),
    seq_len: int = 128,
    parallelism: ParallelismConfig | None = None,
) -> Glm5Model.Config:
    dim = 256
    vocab_size = 2048
    rope = ComplexRoPE.Config(
        dim=32,
        max_context_length=seq_len,
        theta=8_000_000,
        scaling="none",
    )
    effective_index_sources = index_sources or tuple(range(8))
    return Glm5Model.Config(
        max_context_length=seq_len,
        vocab_size=vocab_size,
        dim=dim,
        index_sources=effective_index_sources,
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
            num_layers=8,
            index_sources=effective_index_sources,
            num_dense_layers=num_dense_layers,
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
            mla_norm_eps=1e-5,
            rope=rope,
            enable_sp=bool(parallelism and parallelism.enable_sequence_parallel),
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
        num_dense_layers=3,
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
    source_indexer = config.layers[config.index_sources[-1]].attention.indexer
    assert source_indexer is not None
    return dataclasses.replace(
        config,
        layers=layers,
        mtp_layers=build_glm5_mtp_layers(
            layers[-1],
            indexer=source_indexer,
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
        (
            layer_id
            if layer_id < 3 or (layer_id - 2) % 4 == 0
            else max(
                source
                for source in range(layer_id)
                if source < 3 or (source - 2) % 4 == 0
            )
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
            num_layers=num_layers,
            index_sources=index_sources,
            num_dense_layers=3,
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
            mla_norm_eps=1e-5,
            norm_eps=1e-5,
            rope=rope,
            enable_sp=bool(parallelism and parallelism.enable_sequence_parallel),
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
        (
            layer_id
            if max(layer_id - 2, 0) % 4 == 0
            else max(
                source for source in range(layer_id) if max(source - 2, 0) % 4 == 0
            )
        )
        for layer_id in range(num_layers)
    )
    rope = ComplexRoPE.Config(
        dim=64,
        max_context_length=seq_len,
        theta=8_000_000,
        scaling="none",
    )
    layers = build_glm5_layers(
        num_layers=num_layers,
        index_sources=index_sources,
        num_dense_layers=3,
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
        mla_norm_eps=1e-5,
        norm_eps=1e-5,
        rope=rope,
        enable_sp=bool(parallelism and parallelism.enable_sequence_parallel),
        enable_ep=bool(parallelism and parallelism.expert_parallel_degree > 1),
    )
    layers: list[Glm5TransformerBlock.Config] = [
        dataclasses.replace(layer, preserve_output_for_mtp=True) for layer in layers
    ]
    source_attention = layers[index_sources[-1]].attention
    assert isinstance(source_attention, Glm5Attention.Config)
    source_indexer = source_attention.indexer
    assert source_indexer is not None
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
            indexer=source_indexer,
            dim=dim,
            num_mtp_layers=1,
            norm_eps=1e-5,
            enable_sp=bool(parallelism and parallelism.enable_sequence_parallel),
        ),
    )


MODEL_FLAVORS = {
    "744B-A40B": _glm5_2_744b_a40b,
    "debugmodel": _debugmodel,
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
    indexer_stage: Literal["frozen", "warmup", "sparse"] = "frozen",
    indexer_loss_coeff: float = 1.0,
) -> Glm5Model.Config:
    """Build one GLM-5 flavor using TorchTitan's current model contract."""
    if indexer_stage not in ("frozen", "warmup", "sparse"):
        raise ValueError("GLM indexer_stage must be frozen, warmup or sparse")
    if indexer_loss_coeff < 0:
        raise ValueError("GLM indexer_loss_coeff must be non-negative")
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
    if parallelism.context_parallel_load_balancer is not None:
        raise ValueError(
            "GLM DSA currently requires contiguous context parallel shards"
        )
    config = get_config(seq_len=context_len, parallelism=parallelism)
    config.indexer_training_stage = indexer_stage
    for layer in config.layers + list(config.mtp_layers or ()):
        attention = layer.attention
        assert isinstance(attention, Glm5Attention.Config)
        leaf = attention.inner_attention.inner_attention
        assert isinstance(leaf, Glm5SparseAttention.Config)
        leaf.training_stage = indexer_stage
        if indexer_stage == "warmup" and isinstance(layer.feed_forward, MoE.Config):
            # Keep checkpointed routing bias, but freeze its optimizer-side update.
            layer.feed_forward.freeze_expert_bias = True
        if attention.indexer is not None:
            attention.indexer.training_stage = indexer_stage
            if indexer_stage != "frozen":
                leaf.indexer_loss = Glm5IndexerLoss.Config(
                    coeff=indexer_loss_coeff,
                    reduce_mesh="loss",
                    softmax_scale=attention.softmax_scale,
                    num_heads=attention.n_heads,
                )
    if parallelism.context_parallel_degree > 1:
        ContextParallelTransform(
            inner_attention_map={MLAAttention: Glm5CPMLAAttention}
        ).transform(config)
    if config.mtp_layers and parallelism.pipeline_parallel_degree > 1:
        raise ValueError("GLM MTP does not support pipeline parallelism")
    return config
