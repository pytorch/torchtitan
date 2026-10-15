# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import torch.nn as nn
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common import ComplexRoPE, Embedding, Linear, RMSNorm, SqrtSoftplus
from torchtitan.models.common.config_utils import (
    make_ffn_config,
    make_moe_config,
    make_routed_experts_config,
    make_router_config,
)
from torchtitan.models.common.param_init import depth_scaled_std

from .attention import Attention, CompressedSparseInnerAttention2
from .batched_linear import BatchedLinear
from .compressor import Compressor
from .indexer import (
    FULL,
    REINDEX,
    REUSE,
    HierarchicalIndexer,
    Selector,
)
from .metadata_extension import MetadataExtension
from .mhc import HcPost, HcPre
from .model import DeepSeekV41Model, DeepSeekV41TransformerBlock

if TYPE_CHECKING:
    from torchtitan.models.common.moe import MoE


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}
_SINK_INIT = {"attn_sink": partial(nn.init.trunc_normal_, std=0.02)}
_HC_PARAM_INIT = {
    "hc_fn": partial(nn.init.trunc_normal_, std=0.02),
    "hc_base": partial(nn.init.trunc_normal_, std=0.02),
    "hc_scale": partial(nn.init.trunc_normal_, std=0.02),
}
CANDIDATE_BLOCK_SIZE = 8

V41_CANDIDATE_SOURCE_LAYER = 20
V41_FULL_COMPRESS_RATIOS = (0, 0) + (2,) * 18 + (1,) * 20
V41_KV_SOURCE_LAYERS = (2, 8, 14, 20)
V41_FULL_INDEX_SOURCE_LAYERS = (2, 8, 14, 20, 24, 28, 32, 36)

V41_STRIPPED_COMPRESS_RATIOS = (2, 1, 1)
V41_STRIPPED_KV_SOURCE_LAYERS = (0, 1)
V41_STRIPPED_INDEX_SOURCE_LAYERS = (0, 1, 2)
V41_STRIPPED_CANDIDATE_SOURCE_LAYER = 1


def _output_linear_init(dim: int) -> dict:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def _depth_experts_init(layer_id: int) -> dict:
    return {
        "w1_EFD": partial(nn.init.trunc_normal_, std=0.02),
        "w2_EDF": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "w3_EFD": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
    }


def _make_compressor_config(
    *,
    dim: int,
    head_dim: int,
    compress_ratio: int,
    norm_eps: float,
    is_source: bool,
    rope: ComplexRoPE.Config | None,
) -> Compressor.Config:
    return Compressor.Config(
        compress_ratio=compress_ratio,
        is_source=is_source,
        rope=dataclasses.replace(rope) if rope is not None else None,
        wkv=(
            Linear.Config(
                in_features=dim,
                out_features=head_dim,
                bias=False,
                param_init=_LINEAR_INIT,
            )
            if is_source
            else None
        ),
        wgate=(
            Linear.Config(
                in_features=dim,
                out_features=head_dim,
                bias=False,
                param_init=_LINEAR_INIT,
            )
            if is_source and compress_ratio > 1
            else None
        ),
        norm=(
            RMSNorm.Config(
                normalized_shape=head_dim,
                eps=norm_eps,
                param_init=_NORM_INIT,
            )
            if is_source
            else None
        ),
    )


def _make_indexer_config(
    *,
    dim: int,
    q_lora_rank: int,
    head_dim: int,
    index_head_dim: int,
    num_index_heads: int,
    index_topk: int,
    compress_ratio: int,
    norm_eps: float,
    is_source: bool,
    owns_k: bool,
    is_candidate_source: bool,
    uses_candidates: bool,
    candidate_topk_blocks: int,
    candidate_block_size: int,
    rope: ComplexRoPE.Config | None,
) -> HierarchicalIndexer.Config:
    owns_index_k = owns_k and is_source
    if not is_source:
        mode = REUSE
    elif owns_index_k:
        mode = FULL
    else:
        mode = REINDEX
    wants_pool = is_candidate_source or uses_candidates
    pool_capacity = candidate_topk_blocks if wants_pool else -1
    pool_block_size = candidate_block_size if wants_pool else -1
    return HierarchicalIndexer.Config(
        mode=mode,
        selector=Selector.Config(
            mode=mode,
            compress_ratio=compress_ratio,
            num_index_heads=num_index_heads,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            candidate_topk_blocks=pool_capacity,
            candidate_block_size=pool_block_size,
        ),
        num_index_heads=num_index_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        compress_ratio=compress_ratio,
        rope=dataclasses.replace(rope) if rope is not None else None,
        wq_b=(
            Linear.Config(
                in_features=q_lora_rank,
                out_features=num_index_heads * index_head_dim,
                bias=False,
                param_init=_LINEAR_INIT,
            )
            if is_source
            else None
        ),
        weights_proj=(
            Linear.Config(
                in_features=dim,
                out_features=num_index_heads,
                bias=False,
                param_init=_LINEAR_INIT,
            )
            if is_source
            else None
        ),
        wk=(
            Linear.Config(
                in_features=head_dim,
                out_features=index_head_dim,
                bias=False,
                param_init=_LINEAR_INIT,
            )
            if owns_index_k
            else None
        ),
        k_norm=(
            RMSNorm.Config(
                normalized_shape=index_head_dim,
                eps=norm_eps,
                param_init=_NORM_INIT,
            )
            if owns_index_k
            else None
        ),
    )


def _make_v41_attn_config(
    *,
    dim: int,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    q_lora_rank: int,
    o_lora_rank: int,
    n_groups: int,
    compress_ratio: int,
    window_size: int,
    norm_eps: float,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    rope: ComplexRoPE.Config,
    owns_compressor: bool,
    owns_indexer: bool,
    source_key: bool,
    external_key: bool,
    is_candidate_source: bool,
    uses_candidates: bool,
    candidate_topk_blocks: int,
    candidate_block_size: int,
    layer_id: int,
) -> Attention.Config:
    if source_key and external_key:
        raise ValueError("an indexer cannot own its source-key projection and consume an external key at the same time")
    if is_candidate_source and not owns_compressor:
        raise ValueError("the candidate-pool source must own the compressed KV it scores")

    hd = head_dim
    attention_rope = dataclasses.replace(rope, split=hd - rope_head_dim)
    indexer_rope = dataclasses.replace(rope, split=index_head_dim - rope_head_dim)
    per_group_in = (n_heads * hd) // n_groups
    per_group_out = n_groups * o_lora_rank
    softmax_scale = head_dim**-0.5
    compressor_cfg = _make_compressor_config(
        dim=dim,
        head_dim=hd,
        compress_ratio=compress_ratio,
        norm_eps=norm_eps,
        is_source=owns_compressor,
        rope=attention_rope if owns_compressor else None,
    )
    indexer_cfg = _make_indexer_config(
        dim=dim,
        q_lora_rank=q_lora_rank,
        head_dim=hd,
        index_head_dim=index_head_dim,
        num_index_heads=index_n_heads,
        index_topk=index_topk,
        compress_ratio=compress_ratio,
        norm_eps=norm_eps,
        is_source=owns_indexer,
        owns_k=source_key,
        is_candidate_source=is_candidate_source,
        uses_candidates=uses_candidates,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
        rope=indexer_rope if owns_indexer else None,
    )
    inner_attention_cfg = CompressedSparseInnerAttention2.Config(
        window_size=window_size,
        softmax_scale=softmax_scale,
        compress_ratio=compress_ratio,
    )
    return Attention.Config(
        n_heads=n_heads,
        head_dim=head_dim,
        q_lora_rank=q_lora_rank,
        compress_ratio=compress_ratio,
        inner_attention=inner_attention_cfg,
        rope=attention_rope,
        wq_a=Linear.Config(
            in_features=dim,
            out_features=q_lora_rank,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        q_norm=RMSNorm.Config(
            normalized_shape=q_lora_rank,
            eps=norm_eps,
            param_init=_NORM_INIT,
        ),
        wq_b=Linear.Config(
            in_features=q_lora_rank,
            out_features=n_heads * hd,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        wkv=Linear.Config(
            in_features=dim,
            out_features=hd,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        kv_norm=RMSNorm.Config(
            normalized_shape=hd,
            eps=norm_eps,
            param_init=_NORM_INIT,
        ),
        wo_a=BatchedLinear.Config(
            n_heads=n_groups,
            in_features=per_group_in,
            out_features=o_lora_rank,
            param_init=_LINEAR_INIT,
        ),
        wo_b=Linear.Config(
            in_features=per_group_out,
            out_features=dim,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        param_init=_SINK_INIT,
        compressor=compressor_cfg,
        indexer=indexer_cfg,
    )


def _make_v41_moe_config(
    *,
    layer_id: int,
    dim: int,
    moe_inter_dim: int,
    num_experts: int,
    num_shared_experts: int,
    top_k: int,
    route_scale: float,
    route_norm: bool,
    load_balance_coeff: float,
    enable_sp: bool,
    enable_ep: bool,
) -> "MoE.Config":
    router = make_router_config(
        dim=dim,
        num_experts=num_experts,
        gate_param_init=_depth_init(layer_id),
        score_func=SqrtSoftplus.Config(),
        top_k=top_k,
        route_norm=route_norm,
        route_scale=route_scale,
    )
    return make_moe_config(
        num_experts=num_experts,
        router=router,
        routed_experts=make_routed_experts_config(
            dim=dim,
            hidden_dim=moe_inter_dim,
            num_experts=num_experts,
            top_k=top_k,
            param_init=_depth_experts_init(layer_id),
            enable_ep=enable_ep,
        ),
        shared_experts=(
            make_ffn_config(
                dim=dim,
                hidden_dim=moe_inter_dim * num_shared_experts,
                w13_param_init=_LINEAR_INIT,
                w2_param_init=_depth_init(layer_id),
                enable_sp=enable_sp,
            )
            if num_shared_experts > 0
            else None
        ),
        load_balance_coeff=load_balance_coeff,
        enable_sp=enable_sp,
    )


def _make_v41_config(
    *,
    n_layers: int,
    vocab_size: int,
    window_size: int,
    norm_eps: float,
    hc_mult: int,
    num_shared_experts: int,
    top_k: int,
    route_scale: float,
    route_norm: bool,
    load_balance_coeff: float,
    sinkhorn_iters: int,
    hc_eps: float,
    rope_theta: float,
    compress_rope_theta: float,
    rope_factor: float,
    original_seq_len: int,
    compress_ratios: tuple[int, ...],
    kv_source_layers: tuple[int, ...],
    index_source_layers: tuple[int, ...],
    candidate_source_layer: int,
    num_experts: int,
    dim: int,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    q_lora_rank: int,
    o_lora_rank: int,
    n_groups: int,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    moe_inter_dim: int,
    candidate_topk_blocks: int,
    max_context_length: int,
    parallelism: ParallelismConfig,
) -> DeepSeekV41Model.Config:
    if len(compress_ratios) != n_layers:
        raise ValueError(f"compress_ratios must match n_layers ({n_layers}), got {len(compress_ratios)}")
    for name, sources in (("kv_source_layers", kv_source_layers), ("index_source_layers", index_source_layers)):
        outside = sorted({source for source in sources if source < 0 or source >= n_layers})
        if outside:
            raise ValueError(f"{name} outside the {n_layers}-layer crop: {outside}")
    for layer_id, ratio in enumerate(compress_ratios):
        kv_source = max((source for source in kv_source_layers if source <= layer_id), default=None)
        if ratio > 0 and kv_source is None:
            raise ValueError(f"layer {layer_id} has compress_ratio={ratio} but no KV source precedes it")
        if kv_source is not None and compress_ratios[kv_source] != ratio:
            raise ValueError(
                f"layer {layer_id} (compress_ratio={ratio}) consumes the compressed KV of layer "
                f"{kv_source} (compress_ratio={compress_ratios[kv_source]}); the ratios must match"
            )
        if layer_id in index_source_layers and layer_id not in kv_source_layers:
            index_source = max((source for source in index_source_layers if source < layer_id), default=None)
            if index_source is None:
                raise ValueError(f"re-indexing layer {layer_id} has no preceding index source")
            if compress_ratios[index_source] != ratio:
                raise ValueError(
                    f"layer {layer_id} (compress_ratio={ratio}) scores the shared index keys of layer "
                    f"{index_source} (compress_ratio={compress_ratios[index_source]}); the ratios must match"
                )
    if candidate_source_layer not in set(index_source_layers):
        raise ValueError(
            "the candidate-pool source must also be an index source: "
            f"layer {candidate_source_layer} is not in index_source_layers"
        )
    if candidate_topk_blocks <= 0:
        raise ValueError(f"the candidate pool capacity must be positive, got {candidate_topk_blocks}")

    source_key_indexer_layers = tuple(layer_id for layer_id in index_source_layers if layer_id in kv_source_layers)
    external_key_indexer_layers = tuple(
        layer_id for layer_id in index_source_layers if layer_id not in kv_source_layers
    )

    rope = ComplexRoPE.Config(
        dim=rope_head_dim,
        max_context_length=max_context_length,
        theta=rope_theta,
        scaling="none",
    )
    rope_compress = ComplexRoPE.Config(
        dim=rope_head_dim,
        max_context_length=max_context_length,
        theta=compress_rope_theta,
        scaling="yarn",
        rope_factor=rope_factor,
        beta_fast=32.0,
        beta_slow=1.0,
        original_seq_len=original_seq_len,
    )

    enable_sp = parallelism.enable_sequence_parallel
    enable_ep = parallelism.expert_parallel_degree > 1

    layers = []
    for layer_id in range(n_layers):
        cr = compress_ratios[layer_id]
        attn_cfg = _make_v41_attn_config(
            dim=dim,
            n_heads=n_heads,
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            q_lora_rank=q_lora_rank,
            o_lora_rank=o_lora_rank,
            n_groups=n_groups,
            compress_ratio=cr,
            window_size=window_size,
            norm_eps=norm_eps,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            rope=rope_compress if cr >= 1 else rope,
            owns_compressor=layer_id in kv_source_layers,
            owns_indexer=layer_id in index_source_layers,
            source_key=layer_id in source_key_indexer_layers,
            external_key=layer_id in external_key_indexer_layers,
            is_candidate_source=layer_id == candidate_source_layer,
            uses_candidates=(
                layer_id in index_source_layers
                and layer_id != candidate_source_layer
                and 0 <= candidate_source_layer < layer_id
            ),
            candidate_topk_blocks=candidate_topk_blocks,
            candidate_block_size=CANDIDATE_BLOCK_SIZE,
            layer_id=layer_id,
        )
        moe_cfg = _make_v41_moe_config(
            layer_id=layer_id,
            dim=dim,
            moe_inter_dim=moe_inter_dim,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            top_k=top_k,
            route_scale=route_scale,
            route_norm=route_norm,
            load_balance_coeff=load_balance_coeff,
            enable_sp=enable_sp,
            enable_ep=enable_ep,
        )
        layers.append(
            DeepSeekV41TransformerBlock.Config(
                attention=attn_cfg,
                attention_norm=RMSNorm.Config(
                    normalized_shape=dim,
                    eps=norm_eps,
                    param_init=_NORM_INIT,
                ),
                ffn_norm=RMSNorm.Config(
                    normalized_shape=dim,
                    eps=norm_eps,
                    param_init=_NORM_INIT,
                ),
                moe=moe_cfg,
                hc_attn_pre=HcPre.Config(
                    hc_mult=hc_mult,
                    dim=dim,
                    sinkhorn_iters=sinkhorn_iters,
                    hc_eps=hc_eps,
                    norm_eps=norm_eps,
                    param_init=_HC_PARAM_INIT,
                ),
                hc_ffn_pre=HcPre.Config(
                    hc_mult=hc_mult,
                    dim=dim,
                    sinkhorn_iters=sinkhorn_iters,
                    hc_eps=hc_eps,
                    norm_eps=norm_eps,
                    param_init=_HC_PARAM_INIT,
                ),
                hc_post=HcPost.Config(),
            )
        )

    return DeepSeekV41Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        max_context_length=max_context_length,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, eps=norm_eps, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=layers,
        hc_mult=hc_mult,
        compress_ratios=compress_ratios,
        metadata_extension=MetadataExtension.Config(
            window_size=window_size,
            num_heads=n_heads,
            head_dim=head_dim,
            index_topk=index_topk,
            compress_ratios=compress_ratios,
        ),
        n_layers=n_layers,
        kv_source_layers=kv_source_layers,
        index_source_layers=index_source_layers,
        candidate_source_layer=candidate_source_layer,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=CANDIDATE_BLOCK_SIZE,
    )


def _debugmodel(
    *,
    parallelism: ParallelismConfig,
    seq_len: int = 512,
    num_experts: int = 16,
) -> DeepSeekV41Model.Config:
    return _make_v41_config(
        dim=512,
        n_heads=8,
        head_dim=64,
        rope_head_dim=16,
        q_lora_rank=128,
        o_lora_rank=128,
        n_groups=4,
        index_n_heads=8,
        index_head_dim=32,
        index_topk=32,
        moe_inter_dim=256,
        candidate_topk_blocks=4,
        max_context_length=512,
        n_layers=40,
        vocab_size=129280,
        window_size=128,
        norm_eps=1e-20,
        hc_mult=4,
        num_shared_experts=1,
        top_k=6,
        route_scale=1.5,
        route_norm=True,
        load_balance_coeff=0.001,
        sinkhorn_iters=20,
        hc_eps=1e-06,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        rope_factor=16.0,
        original_seq_len=65536,
        compress_ratios=V41_FULL_COMPRESS_RATIOS,
        kv_source_layers=V41_KV_SOURCE_LAYERS,
        index_source_layers=V41_FULL_INDEX_SOURCE_LAYERS,
        candidate_source_layer=V41_CANDIDATE_SOURCE_LAYER,
        num_experts=num_experts,
        parallelism=parallelism,
    )


def _deepseek_v4_1_flash(
    *,
    parallelism: ParallelismConfig,
    seq_len: int | None = None,
    num_experts: int = 384,
) -> DeepSeekV41Model.Config:
    max_context_length = seq_len or 4096
    return _make_v41_config(
        dim=5120,
        n_heads=64,
        head_dim=512,
        rope_head_dim=64,
        q_lora_rank=1280,
        o_lora_rank=1024,
        n_groups=8,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=512,
        moe_inter_dim=2304,
        candidate_topk_blocks=2048,
        max_context_length=max_context_length,
        n_layers=40,
        vocab_size=129280,
        window_size=128,
        norm_eps=1e-20,
        hc_mult=4,
        num_shared_experts=1,
        top_k=6,
        route_scale=1.5,
        route_norm=True,
        load_balance_coeff=0.001,
        sinkhorn_iters=20,
        hc_eps=1e-06,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        rope_factor=16.0,
        original_seq_len=65536,
        compress_ratios=V41_FULL_COMPRESS_RATIOS,
        kv_source_layers=V41_KV_SOURCE_LAYERS,
        index_source_layers=V41_FULL_INDEX_SOURCE_LAYERS,
        candidate_source_layer=V41_CANDIDATE_SOURCE_LAYER,
        num_experts=num_experts,
        parallelism=parallelism,
    )


def _stripped(
    *,
    parallelism: ParallelismConfig,
    num_experts: int = 4,
) -> DeepSeekV41Model.Config:
    return _make_v41_config(
        dim=5120,
        n_heads=64,
        head_dim=512,
        rope_head_dim=64,
        q_lora_rank=1280,
        o_lora_rank=1024,
        n_groups=8,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=512,
        candidate_topk_blocks=2048,
        moe_inter_dim=2304,
        max_context_length=4096,
        n_layers=len(V41_STRIPPED_COMPRESS_RATIOS),
        vocab_size=129280,
        window_size=128,
        norm_eps=1e-20,
        hc_mult=4,
        num_shared_experts=1,
        top_k=2,
        route_scale=1.5,
        route_norm=True,
        load_balance_coeff=0.001,
        sinkhorn_iters=20,
        hc_eps=1e-06,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        rope_factor=16.0,
        original_seq_len=65536,
        compress_ratios=V41_STRIPPED_COMPRESS_RATIOS,
        kv_source_layers=V41_STRIPPED_KV_SOURCE_LAYERS,
        index_source_layers=V41_STRIPPED_INDEX_SOURCE_LAYERS,
        candidate_source_layer=V41_STRIPPED_CANDIDATE_SOURCE_LAYER,
        num_experts=num_experts,
        parallelism=parallelism,
    )


MODEL_FLAVORS = {
    "debugmodel": (_debugmodel, 512),
    "deepseek_v4_1_flash": (_deepseek_v4_1_flash, 4096),
    "deepseek_v4_1_stripped": (_stripped, 4096),
}


def build_model_config(
    flavor: str,
    *,
    parallelism: ParallelismConfig,
    seq_len: int | None = None,
    converters: list | None = None,
) -> DeepSeekV41Model.Config:
    if flavor not in MODEL_FLAVORS:
        raise ValueError(
            f"Unknown deepseek_v4_1 flavor: {flavor}. "
            f"Available: {list(MODEL_FLAVORS.keys())}"
        )
    get_config, max_context_len = MODEL_FLAVORS[flavor]
    context_len = seq_len or max_context_len
    if context_len > max_context_len:
        raise ValueError(
            f"Requested seq_len {context_len} exceeds max context length "
            f"{max_context_len} for flavor {flavor}"
        )
    config = get_config(
        parallelism=parallelism,
        seq_len=context_len,
    )
    from .sharding import set_deepseek_v4_1_sharding_config

    set_deepseek_v4_1_sharding_config(
        config,
        enable_sp=parallelism.enable_sequence_parallel,
        enable_ep=parallelism.expert_parallel_degree > 1,
    )
    if converters is not None:
        from torchtitan.config.transform import validate_converter_compatibility
        validate_converter_compatibility(converters)
        for converter_cfg in converters:
            config = converter_cfg.build().convert(config)
    return config


def _register_step_pre_hooks(optimizers, model_parts, parallel_dims) -> None:
    from torchtitan.components.optimizer import register_moe_load_balancing_hook

    register_moe_load_balancing_hook(optimizers, model_parts, parallel_dims)


__all__ = [
    "CANDIDATE_BLOCK_SIZE",
    "V41_CANDIDATE_SOURCE_LAYER",
    "V41_FULL_COMPRESS_RATIOS",
    "V41_FULL_INDEX_SOURCE_LAYERS",
    "V41_KV_SOURCE_LAYERS",
    "Attention",
    "DeepSeekV41Model",
    "MODEL_FLAVORS",
    "build_model_config",
]
