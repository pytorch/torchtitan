# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
import dataclasses
from functools import partial

import torch.nn as nn

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import (
    ComplexRoPE,
    Embedding,
    FeedForward,
    Linear,
    RMSNorm,
    RoPE,
)
from torchtitan.models.common.config_utils import (
    make_experts_config,
    make_ffn_config,
)
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

from .attention import (
    Attention,
    collect_dsa_indexer_loss_metrics,
    DSAIndexerAuxLoss,
    DSAFlexAttention,
)
from .model import DeepSeekV4Model, DeepSeekV4TransformerBlock
from .moe import DeepSeekV4MoE, DeepSeekV4Router
from .parallelize import parallelize_deepseek_v4
from .state_dict_adapter import DeepSeekV4StateDictAdapter

__all__ = [
    "parallelize_deepseek_v4",
    "DeepSeekV4Model",
    "deepseek_v4_configs",
    "model_registry",
]

_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}


def _output_linear_init(dim: int) -> dict[str, Callable]:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
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


def _make_compressor_config(
    *,
    dim: int,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    rotate: bool,
    norm_eps: float,
    coff: int,
    rope: RoPE.Config,
) -> "Compressor.Config":
    from .compressor import Compressor
    return Compressor.Config(
        dim=dim,
        rope=dataclasses.replace(rope),
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        rotate=rotate,
        norm_eps=norm_eps,
        wkv=Linear.Config(
            in_features=dim, out_features=coff * head_dim, bias=False,
            param_init=_LINEAR_INIT,
        ),
        wgate=Linear.Config(
            in_features=dim, out_features=coff * head_dim, bias=False,
            param_init=_LINEAR_INIT,
        ),
        norm=RMSNorm.Config(
            normalized_shape=head_dim, eps=norm_eps,
            param_init=_NORM_INIT,
        ),
        # ape holds a (compress_ratio, coff * head_dim) weight in a Linear.
        ape=Linear.Config(
            in_features=coff * head_dim, out_features=compress_ratio, bias=False,
            param_init=_LINEAR_INIT,
        ),
    )


def _make_indexer_config(
    *,
    dim: int,
    num_index_heads: int,
    index_head_dim: int,
    index_topk: int,
    rope_head_dim: int,
    q_lora_rank: int,
    compress_ratio: int,
    norm_eps: float,
    rope: RoPE.Config,
) -> "Indexer.Config":
    from .compressor import Compressor, Indexer
    coff = 2  # overlap always True for indexer
    return Indexer.Config(
        dim=dim,
        rope=dataclasses.replace(rope),
        num_index_heads=num_index_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        rope_head_dim=rope_head_dim,
        q_lora_rank=q_lora_rank,
        compress_ratio=compress_ratio,
        norm_eps=norm_eps,
        wq_b=Linear.Config(
            in_features=q_lora_rank,
            out_features=num_index_heads * index_head_dim,
            bias=False,
            param_init=_LINEAR_INIT,
        ),
        weights_proj=Linear.Config(
            in_features=dim, out_features=num_index_heads, bias=False,
            param_init=_LINEAR_INIT,
        ),
        compressor=_make_compressor_config(
            dim=dim,
            head_dim=index_head_dim,
            rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio,
            rotate=True,
            norm_eps=norm_eps,
            coff=coff,
            rope=rope,
        ),
    )


def _make_v4_attn_config(
    *,
    layer_id: int,
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
    n_layers: int,
    rope: RoPE.Config,
    enable_indexer_loss: bool = True,
) -> Attention.Config:
    hd = head_dim
    per_group_in = (n_heads * hd) // n_groups
    per_group_out = n_groups * o_lora_rank
    softmax_scale = head_dim**-0.5
    # Conditionally build compressor/indexer configs.
    compressor_cfg = None
    indexer_cfg = None
    indexer_aux_loss_cfg = None
    compressor_128_cfg = None
    from .compressor import Compressor, Indexer

    if compress_ratio == 4:
        coff = 2  # 1 + overlap (overlap=True when compress_ratio==4)
        compressor_cfg = _make_compressor_config(
            dim=dim, head_dim=hd, rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio, rotate=False,
            norm_eps=norm_eps, coff=coff, rope=rope,
        )
        indexer_cfg = _make_indexer_config(
            dim=dim, num_index_heads=index_n_heads,
            index_head_dim=index_head_dim, index_topk=index_topk,
            rope_head_dim=rope_head_dim, q_lora_rank=q_lora_rank,
            compress_ratio=compress_ratio, norm_eps=norm_eps, rope=rope,
        )
        if enable_indexer_loss:
            indexer_aux_loss_cfg = DSAIndexerAuxLoss.Config(
                num_heads=n_heads,
                softmax_scale=softmax_scale,
                window_size=window_size,
            )
    elif compress_ratio > 1:
        coff = 1  # no overlap
        compressor_128_cfg = _make_compressor_config(
            dim=dim, head_dim=hd, rope_head_dim=rope_head_dim,
            compress_ratio=compress_ratio, rotate=False,
            norm_eps=norm_eps, coff=coff, rope=rope,
        )
    inner_attention_cfg = DSAFlexAttention.Config(
        window_size=window_size,
        compress_ratio=compress_ratio,
        softmax_scale=softmax_scale,
        return_lse=indexer_aux_loss_cfg is not None,
    )

    return Attention.Config(
        dim=dim,
        n_heads=n_heads,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        q_lora_rank=q_lora_rank,
        o_lora_rank=o_lora_rank,
        n_groups=n_groups,
        compress_ratio=compress_ratio,
        window_size=window_size,
        norm_eps=norm_eps,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        n_layers=n_layers,
        layer_id=layer_id,
        inner_attention=inner_attention_cfg,
        rope=dataclasses.replace(rope),
        wq_a=Linear.Config(
            in_features=dim, out_features=q_lora_rank, bias=False,
            param_init=_LINEAR_INIT,
        ),
        q_norm=RMSNorm.Config(
            normalized_shape=q_lora_rank, eps=norm_eps,
            param_init=_NORM_INIT,
        ),
        wq_b=Linear.Config(
            in_features=q_lora_rank, out_features=n_heads * hd, bias=False,
            param_init=_LINEAR_INIT,
        ),
        wkv=Linear.Config(
            in_features=dim, out_features=hd, bias=False,
            param_init=_LINEAR_INIT,
        ),
        kv_norm=RMSNorm.Config(
            normalized_shape=hd, eps=norm_eps,
            param_init=_NORM_INIT,
        ),
        wo_a=Linear.Config(
            in_features=per_group_in, out_features=per_group_out, bias=False,
            param_init=_LINEAR_INIT,
        ),
        wo_b=Linear.Config(
            in_features=per_group_out, out_features=dim, bias=False,
            param_init=_LINEAR_INIT,
        ),
        # attn_sink uses a Linear wrapper to hold a (n_heads, 1) weight; the
        # forward path squeezes it back to (n_heads,) to match the original
        # parameter semantics.
        attn_sink=Linear.Config(
            in_features=1, out_features=n_heads, bias=False,
            param_init=_LINEAR_INIT,
        ),
        compressor=compressor_cfg,
        compressor_128=compressor_128_cfg,
        indexer=indexer_cfg,
        indexer_aux_loss=indexer_aux_loss_cfg,
    )


def _make_v4_moe_config(
    *,
    layer_id: int,
    dim: int,
    moe_inter_dim: int,
    num_experts: int,
    num_shared_experts: int,
    top_k: int,
    vocab_size: int,
    n_hash_layers: int,
    route_norm: bool,
    route_scale: float,
    load_balance_coeff: float,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None,
):
    return DeepSeekV4MoE.Config(
        num_experts=num_experts,
        router=DeepSeekV4Router.Config(
            num_experts=num_experts,
            gate=Linear.Config(
                in_features=dim,
                out_features=num_experts,
                bias=False,
                param_init=_depth_init(layer_id),
            ),
            top_k=top_k,
            score_func="sqrtsoftplus",
            route_scale=route_scale,
            route_norm=route_norm,
            vocab_size=vocab_size,
            n_hash_layers=n_hash_layers,
            layer_id=layer_id,
        ),
        experts=make_experts_config(
            dim=dim,
            hidden_dim=moe_inter_dim,
            num_experts=num_experts,
            top_k=top_k,
            param_init=_depth_experts_init(layer_id),
            comm_backend=moe_comm_backend,
            non_blocking_capacity_factor=non_blocking_capacity_factor,
        ),
        shared_experts=(
            make_ffn_config(
                dim=dim,
                hidden_dim=moe_inter_dim * num_shared_experts,
                w1_param_init=_LINEAR_INIT,
                w2w3_param_init=_depth_init(layer_id),
            )
            if num_shared_experts > 0
            else None
        ),
        load_balance_coeff=load_balance_coeff,
    )


def _make_v4_dense_config(
    *,
    layer_id: int,
    dim: int,
    hidden_dim: int,
) -> FeedForward.Config:
    return make_ffn_config(
        dim=dim,
        hidden_dim=hidden_dim,
        w1_param_init=_LINEAR_INIT,
        w2w3_param_init=_depth_init(layer_id),
    )


def _build_v4_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    q_lora_rank: int,
    o_lora_rank: int,
    n_groups: int,
    compress_ratios: tuple[int, ...],
    window_size: int,
    norm_eps: float,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    moe_inter_dim: int,
    num_experts: int,
    num_shared_experts: int,
    top_k: int,
    vocab_size: int,
    n_hash_layers: int,
    route_norm: bool,
    route_scale: float,
    load_balance_coeff: float,
    moe_comm_backend: str,
    non_blocking_capacity_factor: float | None,
    rope: RoPE.Config,
    rope_compress: RoPE.Config,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    hc_eps: float = 1e-6,
    dense_hidden_dim: int | None = None,
    dense_layers: set[int] | None = None,
    enable_indexer_loss: bool = True,
) -> list[DeepSeekV4TransformerBlock.Config]:
    if dense_layers is None:
        dense_layers = set()
    if dense_hidden_dim is None:
        dense_hidden_dim = moe_inter_dim * 4

    layers = []
    for layer_id in range(n_layers):
        cr = compress_ratios[layer_id] if layer_id < len(compress_ratios) else 1

        attn_cfg = _make_v4_attn_config(
            layer_id=layer_id,
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
            n_layers=n_layers,
            rope=rope_compress if cr > 1 else rope,
            enable_indexer_loss=enable_indexer_loss,
        )

        if layer_id in dense_layers:
            ffn_cfg = _make_v4_dense_config(
                layer_id=layer_id,
                dim=dim,
                hidden_dim=dense_hidden_dim,
            )
            moe_cfg = None
        else:
            ffn_cfg = None
            moe_cfg = _make_v4_moe_config(
                layer_id=layer_id,
                dim=dim,
                moe_inter_dim=moe_inter_dim,
                num_experts=num_experts,
                num_shared_experts=num_shared_experts,
                top_k=top_k,
                vocab_size=vocab_size,
                n_hash_layers=n_hash_layers,
                route_norm=route_norm,
                route_scale=route_scale,
                load_balance_coeff=load_balance_coeff,
                moe_comm_backend=moe_comm_backend,
                non_blocking_capacity_factor=non_blocking_capacity_factor,
            )

        layers.append(
            DeepSeekV4TransformerBlock.Config(
                attention=attn_cfg,
                attention_norm=RMSNorm.Config(
                    normalized_shape=dim,
                    param_init=_NORM_INIT,
                ),
                ffn_norm=RMSNorm.Config(
                    normalized_shape=dim,
                    param_init=_NORM_INIT,
                ),
                feed_forward=ffn_cfg,
                moe=moe_cfg,
                hc_mult=hc_mult,
                dim=dim,
                norm_eps=norm_eps,
                sinkhorn_iters=sinkhorn_iters,
                hc_eps=hc_eps,
            )
        )
    return layers


def _debugmodel(
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
    enable_indexer_loss: bool = True,
) -> DeepSeekV4Model.Config:
    dim = 256
    n_layers = 4
    vocab_size = 2048
    n_heads = 16
    head_dim = 256
    rope_head_dim = 32
    q_lora_rank = 128
    o_lora_rank = 128
    n_groups = 2
    compress_ratios = (4, 1, 1, 4)
    window_size = 16
    norm_eps = 1e-6
    index_n_heads = 8
    index_head_dim = 64
    index_topk = 16
    moe_inter_dim = 256
    num_experts = 4
    num_shared_experts = 1
    top_k = 3
    n_hash_layers = 2
    route_norm = False
    route_scale = 1.5
    load_balance_coeff = 1e-3
    hc_mult = 4
    sinkhorn_iters = 20
    hc_eps = 1e-6
    dense_layers = set()
    max_seq_len = 4096 * 4
    compress_rope_theta = 40000.0
    original_seq_len = 65536

    rope = ComplexRoPE.Config(
        dim=rope_head_dim,
        max_seq_len=max_seq_len,
        theta=10000.0,
        scaling="none",
    )
    rope_compress = ComplexRoPE.Config(
        dim=rope_head_dim,
        max_seq_len=max_seq_len,
        theta=compress_rope_theta,
        scaling="yarn",
        rope_factor=4.0,
        beta_fast=32.0,
        beta_slow=1.0,
        original_seq_len=original_seq_len,
    )

    layers = _build_v4_layers(
        n_layers=n_layers,
        dim=dim,
        n_heads=n_heads,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        q_lora_rank=q_lora_rank,
        o_lora_rank=o_lora_rank,
        n_groups=n_groups,
        compress_ratios=compress_ratios,
        window_size=window_size,
        norm_eps=norm_eps,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        moe_inter_dim=moe_inter_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        top_k=top_k,
        vocab_size=vocab_size,
        n_hash_layers=n_hash_layers,
        route_norm=route_norm,
        route_scale=route_scale,
        load_balance_coeff=load_balance_coeff,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        rope=rope,
        rope_compress=rope_compress,
        hc_mult=hc_mult,
        sinkhorn_iters=sinkhorn_iters,
        hc_eps=hc_eps,
        dense_layers=dense_layers,
        enable_indexer_loss=enable_indexer_loss,
    )

    return DeepSeekV4Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        norm_eps=norm_eps,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=layers,
        hc_mult=hc_mult,
        compress_ratios=compress_ratios,
        n_layers=n_layers,
    )


def _deepseek_v4_flash(
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
    enable_indexer_loss: bool = True,
) -> DeepSeekV4Model.Config:
    dim = 4096
    n_layers = 43
    vocab_size = 129280
    n_heads = 64
    head_dim = 512
    rope_head_dim = 64
    q_lora_rank = 1024
    o_lora_rank = 1024
    n_groups = 8
    compress_ratios = (1, 1) + (4, 128) * 20 + (4,)
    window_size = 128
    norm_eps = 1e-6
    index_n_heads = 64
    index_head_dim = 128
    index_topk = 512
    moe_inter_dim = 2048
    num_experts = 256
    num_shared_experts = 1
    top_k = 6
    n_hash_layers = 3
    route_norm = True
    route_scale = 1.5
    load_balance_coeff = 1e-3
    hc_mult = 4
    sinkhorn_iters = 20
    hc_eps = 1e-6
    dense_layers = set()
    max_seq_len = 4096
    compress_rope_theta = 160000.0
    original_seq_len = 65536

    rope = ComplexRoPE.Config(
        dim=rope_head_dim,
        max_seq_len=max_seq_len,
        theta=10000.0,
        scaling="none",
    )
    rope_compress = ComplexRoPE.Config(
        dim=rope_head_dim,
        max_seq_len=max_seq_len,
        theta=compress_rope_theta,
        scaling="yarn",
        rope_factor=16.0,
        beta_fast=32.0,
        beta_slow=1.0,
        original_seq_len=original_seq_len,
    )

    layers = _build_v4_layers(
        n_layers=n_layers,
        dim=dim,
        n_heads=n_heads,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        q_lora_rank=q_lora_rank,
        o_lora_rank=o_lora_rank,
        n_groups=n_groups,
        compress_ratios=compress_ratios,
        window_size=window_size,
        norm_eps=norm_eps,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        moe_inter_dim=moe_inter_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        top_k=top_k,
        vocab_size=vocab_size,
        n_hash_layers=n_hash_layers,
        route_norm=route_norm,
        route_scale=route_scale,
        load_balance_coeff=load_balance_coeff,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        rope=rope,
        rope_compress=rope_compress,
        hc_mult=hc_mult,
        sinkhorn_iters=sinkhorn_iters,
        hc_eps=hc_eps,
        dense_layers=dense_layers,
        enable_indexer_loss=enable_indexer_loss,
    )

    return DeepSeekV4Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        norm_eps=norm_eps,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=layers,
        hc_mult=hc_mult,
        compress_ratios=compress_ratios,
        n_layers=n_layers,
    )


def _deepseek_v4_pro(
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
    enable_indexer_loss: bool = True,
) -> DeepSeekV4Model.Config:
    dim = 7168
    n_layers = 61
    vocab_size = 129280
    n_heads = 128
    head_dim = 512
    rope_head_dim = 64
    q_lora_rank = 1536
    o_lora_rank = 1024
    n_groups = 16
    compress_ratios = (128,) + (128, 4) * 30
    window_size = 128
    norm_eps = 1e-6
    index_n_heads = 64
    index_head_dim = 128
    index_topk = 1024
    moe_inter_dim = 3072
    num_experts = 384
    num_shared_experts = 1
    top_k = 6
    n_hash_layers = 3
    route_norm = True
    route_scale = 1.5
    load_balance_coeff = 1e-3
    hc_mult = 4
    sinkhorn_iters = 20
    hc_eps = 1e-6
    dense_layers = set()
    max_seq_len = 4096
    compress_rope_theta = 160000.0
    original_seq_len = 65536

    rope = ComplexRoPE.Config(
        dim=rope_head_dim,
        max_seq_len=max_seq_len,
        theta=10000.0,
        scaling="none",
    )
    rope_compress = ComplexRoPE.Config(
        dim=rope_head_dim,
        max_seq_len=max_seq_len,
        theta=compress_rope_theta,
        scaling="yarn",
        rope_factor=16.0,
        beta_fast=32.0,
        beta_slow=1.0,
        original_seq_len=original_seq_len,
    )

    layers = _build_v4_layers(
        n_layers=n_layers,
        dim=dim,
        n_heads=n_heads,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        q_lora_rank=q_lora_rank,
        o_lora_rank=o_lora_rank,
        n_groups=n_groups,
        compress_ratios=compress_ratios,
        window_size=window_size,
        norm_eps=norm_eps,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        moe_inter_dim=moe_inter_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        top_k=top_k,
        vocab_size=vocab_size,
        n_hash_layers=n_hash_layers,
        route_norm=route_norm,
        route_scale=route_scale,
        load_balance_coeff=load_balance_coeff,
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        rope=rope,
        rope_compress=rope_compress,
        hc_mult=hc_mult,
        sinkhorn_iters=sinkhorn_iters,
        hc_eps=hc_eps,
        dense_layers=dense_layers,
        enable_indexer_loss=enable_indexer_loss,
    )

    return DeepSeekV4Model.Config(
        dim=dim,
        vocab_size=vocab_size,
        norm_eps=norm_eps,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=layers,
        hc_mult=hc_mult,
        compress_ratios=compress_ratios,
        n_layers=n_layers,
    )


deepseek_v4_configs = {
    "debugmodel": _debugmodel,
    "deepseek_v4_flash": _deepseek_v4_flash,
    "deepseek_v4_pro": _deepseek_v4_pro,
}


def model_registry(
    flavor: str,
    moe_comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
    enable_indexer_loss: bool = True,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    if flavor not in deepseek_v4_configs:
        raise ValueError(
            f"Unknown deepseek_v4 flavor: {flavor}. "
            f"Available: {list(deepseek_v4_configs.keys())}"
        )
    config = deepseek_v4_configs[flavor](
        moe_comm_backend=moe_comm_backend,
        non_blocking_capacity_factor=non_blocking_capacity_factor,
        enable_indexer_loss=enable_indexer_loss,
    )
    if converters is not None:
        validate_converter_order(converters)
        for converter_cfg in converters:
            config = converter_cfg.build().convert(config)
    return ModelSpec(
        name="deepseek_v4",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_deepseek_v4,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        metrics_fn=collect_dsa_indexer_loss_metrics,
        state_dict_adapter=DeepSeekV4StateDictAdapter,
    )
