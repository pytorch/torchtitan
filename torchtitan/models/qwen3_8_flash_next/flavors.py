# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configurations for the released Flash-Next model and reduced feature fixture."""

import dataclasses
from functools import partial

import spmd_types as spmd

import torch.nn as nn

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import (
    ModelConfigConverter,
    validate_converter_compatibility,
)
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common import Conv1d, Embedding, GELU, Identity, Linear, Sigmoid
from torchtitan.models.common.attention.gdn import GDN
from torchtitan.models.common.decoder_sharding import (
    dense_param_placement,
    lm_head_sharding_config,
    pre_lm_head_norm_sharding_config,
    replicated_activation_placement,
    residual_activation_placement,
    rope_cache_sharding_config,
)
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.vision_encoder_sharding import (
    multimodal_token_embedding_sharding_config,
)
from torchtitan.models.qwen3_5.flavors import (
    _build_qwen35_moe_layers,
    _LINEAR_INIT,
    _OFFSET_NORM_INIT,
    _output_linear_init,
    _qwen35_vision_encoder_config,
    QWEN3_5_SPECIAL_TOKENS,
)
from torchtitan.models.qwen3_5.model import OffsetRMSNorm
from torchtitan.models.qwen3_5.rope import MRoPE

from .attention import QSAAttention, QSAIndexer, QSAIndexerLoss, QSAInnerAttention
from .embedding import ngram_vocab_sizes, NGramEmbedding, PLELayer
from .model import Qwen38FlashNextBlock, Qwen38FlashNextModel
from .moe import FlashNextRouter, RouterLinear
from .residual import GatedResidual, GroupedOffsetRMSNorm
from .sharding import (
    block_sharding,
    grouped_norm_sharding,
    indexer_sharding,
    ngram_embedding_sharding,
    ngram_table_sharding,
    ple_sharding,
    qsa_inner_sharding,
    replicated_linear_sharding,
    residual_mixer_sharding,
)

QWEN3_8_FLASH_NEXT_SPECIAL_TOKENS = QWEN3_5_SPECIAL_TOKENS
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=0.02)}


def _grouped_norm(dim, *, group_size, layout):
    return GroupedOffsetRMSNorm.Config(
        dim=dim,
        group_size=group_size,
        eps=1e-6,
        param_init=_OFFSET_NORM_INIT,
        sharding_config=grouped_norm_sharding(layout),
    )


def _gated_residual(*, dim, num_streams, rank, enable_sp, use_combine=True):
    layout = residual_activation_placement(enable_sp=enable_sp)
    channels = dim * num_streams

    def linear(in_features, out_features):
        return Linear.Config(
            in_features=in_features,
            out_features=out_features,
            param_init=_LINEAR_INIT,
            sharding_config=replicated_linear_sharding(layout),
        )

    return GatedResidual.Config(
        dim=dim,
        num_streams=num_streams,
        hc_norm=_grouped_norm(channels, group_size=dim, layout=layout),
        input_mix_weight_down=linear(channels, rank),
        input_mix_weight_up=linear(rank, channels),
        block_inject_weight=linear(channels, num_streams) if use_combine else None,
        sharding_config=residual_mixer_sharding(
            enable_sp=enable_sp, use_combine=use_combine
        ),
    )


def _ple_config(
    *,
    dim,
    num_streams,
    vocab_size,
    embed_dim,
    heads_per_ngram,
    vocab_base,
    split_parts,
    enable_sp,
):
    layout = replicated_activation_placement()
    sizes = ngram_vocab_sizes(vocab_base, heads_per_ngram * 2, ple_layer_index=0)
    table_rows = ((sum(sizes) + 127) // 128) * 128

    def linear(out_features):
        return Linear.Config(
            in_features=embed_dim,
            out_features=out_features,
            param_init=_LINEAR_INIT,
            sharding_config=replicated_linear_sharding(layout),
        )

    return PLELayer.Config(
        dim=dim,
        num_streams=num_streams,
        ple_embedding=NGramEmbedding.Config(
            ngram_size=3,
            heads_per_ngram=heads_per_ngram,
            vocab_size=vocab_size,
            head_vocab_sizes=sizes,
            eos_token_id=248044,
            split_ngram_parts=split_parts,
            ngram_embedding=Embedding.Config(
                num_embeddings=table_rows,
                embedding_dim=embed_dim // (heads_per_ngram * 2),
                vocab_parallel=False,
                param_init=_EMBEDDING_INIT,
                sharding_config=ngram_table_sharding(),
            ),
            sharding_config=ngram_embedding_sharding(),
        ),
        key_proj=linear(dim * num_streams),
        value_proj=linear(dim),
        norm_key=_grouped_norm(dim * num_streams, group_size=dim, layout=layout),
        norm_query=_grouped_norm(dim * num_streams, group_size=dim, layout=layout),
        norm_conv=_grouped_norm(dim * num_streams, group_size=dim, layout=layout),
        conv1d=Conv1d.Config(
            in_channels=dim * num_streams,
            out_channels=dim * num_streams,
            kernel_size=4,
            groups=dim * num_streams,
            bias=False,
            param_init={"weight": nn.init.zeros_},
            sharding_config=ShardingConfig(
                state_shardings={"weight": dense_param_placement(tp=spmd.R)}
            ),
        ),
        sharding_config=ple_sharding(enable_sp=enable_sp),
    )


def _config(*, debug, seq_len, parallelism):
    dim = 256 if debug else 2560
    head_dim = 64 if debug else 256
    rotary_dim = 16 if debug else 64
    index_head_dim = 64 if debug else 128
    num_streams = 4
    rank = 32 if debug else 320
    enable_sp = parallelism.enable_sequence_parallel
    vocab_size = 248320
    rope = MRoPE.Config(
        compute_in_float32=False,
        dim=rotary_dim,
        max_context_length=seq_len,
        theta=10_000_000,
        mrope_section=[3, 3, 2] if debug else [11, 11, 10],
    )
    base_layers = _build_qwen35_moe_layers(
        n_layers=8 if debug else 48,
        dim=dim,
        n_heads=4 if debug else 24,
        n_kv_heads=2,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        rope=rope,
        moe_hidden_dim=128 if debug else 640,
        num_experts=8 if debug else 512,
        top_k=2 if debug else 10,
        shared_expert_hidden_dim=128 if debug else 640,
        n_key_heads=2 if debug else 16,
        n_value_heads=4 if debug else 48,
        key_head_dim=128,
        value_head_dim=128,
        attn_backend="varlen",
        enable_sp=enable_sp,
        enable_ep=parallelism.expert_parallel_degree > 1,
        has_vision_encoder=True,
    )
    layers = []
    replicated = replicated_activation_placement()
    for layer_id, base_layer in enumerate(base_layers):
        assert isinstance(base_layer.feed_forward, MoE.Config)
        router = base_layer.feed_forward.router
        gate = router.gate
        router_fields = {
            item.name: getattr(router, item.name) for item in dataclasses.fields(router)
        }
        router_fields["gate"] = RouterLinear.Config(
            **{item.name: getattr(gate, item.name) for item in dataclasses.fields(gate)}
        )
        feed_forward = dataclasses.replace(
            base_layer.feed_forward, router=FlashNextRouter.Config(**router_fields)
        )
        attention = base_layer.attention
        if isinstance(attention, GDN.Config):
            attention = dataclasses.replace(
                attention,
                norm=dataclasses.replace(
                    attention.norm,
                    activation_fn=Sigmoid.Config(),
                    cast_before_gate=True,
                ),
            )
        else:
            fields = {
                item.name: getattr(attention, item.name)
                for item in dataclasses.fields(attention)
            }
            fields["inner_attention"] = QSAInnerAttention.Config(
                compress_ratio=4,
                num_heads=attention.n_heads,
                indexer_loss=QSAIndexerLoss.Config(
                    coeff=1.0,
                    reduce_mesh="loss",
                    sharding_config=ShardingConfig(
                        state_shardings={
                            "instance_acc": dense_param_placement(tp=spmd.R)
                        }
                    ),
                ),
                sharding_config=qsa_inner_sharding(),
            )
            attention = QSAAttention.Config(
                **fields,
                indexer=QSAIndexer.Config(
                    num_heads=4,
                    head_dim=index_head_dim,
                    rotary_dim=rotary_dim,
                    token_budget=16 if debug else 2048,
                    compress_ratio=4,
                    index_qk_proj=Linear.Config(
                        in_features=dim,
                        out_features=5 * index_head_dim,
                        param_init=_LINEAR_INIT,
                        sharding_config=replicated_linear_sharding(replicated),
                    ),
                    q_layernorm=OffsetRMSNorm.Config(
                        dim=index_head_dim,
                        eps=1e-6,
                        param_init=_OFFSET_NORM_INIT,
                        sharding_config=ShardingConfig(
                            state_shardings={"weight": dense_param_placement(tp=spmd.R)}
                        ),
                    ),
                    k_layernorm=OffsetRMSNorm.Config(
                        dim=index_head_dim,
                        eps=1e-6,
                        param_init=_OFFSET_NORM_INIT,
                        sharding_config=ShardingConfig(
                            state_shardings={"weight": dense_param_placement(tp=spmd.R)}
                        ),
                    ),
                    rope=dataclasses.replace(
                        rope, sharding_config=rope_cache_sharding_config()
                    ),
                    sharding_config=indexer_sharding(),
                ),
            )
        layers.append(
            Qwen38FlashNextBlock.Config(
                attention=attention,
                feed_forward=feed_forward,
                # Normalization belongs to the gated residuals.
                attention_norm=Identity.Config(),
                ffn_norm=Identity.Config(),
                attn_hyper_connection=_gated_residual(
                    dim=dim, num_streams=num_streams, rank=rank, enable_sp=enable_sp
                ),
                mlp_hyper_connection=_gated_residual(
                    dim=dim, num_streams=num_streams, rank=rank, enable_sp=enable_sp
                ),
                ple=_ple_config(
                    dim=dim,
                    num_streams=num_streams,
                    vocab_size=vocab_size,
                    embed_dim=dim,
                    heads_per_ngram=2 if debug else 8,
                    vocab_base=257 if debug else 20_000_000,
                    split_parts=4 if debug else 128,
                    enable_sp=enable_sp,
                )
                if layer_id == 1
                else None,
                sharding_config=block_sharding(
                    enable_sp=enable_sp, first_layer=layer_id == 0
                ),
            )
        )
    vision_encoder = _qwen35_vision_encoder_config(
        dim=128 if debug else 1152,
        ffn_dim=256 if debug else 4304,
        num_layers=2 if debug else 27,
        num_heads=4 if debug else 16,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=dim,
        num_position_embeddings=1024 if debug else 2304,
    )
    vision_encoder.merger.act_fn = GELU.Config(approximate="none")
    return Qwen38FlashNextModel.Config(
        dim=dim,
        vocab_size=vocab_size,
        max_context_length=seq_len,
        num_streams=num_streams,
        layers=layers,
        hyper_connection_mixer=_gated_residual(
            dim=dim,
            num_streams=num_streams,
            rank=rank,
            enable_sp=enable_sp,
            use_combine=False,
        ),
        # The final gated mixer contains the norm; there is no separate RMSNorm.
        # pyrefly: ignore [bad-argument-type]
        norm=Identity.Config(
            sharding_config=pre_lm_head_norm_sharding_config(enable_sp=enable_sp)
        ),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=multimodal_token_embedding_sharding_config(),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        vision_encoder=vision_encoder,
    )


MODEL_FLAVORS = {
    "debugmodel": (partial(_config, debug=True), 4096),
    "125B-A6B": (partial(_config, debug=False), 262144),
}


def build_model_config(
    flavor: str,
    *,
    parallelism: ParallelismConfig,
    seq_len: int | None = None,
    attn_backend: str = "qsa",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> Qwen38FlashNextModel.Config:
    if attn_backend != "qsa":
        raise ValueError("Flash-Next requires attn_backend='qsa'")
    if parallelism.context_parallel_degree > 1:
        raise ValueError(
            "Flash-Next QSA and PLE do not yet support context parallelism"
        )
    factory, max_context = MODEL_FLAVORS[flavor]
    context = max_context if seq_len is None else seq_len
    if not 0 < context <= max_context:
        raise ValueError(f"seq_len must be between 1 and {max_context} for {flavor}")
    config = factory(seq_len=context, parallelism=parallelism)
    if converters is not None:
        validate_converter_compatibility(converters)
        for converter in converters:
            config = converter.build().convert(config)
    return config
