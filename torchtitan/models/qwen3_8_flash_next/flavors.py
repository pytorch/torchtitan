# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Released architecture and reduced text/multimodal feature configurations."""

from dataclasses import replace
from functools import partial

import spmd_types as spmd
import torch.nn as nn

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import (
    ModelConfigConverter,
    validate_converter_compatibility,
)
from torchtitan.models.base.sharding import ShardingConfig
from torchtitan.models.common import (
    Conv1d,
    Embedding,
    GELU,
    Identity,
    Linear,
    RowParallelLinear,
    Sigmoid,
)
from torchtitan.models.common.config_utils import (
    fused_gate_up_param_init,
    make_moe_config,
    make_routed_experts_config,
    make_shared_expert_down_proj_config,
)
from torchtitan.models.common.decoder_sharding import (
    dense_param_placement,
    head_norm_sharding_config,
    lm_head_sharding_config,
    pre_lm_head_norm_sharding_config,
    replicated_activation_placement,
    replicated_input_colwise_sharding_config,
    residual_activation_placement,
    rope_cache_sharding_config,
    rowwise_sharding_config,
)
from torchtitan.models.common.moe.sharding import (
    router_gate_sharding_config,
    router_sharding_config,
    shared_experts_sharding_config,
)
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.models.common.vision_encoder_sharding import (
    multimodal_token_embedding_sharding_config,
)
from torchtitan.models.qwen3_5.flavors import (
    make_qwen35_deltanet_config,
    make_qwen35_vision_encoder_config,
    QWEN3_5_SPECIAL_TOKENS,
)
from torchtitan.models.qwen3_5.model import OffsetRMSNorm
from torchtitan.models.qwen3_5.moe import SigmoidGatedFeedForward
from torchtitan.models.qwen3_5.sharding import (
    attention_sharding_config,
    gated_rms_norm_sharding_config,
    sigmoid_gate_sharding_config,
    sigmoid_gated_w13_sharding_config,
)

from .attention import QSAAttention, QSAIndexer, QSAIndexerLoss, QSAInnerAttention
from .embedding import ngram_vocab_sizes, NGramEmbedding, NGramTable, PLELayer
from .model import Qwen38FlashNextBlock, Qwen38FlashNextModel
from .moe import FlashNextRouter, Float32Softmax
from .norm import GroupedOffsetRMSNorm, InputDtypeGatedRMSNorm
from .residual import GatedResidual
from .rope import MRoPE
from .sharding import (
    block_sharding,
    grouped_norm_sharding,
    indexer_sharding,
    ngram_embedding_sharding,
    ngram_table_sharding,
    ple_sharding,
    qsa_inner_sharding,
    residual_mixer_sharding,
    unsharded_linear_sharding,
)

QWEN3_8_FLASH_NEXT_SPECIAL_TOKENS = QWEN3_5_SPECIAL_TOKENS
_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_OFFSET_NORM_INIT = {"weight": nn.init.zeros_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=0.02)}


def _depth_init(layer_id):
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


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
            sharding_config=unsharded_linear_sharding(layout),
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
    eos_token_id,
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
            sharding_config=unsharded_linear_sharding(layout),
        )

    return PLELayer.Config(
        dim=dim,
        num_streams=num_streams,
        ple_embedding=NGramEmbedding.Config(
            ngram_size=3,
            heads_per_ngram=heads_per_ngram,
            vocab_size=vocab_size,
            head_vocab_sizes=sizes,
            eos_token_id=eos_token_id,
            split_ngram_parts=split_parts,
            ngram_embedding=NGramTable.Config(
                num_embeddings=table_rows,
                embedding_dim=embed_dim // (heads_per_ngram * 2),
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


def _shared_experts_config(*, dim, hidden_dim, layer_id, enable_sp):
    return SigmoidGatedFeedForward.Config(
        w13=Linear.Config(
            in_features=dim,
            out_features=hidden_dim,
            num_linears=2,
            param_init=fused_gate_up_param_init(_LINEAR_INIT, _LINEAR_INIT),
            sharding_config=sigmoid_gated_w13_sharding_config(),
        ),
        w2=make_shared_expert_down_proj_config(
            in_features=hidden_dim,
            out_features=dim,
            param_init=_depth_init(layer_id),
            enable_sp=enable_sp,
        ),
        gate=Linear.Config(
            in_features=dim,
            out_features=1,
            param_init=_LINEAR_INIT,
            sharding_config=sigmoid_gate_sharding_config(),
        ),
        sharding_config=shared_experts_sharding_config(enable_sp=enable_sp),
    )


def _qsa_config(
    *,
    dim,
    num_heads,
    head_dim,
    rotary_dim,
    index_head_dim,
    rope,
    layer_id,
    debug,
    enable_sp,
    indexer_loss,
):
    replicated = replicated_activation_placement()

    def norm():
        return OffsetRMSNorm.Config(
            dim=head_dim,
            eps=1e-6,
            param_init=_OFFSET_NORM_INIT,
            sharding_config=head_norm_sharding_config(),
        )

    def norm_index():
        return OffsetRMSNorm.Config(
            dim=index_head_dim,
            eps=1e-6,
            param_init=_OFFSET_NORM_INIT,
            sharding_config=ShardingConfig(
                state_shardings={"weight": dense_param_placement(tp=spmd.R)}
            ),
        )

    loss = (
        None
        if indexer_loss is None
        else replace(
            indexer_loss,
            sharding_config=ShardingConfig(
                state_shardings={"instance_acc": dense_param_placement(tp=spmd.R)}
            ),
        )
    )
    return QSAAttention.Config(
        n_heads=num_heads,
        n_kv_heads=2,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        rope=replace(rope, sharding_config=rope_cache_sharding_config()),
        wq=Linear.Config(
            in_features=dim,
            out_features=num_heads * head_dim * 2,
            param_init=_LINEAR_INIT,
            sharding_config=replicated_input_colwise_sharding_config(),
        ),
        wk=Linear.Config(
            in_features=dim,
            out_features=2 * head_dim,
            param_init=_LINEAR_INIT,
            sharding_config=replicated_input_colwise_sharding_config(),
        ),
        wv=Linear.Config(
            in_features=dim,
            out_features=2 * head_dim,
            param_init=_LINEAR_INIT,
            sharding_config=replicated_input_colwise_sharding_config(),
        ),
        wo=RowParallelLinear.Config(
            in_features=num_heads * head_dim,
            out_features=dim,
            param_init=_depth_init(layer_id),
            sharding_config=rowwise_sharding_config(
                output_layout=residual_activation_placement(enable_sp=enable_sp)
            ),
        ),
        q_norm=norm(),
        k_norm=norm(),
        inner_attention=QSAInnerAttention.Config(
            compress_ratio=4,
            indexer_loss=loss,
            sharding_config=qsa_inner_sharding(),
        ),
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
                sharding_config=unsharded_linear_sharding(replicated),
            ),
            q_layernorm=norm_index(),
            k_layernorm=norm_index(),
            rope=replace(rope, sharding_config=rope_cache_sharding_config()),
            sharding_config=indexer_sharding(),
        ),
        sharding_config=attention_sharding_config(enable_sp=enable_sp),
    )


def _config(*, debug, multimodal, seq_len, parallelism, indexer_loss):
    dim = 256 if debug else 2560
    head_dim = 64 if debug else 256
    rotary_dim = 16 if debug else 64
    index_head_dim = 64 if debug else 128
    num_streams = 4
    rank = 32 if debug else 320
    vocab_size = 2048 if debug else 248320
    eos_token_id = 2003 if debug else 248044
    num_experts = 8 if debug else 512
    top_k = 2 if debug else 10
    hidden_dim = 128 if debug else 640
    enable_sp = parallelism.enable_sequence_parallel
    rope = MRoPE.Config(
        dim=rotary_dim,
        max_context_length=seq_len,
        theta=10_000_000,
        mrope_section=[3, 3, 2] if debug else [11, 11, 10],
    )
    layers = []
    for layer_id in range(8 if debug else 48):
        router = FlashNextRouter.Config(
            num_experts=num_experts,
            top_k=top_k,
            route_norm=True,
            score_func=Float32Softmax.Config(),
            gate=Linear.Config(
                in_features=dim,
                out_features=num_experts,
                param_init=_depth_init(layer_id),
                sharding_config=router_gate_sharding_config(),
            ),
            sharding_config=router_sharding_config(),
        )
        feed_forward = make_moe_config(
            num_experts=num_experts,
            router=router,
            routed_experts=make_routed_experts_config(
                dim=dim,
                hidden_dim=hidden_dim,
                num_experts=num_experts,
                top_k=top_k,
                param_init={
                    "w1_EFD": _LINEAR_INIT["weight"],
                    "w2_EDF": _depth_init(layer_id)["weight"],
                    "w3_EFD": _LINEAR_INIT["weight"],
                },
                enable_ep=parallelism.expert_parallel_degree > 1,
            ),
            shared_experts=_shared_experts_config(
                dim=dim, hidden_dim=hidden_dim, layer_id=layer_id, enable_sp=enable_sp
            ),
            aux_loss_coeff=1e-3,
            aux_loss_type="batch_wise",
            enable_sp=enable_sp,
        )
        if (layer_id + 1) % 4 == 0:
            attention = _qsa_config(
                dim=dim,
                num_heads=4 if debug else 24,
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                index_head_dim=index_head_dim,
                rope=rope,
                layer_id=layer_id,
                debug=debug,
                enable_sp=enable_sp,
                indexer_loss=indexer_loss,
            )
        else:
            attention = make_qwen35_deltanet_config(
                dim=dim,
                n_key_heads=2 if debug else 16,
                n_value_heads=4 if debug else 48,
                key_head_dim=128,
                value_head_dim=128,
                layer_id=layer_id,
                enable_sp=enable_sp,
            )
            # Flash-Next uses BF16 A_log and dt_bias in the released reference.
            attention.preserve_dtype_params = ()
            attention.norm = InputDtypeGatedRMSNorm.Config(
                dim=128,
                eps=1e-6,
                activation_fn=Sigmoid.Config(),
                param_init={"weight": nn.init.ones_},
                sharding_config=gated_rms_norm_sharding_config(),
            )
        layers.append(
            Qwen38FlashNextBlock.Config(
                attention=attention,
                feed_forward=feed_forward,
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
                    eos_token_id=eos_token_id,
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
    vision_encoder = None
    if multimodal:
        vision_encoder = make_qwen35_vision_encoder_config(
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
    output_std = dim**-0.5
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
        # The final gated mixer owns the RMSNorm and restores the TP-replicated LM-head input.
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
            param_init={
                "weight": partial(
                    nn.init.trunc_normal_,
                    std=output_std,
                    a=-3 * output_std,
                    b=3 * output_std,
                )
            },
            sharding_config=lm_head_sharding_config(),
        ),
        vision_encoder=vision_encoder,
    )


MODEL_FLAVORS = {
    "debugmodel": (partial(_config, debug=True, multimodal=False), 4096),
    "debugmodel_mm": (partial(_config, debug=True, multimodal=True), 4096),
    "125B-A6B": (partial(_config, debug=False, multimodal=True), 262144),
}


def build_model_config(
    flavor: str,
    *,
    parallelism: ParallelismConfig,
    seq_len: int | None = None,
    indexer_loss: QSAIndexerLoss.Config | None = None,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> Qwen38FlashNextModel.Config:
    """Build Flash-Next; choose an optional indexer-training loss explicitly.

    The checkpoint reference defines selection but supplies no KL coefficient.
    Training recipes select that objective independently of the architecture.
    """
    if parallelism.context_parallel_degree > 1:
        raise ValueError("Flash-Next QSA and PLE do not support context parallelism")
    if parallelism.fsdp_enable_cpu_offloading:
        raise ValueError(
            "Row-owned PLE requires device-resident parameters; CPU offload is unsupported"
        )
    if parallelism.fsdp_defer_gradient_reduction:
        raise ValueError(
            "Row-owned PLE does not support deferred FSDP gradient reduction"
        )
    factory, max_context = MODEL_FLAVORS[flavor]
    context = max_context if seq_len is None else seq_len
    if not 0 < context <= max_context:
        raise ValueError(f"seq_len must be between 1 and {max_context} for {flavor}")
    config = factory(
        seq_len=context, parallelism=parallelism, indexer_loss=indexer_loss
    )
    for layer in config.layers:
        attention = layer.attention
        if (
            isinstance(attention, QSAAttention.Config)
            and attention.n_kv_heads % parallelism.tensor_parallel_degree != 0
        ):
            raise ValueError(
                f"Flash-Next tensor parallel degree ({parallelism.tensor_parallel_degree}) "
                f"must divide QSA KV heads ({attention.n_kv_heads})"
            )
    if converters is not None:
        validate_converter_compatibility(converters)
        for converter in converters:
            config = converter.build().convert(config)
    return config
