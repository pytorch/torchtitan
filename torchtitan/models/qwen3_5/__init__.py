# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_vlm
from torchtitan.models.common import Embedding, Linear
from torchtitan.models.qwen3_8 import (
    _build_qwen38_layers,
    _build_qwen38_moe_layers,
    _debugmodel,
    _debugmodel_moe,
    _EMBEDDING_INIT,
    _offset_norm,
    _output_linear_init,
    _qwen38_vision_encoder_config,
    _qwen3_8_27b,
)
from torchtitan.models.qwen3_8.model import Qwen35Model
from torchtitan.models.qwen3_8.parallelize import parallelize_qwen3_8
from torchtitan.models.qwen3_8.rope import MRoPE
from torchtitan.models.qwen3_8.state_dict_adapter import Qwen35StateDictAdapter
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

__all__ = [
    "QWEN3_5_SPECIAL_TOKENS",
    "Qwen35Model",
    "Qwen35StateDictAdapter",
    "model_registry",
    "parallelize_qwen3_5",
    "qwen3_5_configs",
]

# Qwen3.5 and Qwen3.8 share the implementation. Keep the dependency in this
# direction so removing Qwen3.5 later does not require moving model code again.
parallelize_qwen3_5 = parallelize_qwen3_8

QWEN3_5_SPECIAL_TOKENS = {
    "image_token": "<|image_pad|>",
    "video_token": "<|video_pad|>",
    "vision_start_token": "<|vision_start|>",
    "vision_end_token": "<|vision_end|>",
    "pad_token": "<|endoftext|>",
}


def _dense_config(
    attn_backend: str,
    *,
    dim: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    hidden_dim: int,
    num_key_heads: int,
    num_value_heads: int,
    vision_dim: int,
    vision_ffn_dim: int,
    num_vision_layers: int,
    num_vision_heads: int,
) -> Qwen35Model.Config:
    head_dim = 256
    rotary_dim = 64
    vocab_size = 248320
    return Qwen35Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        # pyrefly: ignore [bad-argument-type]
        norm=_offset_norm(dim),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_qwen38_layers(
            rope=MRoPE.Config(
                dim=rotary_dim,
                max_context_length=262144,
                theta=10_000_000.0,
                mrope_section=[11, 11, 10],
            ),
            attn_backend=attn_backend,
            n_layers=num_layers,
            dim=dim,
            n_heads=num_heads,
            n_kv_heads=num_kv_heads,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            hidden_dim=hidden_dim,
            n_key_heads=num_key_heads,
            n_value_heads=num_value_heads,
            key_head_dim=128,
            value_head_dim=128,
        ),
        vision_encoder=_qwen38_vision_encoder_config(
            dim=vision_dim,
            ffn_dim=vision_ffn_dim,
            num_layers=num_vision_layers,
            num_heads=num_vision_heads,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=dim,
            num_position_embeddings=2304,
        ),
    )


def _moe_config(
    attn_backend: str,
    moe_comm_backend: str = "standard",
    *,
    dim: int,
    num_layers: int,
    num_heads: int,
    moe_hidden_dim: int,
    num_experts: int,
    top_k: int,
    num_value_heads: int,
) -> Qwen35Model.Config:
    head_dim = 256
    rotary_dim = 64
    vocab_size = 248320
    return Qwen35Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        # pyrefly: ignore [bad-argument-type]
        norm=_offset_norm(dim),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_qwen38_moe_layers(
            rope=MRoPE.Config(
                dim=rotary_dim,
                max_context_length=262144,
                theta=10_000_000.0,
                mrope_section=[11, 11, 10],
            ),
            attn_backend=attn_backend,
            n_layers=num_layers,
            dim=dim,
            n_heads=num_heads,
            n_kv_heads=2,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            moe_hidden_dim=moe_hidden_dim,
            num_experts=num_experts,
            top_k=top_k,
            shared_expert_hidden_dim=moe_hidden_dim,
            n_key_heads=16,
            n_value_heads=num_value_heads,
            key_head_dim=128,
            value_head_dim=128,
            moe_comm_backend=moe_comm_backend,
        ),
        vision_encoder=_qwen38_vision_encoder_config(
            dim=1152,
            ffn_dim=4304,
            num_layers=27,
            num_heads=16,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=dim,
            num_position_embeddings=2304,
        ),
    )


qwen3_5_configs = {
    "debugmodel": _debugmodel,
    "debugmodel_moe": _debugmodel_moe,
    "0.8B": partial(
        _dense_config,
        dim=1024,
        num_layers=24,
        num_heads=8,
        num_kv_heads=2,
        hidden_dim=3584,
        num_key_heads=16,
        num_value_heads=16,
        vision_dim=768,
        vision_ffn_dim=3072,
        num_vision_layers=12,
        num_vision_heads=12,
    ),
    "2B": partial(
        _dense_config,
        dim=2048,
        num_layers=24,
        num_heads=8,
        num_kv_heads=2,
        hidden_dim=6144,
        num_key_heads=16,
        num_value_heads=16,
        vision_dim=1024,
        vision_ffn_dim=4096,
        num_vision_layers=24,
        num_vision_heads=16,
    ),
    "4B": partial(
        _dense_config,
        dim=2560,
        num_layers=32,
        num_heads=16,
        num_kv_heads=4,
        hidden_dim=9216,
        num_key_heads=16,
        num_value_heads=32,
        vision_dim=1024,
        vision_ffn_dim=4096,
        num_vision_layers=24,
        num_vision_heads=16,
    ),
    "9B": partial(
        _dense_config,
        dim=4096,
        num_layers=32,
        num_heads=16,
        num_kv_heads=4,
        hidden_dim=12288,
        num_key_heads=16,
        num_value_heads=32,
        vision_dim=1152,
        vision_ffn_dim=4304,
        num_vision_layers=27,
        num_vision_heads=16,
    ),
    "27B": _qwen3_8_27b,
    "35B-A3B": partial(
        _moe_config,
        dim=2048,
        num_layers=40,
        num_heads=16,
        moe_hidden_dim=512,
        num_experts=256,
        top_k=8,
        num_value_heads=32,
    ),
    "122B-A10B": partial(
        _moe_config,
        dim=3072,
        num_layers=48,
        num_heads=32,
        moe_hidden_dim=1024,
        num_experts=256,
        top_k=8,
        num_value_heads=64,
    ),
    "397B-A17B": partial(
        _moe_config,
        dim=4096,
        num_layers=60,
        num_heads=32,
        moe_hidden_dim=1024,
        num_experts=512,
        top_k=10,
        num_value_heads=64,
    ),
}


def model_registry(
    flavor: str,
    attn_backend: str = "flex",
    moe_comm_backend: str | None = None,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    kwargs = {"attn_backend": attn_backend}
    if moe_comm_backend is not None:
        kwargs["moe_comm_backend"] = moe_comm_backend
    config = qwen3_5_configs[flavor](**kwargs)
    if converters is not None:
        validate_converter_order(converters)
        for converter_config in converters:
            config = converter_config.build().convert(config)

    return ModelSpec(
        name="qwen3_5",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_qwen3_5,
        pipelining_fn=pipeline_vlm,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=Qwen35StateDictAdapter,
    )
