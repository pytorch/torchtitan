# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.8 model flavors."""

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import (
    ModelConfigConverter,
    validate_converter_compatibility,
)
from torchtitan.models.common import Embedding, Linear
from torchtitan.models.common.decoder_sharding import (
    lm_head_sharding_config,
    pre_lm_head_norm_sharding_config,
    token_embedding_sharding_config,
)
from torchtitan.models.qwen3_5.flavors import (
    _27b,
    _build_qwen35_moe_layers,
    _debugmodel,
    _debugmodel_moe,
    _EMBEDDING_INIT,
    _offset_norm,
    _output_linear_init,
    Qwen35Model,
    QWEN3_5_SPECIAL_TOKENS,
)
from torchtitan.models.qwen3_5.rope import MRoPE

__all__ = [
    "build_model_config",
    "QWEN3_8_SPECIAL_TOKENS",
    "Qwen35Model",
    "MODEL_FLAVORS",
]

QWEN3_8_SPECIAL_TOKENS = dict(QWEN3_5_SPECIAL_TOKENS)


def _qwen3_8_2_4t_a95b(
    attn_backend: str,
    *,
    seq_len: int,
    parallelism: ParallelismConfig,
) -> Qwen35Model.Config:
    """Qwen3.8-2.4T-A95B text-only MoE config."""
    dim = 8192
    head_dim = 256
    rotary_dim = 64
    num_layers = 92
    vocab_size = 248320
    return Qwen35Model.Config(
        max_context_length=seq_len,
        vocab_size=vocab_size,
        dim=dim,
        # pyrefly: ignore [bad-argument-type]
        norm=_offset_norm(
            dim,
            sharding_config=pre_lm_head_norm_sharding_config(
                enable_sp=parallelism.enable_sequence_parallel
            ),
        ),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
            sharding_config=token_embedding_sharding_config(
                enable_sp=parallelism.enable_sequence_parallel
            ),
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
            sharding_config=lm_head_sharding_config(),
        ),
        layers=_build_qwen35_moe_layers(
            rope=MRoPE.Config(
                dim=rotary_dim,
                max_context_length=seq_len,
                theta=10_000_000.0,
                mrope_section=[11, 11, 10],
            ),
            attn_backend=attn_backend,
            n_layers=num_layers,
            dim=dim,
            n_heads=64,
            n_kv_heads=4,
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            moe_hidden_dim=2048,
            num_experts=512,
            top_k=10,
            shared_expert_hidden_dim=2048,
            n_key_heads=16,
            n_value_heads=128,
            key_head_dim=128,
            value_head_dim=128,
            enable_sp=parallelism.enable_sequence_parallel,
            enable_ep=parallelism.expert_parallel_degree > 1,
            has_vision_encoder=False,
        ),
    )


MODEL_FLAVORS = {
    "debugmodel": (_debugmodel, 4096),
    "debugmodel_moe": (_debugmodel_moe, 4096),
    "27B": (_27b, 262144),
    "2.4T-A95B": (_qwen3_8_2_4t_a95b, 262144),
}


def build_model_config(
    flavor: str,
    *,
    parallelism: ParallelismConfig,
    seq_len: int | None = None,
    attn_backend: str = "varlen",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> Qwen35Model.Config:
    get_config, max_context_len = MODEL_FLAVORS[flavor]
    context_len = seq_len or max_context_len
    if context_len > max_context_len:
        raise ValueError(
            f"Requested seq_len {context_len} exceeds max context length "
            f"{max_context_len} for flavor {flavor}"
        )
    config = get_config(
        attn_backend=attn_backend,
        seq_len=context_len,
        parallelism=parallelism,
    )
    if converters is not None:
        validate_converter_compatibility(converters)
        for converter_config in converters:
            config = converter_config.build().convert(config)
    return config
