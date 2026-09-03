# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_vlm
from torchtitan.models.common import Embedding, Linear
from torchtitan.models.qwen3_5 import (
    _27b,
    _build_qwen35_moe_layers,
    _debugmodel,
    _debugmodel_moe,
    _EMBEDDING_INIT,
    _offset_norm,
    _output_linear_init,
    parallelize_qwen3_5,
    Qwen35Model,
    QWEN3_5_SPECIAL_TOKENS,
)
from torchtitan.models.qwen3_5.rope import MRoPE
from torchtitan.models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter
from torchtitan.models.utils import validate_converter_order
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

__all__ = [
    "model_registry",
    "QWEN3_8_SPECIAL_TOKENS",
    "Qwen35Model",
    "Qwen35StateDictAdapter",
    "qwen3_8_configs",
]

QWEN3_8_SPECIAL_TOKENS = dict(QWEN3_5_SPECIAL_TOKENS)


def _qwen3_8_2_4t_a95b(
    attn_backend: str,
    moe_comm_backend: str = "standard",
    *,
    seq_len: int,
) -> Qwen35Model.Config:
    """Qwen3.8-2.4T-A95B text-only MoE config."""
    dim = 8192
    head_dim = 256
    rotary_dim = 64
    num_layers = 92
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
            moe_comm_backend=moe_comm_backend,
        ),
    )


qwen3_8_configs = {
    "debugmodel": (_debugmodel, 4096),
    "debugmodel_moe": (_debugmodel_moe, 4096),
    "27B": (_27b, 262144),
    "2.4T-A95B": (_qwen3_8_2_4t_a95b, 262144),
}


def model_registry(
    flavor: str,
    *,
    seq_len: int | None = None,
    attn_backend: str = "flex",
    moe_comm_backend: str | None = None,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    get_config, max_context_len = qwen3_8_configs[flavor]
    context_len = seq_len or max_context_len
    if context_len > max_context_len:
        raise ValueError(
            f"Requested seq_len {context_len} exceeds max context length "
            f"{max_context_len} for flavor {flavor}"
        )
    config = get_config(
        attn_backend=attn_backend,
        seq_len=context_len,
        **(
            {"moe_comm_backend": moe_comm_backend}
            if moe_comm_backend is not None
            else {}
        ),
    )
    if converters is not None:
        validate_converter_order(converters)
        for converter_config in converters:
            config = converter_config.build().convert(config)

    return ModelSpec(
        name="qwen3_8",
        flavor=flavor,
        model=config,
        max_context_length=context_len,
        parallelize_fn=parallelize_qwen3_5,
        pipelining_fn=pipeline_vlm,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=Qwen35StateDictAdapter,
    )
