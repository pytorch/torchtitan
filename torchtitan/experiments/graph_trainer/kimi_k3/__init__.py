# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.models.kimi_k3 import (
    _kimi_k3_config,
    _vision_encoder_config,
    kimi_k3_configs,
)
from torchtitan.models.kimi_k3.kda import KimiKDAKernel
from torchtitan.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter
from torchtitan.protocols.model_spec import ModelSpec

from .kda import GraphTrainerKDAKernel
from .model import GraphTrainerKimiK3Model
from .parallelize import parallelize_kimi_k3


def _to_graph_trainer_config(base) -> GraphTrainerKimiK3Model.Config:
    for layer in base.layers:
        if layer.delta_attention is None:
            continue
        kernel = layer.delta_attention.kernel
        if not isinstance(kernel, KimiKDAKernel.Config):
            raise ValueError("Kimi K3 delta attention requires KimiKDAKernel.Config")
        layer.delta_attention.kernel = GraphTrainerKDAKernel.Config(
            lower_bound=kernel.lower_bound
        )
    return GraphTrainerKimiK3Model.Config(
        **{field.name: getattr(base, field.name) for field in fields(base)}
    )


def _kimi_k3_16b(attn_backend: str) -> GraphTrainerKimiK3Model.Config:
    """Build a 15.4B Kimi K3 configuration for two-GPU graph training."""
    dim = 2048
    num_layers = 27
    base = _kimi_k3_config(
        dim=dim,
        vocab_size=163840,
        num_layers=num_layers,
        full_attention_layers=set(range(3, num_layers, 4)) | {num_layers - 1},
        attn_res_block_size=12,
        num_heads=32,
        q_lora_rank=512,
        kv_lora_rank=256,
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        v_head_dim=64,
        kda_head_dim=64,
        conv_kernel_size=4,
        dense_hidden_dim=9728,
        latent_dim=1024,
        expert_hidden_dim=896,
        num_experts=192,
        top_k=8,
        num_shared_experts=2,
        vision_encoder=_vision_encoder_config(
            text_dim=dim,
            dim=512,
            qkv_dim=768,
            hidden_dim=2048,
            num_layers=8,
            num_heads=6,
            init_pos_emb_height=32,
            init_pos_emb_width=32,
        ),
        attn_backend=attn_backend,
    )
    base.vision_encoder = None
    return _to_graph_trainer_config(base)


def _kimi_k3_15b_compute_bound(
    attn_backend: str,
) -> GraphTrainerKimiK3Model.Config:
    """Build a Kimi K3 configuration sized for BS=16 on two GB200 GPUs."""
    dim = 1536
    num_layers = 64
    base = _kimi_k3_config(
        dim=dim,
        vocab_size=131072,
        num_layers=num_layers,
        full_attention_layers=set(range(3, num_layers, 4)) | {num_layers - 1},
        attn_res_block_size=12,
        num_heads=24,
        q_lora_rank=384,
        kv_lora_rank=192,
        qk_nope_head_dim=64,
        qk_rope_head_dim=32,
        v_head_dim=64,
        kda_head_dim=64,
        conv_kernel_size=4,
        dense_hidden_dim=6144,
        latent_dim=512,
        expert_hidden_dim=768,
        num_experts=176,
        top_k=16,
        num_shared_experts=2,
        vision_encoder=_vision_encoder_config(
            text_dim=dim,
            dim=512,
            qkv_dim=768,
            hidden_dim=2048,
            num_layers=8,
            num_heads=6,
            init_pos_emb_height=32,
            init_pos_emb_width=32,
        ),
        attn_backend=attn_backend,
    )
    base.vision_encoder = None
    return _to_graph_trainer_config(base)


def model_registry(flavor: str, attn_backend: str = "flex") -> ModelSpec:
    if flavor in ("16B", "16B-text"):
        config = _kimi_k3_16b(attn_backend)
    elif flavor == "15B-compute-bound":
        config = _kimi_k3_15b_compute_bound(attn_backend)
    else:
        base = kimi_k3_configs[flavor](attn_backend=attn_backend)
        base.vision_encoder = None
        config = _to_graph_trainer_config(base)

    return ModelSpec(
        name="graph_trainer/kimi_k3",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_kimi_k3,
        pipelining_fn=None,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=KimiK3StateDictAdapter,
    )
