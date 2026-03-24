# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields
from functools import partial

import torch.nn as nn

from torchtitan.components.loss import build_cross_entropy_loss

from torchtitan.models.llama3 import llama3_configs, setup_llama3_param_init
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.protocols.module import set_param_init

from .datasets.mm_datasets import HuggingFaceMultiModalDataLoader
from .infra.parallelize import parallelize_vlm
from .model.args import Siglip2Config
from .model.model import Llama3Siglip2Transformer

__all__ = [
    "HuggingFaceMultiModalDataLoader",
    "parallelize_vlm",
    "Llama3Siglip2Transformer",
    "llama3_siglip2_configs",
]


def _get_dict(obj) -> dict:
    """Convert dataclass to dict, preserving nested dataclasses (unlike asdict)."""
    return {field.name: getattr(obj, field.name) for field in fields(obj)}


def setup_vlm_param_init(model: Llama3Siglip2Transformer) -> None:
    # Decoder portion (tok_embeddings, layers, norm, output)
    setup_llama3_param_init(model)
    # Projector: xavier
    set_param_init(
        model.projector.w1,
        {"weight": nn.init.xavier_uniform_, "bias": nn.init.zeros_},
    )
    set_param_init(
        model.projector.w2,
        {"weight": nn.init.xavier_uniform_, "bias": nn.init.zeros_},
    )
    # Encoder (Siglip2) — Linear params; LayerNorm uses reset_parameters via from_nn_module
    default = partial(nn.init.trunc_normal_, std=0.02)
    set_param_init(
        model.encoder.embeddings.patch_embedding,
        {"weight": default, "bias": nn.init.zeros_},
    )
    set_param_init(
        model.encoder.embeddings.position_embedding,
        {"weight": default},
    )
    for layer in model.encoder.layers.values():
        for proj in (
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.self_attn.out_proj,
        ):
            set_param_init(proj, {"weight": default, "bias": nn.init.zeros_})
        set_param_init(layer.mlp.fc1, {"weight": default, "bias": nn.init.zeros_})
        set_param_init(layer.mlp.fc2, {"weight": default, "bias": nn.init.zeros_})


llama3_siglip2_configs = {
    "debugmodel": Llama3Siglip2Transformer.Config(
        **{
            **_get_dict(llama3_configs["debugmodel_flex_attn"]),
            "param_init_fn": setup_vlm_param_init,
        },
        encoder=Siglip2Config(
            dim=128,
            ffn_dim=256,
            n_layers=4,
            n_heads=2,
        ),
    ),
}


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="vlm",
        flavor=flavor,
        model=llama3_siglip2_configs[flavor],
        parallelize_fn=parallelize_vlm,
        pipelining_fn=None,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )
