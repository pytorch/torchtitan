# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields
from functools import partial
from typing import Any

import torch.nn as nn

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.linear import Linear
from torchtitan.models.llama3 import llama3_configs
from torchtitan.protocols.model_spec import ModelSpec

from .datasets.mm_datasets import HuggingFaceMultiModalDataLoader
from .infra.parallelize import parallelize_vlm
from .model.model import Llama3Siglip2Transformer, Projector
from .model.siglip2 import (
    Attention as VisionAttention,
    FeedForward as VisionFeedForward,
    TransformerLayer as VisionTransformerLayer,
    VisionEmbeddings,
    VisionTransformer,
)

__all__ = [
    "HuggingFaceMultiModalDataLoader",
    "parallelize_vlm",
    "Llama3Siglip2Transformer",
    "llama3_siglip2_configs",
]

_XAVIER_LINEAR = {"weight": nn.init.xavier_uniform_, "bias": nn.init.zeros_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=0.02)}


def _get_dict(obj) -> dict[str, Any]:
    """Convert dataclass to dict, preserving nested dataclasses (unlike asdict)."""
    return {field.name: getattr(obj, field.name) for field in fields(obj)}


def _debugmodel() -> Llama3Siglip2Transformer.Config:
    base = llama3_configs["debugmodel_flex_attn"]()
    dim = 128
    ffn_dim = 256
    n_layers = 4
    n_channels = 3
    patch_size = 16
    n_pos_embs = 16
    patch_in_features = n_channels * patch_size * patch_size

    layer_config = VisionTransformerLayer.Config(
        dim=dim,
        layer_norm_eps=1e-6,
        self_attn=VisionAttention.Config(
            dim=dim,
            n_heads=2,
            qkv_proj=Linear.Config(
                in_features=dim,
                out_features=dim,
                bias=True,
                param_init=_XAVIER_LINEAR,
            ),
            out_proj=Linear.Config(
                in_features=dim,
                out_features=dim,
                bias=True,
                param_init=_XAVIER_LINEAR,
            ),
        ),
        mlp=VisionFeedForward.Config(
            fc1=Linear.Config(
                in_features=dim,
                out_features=ffn_dim,
                bias=True,
                param_init=_XAVIER_LINEAR,
            ),
            fc2=Linear.Config(
                in_features=ffn_dim,
                out_features=dim,
                bias=True,
                param_init=_XAVIER_LINEAR,
            ),
        ),
    )

    llm_dim = base.dim
    return Llama3Siglip2Transformer.Config(
        **_get_dict(base),
        encoder=VisionTransformer.Config(
            dim=dim,
            layer_norm_eps=1e-6,
            attn_mask_type="causal",
            embeddings=VisionEmbeddings.Config(
                n_pos_embs=n_pos_embs,
                patch_embedding=Linear.Config(
                    in_features=patch_in_features,
                    out_features=dim,
                    bias=True,
                    param_init=_XAVIER_LINEAR,
                ),
                position_embedding=Embedding.Config(
                    num_embeddings=n_pos_embs**2,
                    embedding_dim=dim,
                    param_init=_EMBEDDING_INIT,
                ),
            ),
            layers=[layer_config] * n_layers,
        ),
        projector=Projector.Config(
            w1=Linear.Config(
                in_features=dim,
                out_features=dim,
                bias=True,
                param_init=_XAVIER_LINEAR,
            ),
            w2=Linear.Config(
                in_features=dim,
                out_features=llm_dim,
                bias=True,
                param_init=_XAVIER_LINEAR,
            ),
        ),
    )


llama3_siglip2_configs = {"debugmodel": _debugmodel}


def model_registry(flavor: str) -> ModelSpec:
    config = llama3_siglip2_configs[flavor]()
    return ModelSpec(
        name="vlm",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_vlm,
        pipelining_fn=None,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )
