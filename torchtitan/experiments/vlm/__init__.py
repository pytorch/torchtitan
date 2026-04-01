# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
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


def _expand_vlm_layer_configs(config) -> None:
    """Expand encoder layer template into per-layer configs via deepcopy.

    Also sets computed init=False fields (dim, patch_in_features) on the
    encoder and projector sub-configs.
    Mutates config in place.
    """
    encoder = config.encoder
    dim = encoder.dim

    # Set computed fields on embeddings
    encoder.embeddings.dim = dim
    encoder.embeddings.patch_in_features = (
        encoder.n_channels * encoder.patch_size * encoder.patch_size
    )

    # Set computed fields on layer template
    encoder.layer.dim = dim
    encoder.layer.self_attn.dim = dim
    encoder.layer.mlp.dim = dim

    # Expand layers
    encoder.layers = [deepcopy(encoder.layer) for _ in range(encoder.n_layers)]

    # Set computed fields on projector
    config.projector.in_dim = encoder.dim
    config.projector.out_dim = config.dim


def _debugmodel() -> Llama3Siglip2Transformer.Config:
    from torchtitan.models.llama3 import expand_layer_configs as _expand_llama3

    base = llama3_configs["debugmodel_flex_attn"]()
    _expand_llama3(base)
    return Llama3Siglip2Transformer.Config(
        **_get_dict(base),
        encoder=VisionTransformer.Config(
            dim=128,
            n_layers=4,
            layer_norm_eps=1e-6,
            attn_mask_type="causal",
            embeddings=VisionEmbeddings.Config(
                n_pos_embs=16,
                patch_embedding=Linear.Config(bias=True, param_init=_XAVIER_LINEAR),
                position_embedding=Embedding.Config(param_init=_EMBEDDING_INIT),
            ),
            layer=VisionTransformerLayer.Config(
                layer_norm_eps=1e-6,
                self_attn=VisionAttention.Config(
                    n_heads=2,
                    qkv=Linear.Config(bias=True, param_init=_XAVIER_LINEAR),
                    out_proj=Linear.Config(bias=True, param_init=_XAVIER_LINEAR),
                ),
                mlp=VisionFeedForward.Config(
                    ffn_dim=256,
                    fc1=Linear.Config(bias=True, param_init=_XAVIER_LINEAR),
                    fc2=Linear.Config(bias=True, param_init=_XAVIER_LINEAR),
                ),
            ),
        ),
        projector=Projector.Config(
            w1=Linear.Config(bias=True, param_init=_XAVIER_LINEAR),
            w2=Linear.Config(bias=True, param_init=_XAVIER_LINEAR),
        ),
    )


llama3_siglip2_configs = {"debugmodel": _debugmodel}


def model_registry(flavor: str) -> ModelSpec:
    config = llama3_siglip2_configs[flavor]()
    _expand_vlm_layer_configs(config)
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
