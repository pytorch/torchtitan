# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields
from typing import Any

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.models.llama3 import llama3_configs
from torchtitan.protocols.model_spec import ModelSpec

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


def _get_dict(obj) -> dict[str, Any]:
    """Convert dataclass to dict, preserving nested dataclasses (unlike asdict)."""
    return {field.name: getattr(obj, field.name) for field in fields(obj)}


llama3_siglip2_configs = {
    "debugmodel": Llama3Siglip2Transformer.Config(
        **_get_dict(llama3_configs["debugmodel_flex_attn"]),
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
