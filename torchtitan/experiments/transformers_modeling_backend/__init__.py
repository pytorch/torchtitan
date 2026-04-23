# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.protocols.model_spec import ModelSpec
from .model import HFTransformerModel

from .parallelize import parallelize_hf_transformers
from .pipeline import pipeline_hf_transformers

__all__ = [
    "HFTransformerModel",
]


@dataclass
class TitanDenseModelConfig:
    """Arguments for the base TorchTitan model."""

    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int | None = None
    vocab_size: int | None = None
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000
    max_seq_len: int = 2048
    depth_init: bool = True
    use_flex_attn: bool = False
    attn_mask_type: str = "causal"


flavors = {
    "debugmodel": HFTransformerModel.Config(
        titan_dense_config=TitanDenseModelConfig(
            dim=256,
            n_layers=2,
            n_heads=16,
            n_kv_heads=16,
        ),
    ),
    "full": HFTransformerModel.Config(
        titan_dense_config=TitanDenseModelConfig(),
    ),
}


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="transformers_modeling_backend",
        flavor=flavor,
        model=flavors[flavor],
        parallelize_fn=parallelize_hf_transformers,
        pipelining_fn=pipeline_hf_transformers,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )
