# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass

from torchtitan.protocols.model_spec import ModelSpec
from .model import HFTransformerModel

from .parallelize import parallelize_hf_transformers
from .pipeline import pipeline_hf_transformers
from .state_dict_adapter import HFTransformerStateDictAdapter

__all__ = [
    "HFTransformerModel",
]


@dataclass
class TitanDenseModelConfig:
    """Arguments for the base TorchTitan model.

    HF-derived fields default to None so they do NOT override the values loaded
    from the HF config via AutoConfig.from_pretrained(). A non-None default is
    injected over the HF config (see HFTransformerModel.Config.update_from_config),
    which silently forces the wrong architecture/hyperparameters for any model
    whose config differs (e.g. rope_theta=1e6 for Qwen3). Set a field explicitly
    only to intentionally override the HF config (e.g. debugmodel sizes).
    """

    dim: int | None = None
    n_layers: int | None = None
    n_heads: int | None = None
    n_kv_heads: int | None = None
    vocab_size: int | None = None
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None
    norm_eps: float | None = None
    rope_theta: float | None = None
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
        post_optimizer_build_fn=None,
        state_dict_adapter=HFTransformerStateDictAdapter,
    )
