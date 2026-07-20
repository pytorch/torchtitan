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

    Two kinds of fields (see the groups below): those that mirror an HF config
    key default to None so AutoConfig's value is used; TorchTitan-only fields
    keep concrete defaults since they don't override anything from the HF config.
    """

    # Fields that map to an HF config key: default None so the value from
    # AutoConfig.from_pretrained is kept. A non-None default would be injected
    # over the HF config and force the wrong architecture (e.g. rope_theta).
    # Set explicitly only to intentionally override (e.g. debugmodel sizes).
    dim: int | None = None
    n_layers: int | None = None
    n_heads: int | None = None
    n_kv_heads: int | None = None
    vocab_size: int | None = None
    norm_eps: float | None = None
    rope_theta: float | None = None

    # TorchTitan-only fields with no HF equivalent: they don't override anything
    # from the HF config, so they keep concrete defaults. (multiple_of and
    # ffn_dim_multiplier are only used when deriving FFN size from an explicitly
    # overridden dim; max_seq_len is set from training.batch.seq_len.)
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None
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
    "sft_debugmodel": HFTransformerModel.Config(
        titan_dense_config=TitanDenseModelConfig(
            dim=256,
            n_layers=2,
            n_heads=16,
            n_kv_heads=16,
            attn_mask_type="block_causal",
        ),
        attn_implementation="flex_torchtitan",
    ),
    "full": HFTransformerModel.Config(
        titan_dense_config=TitanDenseModelConfig(),
    ),
    "sft_full": HFTransformerModel.Config(
        titan_dense_config=TitanDenseModelConfig(
            attn_mask_type="block_causal",
        ),
        attn_implementation="flex_torchtitan",
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
