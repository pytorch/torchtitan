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
class TitanModelConfig:
    """Base model config for the transformers backend."""

    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int | None = None
    vocab_size: int | None = None
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None
    intermediate_size: int | None = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000
    max_seq_len: int = 2048
    depth_init: bool = True
    use_flex_attn: bool = False
    attn_mask_type: str = "causal"


@dataclass
class TitanMoeModelConfig(TitanModelConfig):
    """MoE model config — adds expert parameters to the base config."""

    num_experts: int = 128
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 768
    decoder_sparse_step: int = 1
    norm_topk_prob: bool = False
    num_nextn_predict_layers: int | None = None


flavors = {
    "debugmodel": HFTransformerModel.Config(
        model_config=TitanModelConfig(
            dim=256,
            n_layers=2,
            n_heads=16,
            n_kv_heads=16,
        ),
    ),
    # NOTE: dim=2048 is the minimum required for models with LoRA attention
    # (DeepSeek V3 needs q_lora_rank=1536, GLM-5 needs q_lora_rank=2048).
    # n_heads must be divisible by TP degree and n_kv_heads.
    "debugmodel_moe": HFTransformerModel.Config(
        model_config=TitanMoeModelConfig(
            dim=2048,
            n_layers=4,
            n_heads=16,
            n_kv_heads=8,
            intermediate_size=512,
            num_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=128,
            num_nextn_predict_layers=0,
        ),
    ),
    "full": HFTransformerModel.Config(
        model_config=TitanModelConfig(),
    ),
    "full_moe": HFTransformerModel.Config(
        model_config=TitanMoeModelConfig(
            dim=2048,
            n_layers=48,
            n_heads=32,
            n_kv_heads=4,
            norm_eps=1e-6,
            num_experts=128,
            num_experts_per_tok=8,
            moe_intermediate_size=768,
            norm_topk_prob=True,
        ),
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
