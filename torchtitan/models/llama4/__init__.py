# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import torch.nn as nn

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.models.common import (
    compute_ffn_hidden_dim,
    Embedding,
    FeedForward,
    GQAttention,
    Linear,
    RMSNorm,
    RoPE,
)
from torchtitan.models.common.moe import MoE, TokenChoiceTopKRouter
from torchtitan.models.common.moe.moe import GroupedExperts
from torchtitan.models.common.param_init import depth_scaled_std, PerLayer
from torchtitan.protocols.model_spec import ModelSpec

from .model import compute_moe_hidden_dim, Llama4Model, Llama4TransformerBlock

from .parallelize import parallelize_llama
from .state_dict_adapter import Llama4StateDictAdapter

__all__ = [
    "Llama4Model",
    "llama4_configs",
]

_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_LINEAR_DEPTH_INIT = PerLayer(
    lambda layer_id: {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }
)
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}


def _output_linear_init(dim: int):
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


llama4_configs = {
    "debugmodel": Llama4Model.Config(
        dim=256,
        n_layers=6,
        vocab_size=2048,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        norm=RMSNorm.Config(param_init=_NORM_INIT),
        output=Linear.Config(param_init=_output_linear_init(256)),
        layer=Llama4TransformerBlock.Config(
            every_n_layers_nope=4,
            fixed_attn_block_size=256,
            attention_norm=RMSNorm.Config(param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(param_init=_NORM_INIT),
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(256, multiple_of=256),
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            attention=GQAttention.Config(
                n_heads=16,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                attn_backend="flex",
                attn_mask_type="block_causal",
                rope_backend="complex",
            ),
            moe=MoE.Config(
                hidden_dim=compute_moe_hidden_dim(256),
                router=TokenChoiceTopKRouter.Config(
                    gate=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
                experts=GroupedExperts.Config(
                    param_init=PerLayer(
                        lambda layer_id: {
                            "w1": partial(nn.init.trunc_normal_, std=0.02),
                            "w2": partial(
                                nn.init.trunc_normal_,
                                std=depth_scaled_std(0.02, layer_id),
                            ),
                            "w3": partial(
                                nn.init.trunc_normal_,
                                std=depth_scaled_std(0.02, layer_id),
                            ),
                        }
                    )
                ),
                shared_experts=FeedForward.Config(
                    hidden_dim=compute_moe_hidden_dim(256),
                    w1=Linear.Config(param_init=_LINEAR_INIT),
                    w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
            ),
        ),
        rope=RoPE.Config(
            dim=256 // 16,
            max_seq_len=1048576,
            theta=500000,
            backend="complex",
            scaling="llama",
            scaling_factor=16.0,
            high_freq_factor=1.0,
        ),
    ),
    "17bx16e": Llama4Model.Config(
        dim=5120,
        n_layers=48,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        norm=RMSNorm.Config(param_init=_NORM_INIT),
        output=Linear.Config(param_init=_output_linear_init(5120)),
        layer=Llama4TransformerBlock.Config(
            every_n_layers_nope=4,
            interleave_moe_layer_step=1,
            attention_norm=RMSNorm.Config(param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(param_init=_NORM_INIT),
            moe=MoE.Config(
                num_experts=16,
                hidden_dim=compute_moe_hidden_dim(
                    5120,
                    multiple_of=2048,
                    ffn_dim_multiplier=1.2,
                    top_k=1,
                    num_shared_experts=1,
                ),
                router=TokenChoiceTopKRouter.Config(
                    gate=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
                experts=GroupedExperts.Config(
                    param_init=PerLayer(
                        lambda layer_id: {
                            "w1": partial(nn.init.trunc_normal_, std=0.02),
                            "w2": partial(
                                nn.init.trunc_normal_,
                                std=depth_scaled_std(0.02, layer_id),
                            ),
                            "w3": partial(
                                nn.init.trunc_normal_,
                                std=depth_scaled_std(0.02, layer_id),
                            ),
                        }
                    )
                ),
                shared_experts=FeedForward.Config(
                    hidden_dim=compute_moe_hidden_dim(
                        5120,
                        multiple_of=2048,
                        ffn_dim_multiplier=1.2,
                        top_k=1,
                        num_shared_experts=1,
                    ),
                    w1=Linear.Config(param_init=_LINEAR_INIT),
                    w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
            ),
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    5120, multiple_of=2048, ffn_dim_multiplier=1.2
                ),
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            attention=GQAttention.Config(
                n_heads=40,
                n_kv_heads=8,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                attn_backend="flex",
                attn_mask_type="block_causal",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=5120 // 40,
            max_seq_len=10485760,
            theta=500000,
            backend="complex",
            scaling="llama",
            scaling_factor=16.0,
            high_freq_factor=1.0,
        ),
    ),
    "17bx128e": Llama4Model.Config(
        dim=5120,
        n_layers=48,
        tok_embeddings=Embedding.Config(param_init=_EMBEDDING_INIT),
        norm=RMSNorm.Config(param_init=_NORM_INIT),
        output=Linear.Config(param_init=_output_linear_init(5120)),
        layer=Llama4TransformerBlock.Config(
            every_n_layers_nope=4,
            attention_norm=RMSNorm.Config(param_init=_NORM_INIT),
            ffn_norm=RMSNorm.Config(param_init=_NORM_INIT),
            moe=MoE.Config(
                num_experts=128,
                hidden_dim=compute_moe_hidden_dim(
                    5120,
                    multiple_of=2048,
                    ffn_dim_multiplier=1.2,
                    top_k=1,
                    num_shared_experts=1,
                ),
                router=TokenChoiceTopKRouter.Config(
                    gate=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
                experts=GroupedExperts.Config(
                    param_init=PerLayer(
                        lambda layer_id: {
                            "w1": partial(nn.init.trunc_normal_, std=0.02),
                            "w2": partial(
                                nn.init.trunc_normal_,
                                std=depth_scaled_std(0.02, layer_id),
                            ),
                            "w3": partial(
                                nn.init.trunc_normal_,
                                std=depth_scaled_std(0.02, layer_id),
                            ),
                        }
                    )
                ),
                shared_experts=FeedForward.Config(
                    hidden_dim=compute_moe_hidden_dim(
                        5120,
                        multiple_of=2048,
                        ffn_dim_multiplier=1.2,
                        top_k=1,
                        num_shared_experts=1,
                    ),
                    w1=Linear.Config(param_init=_LINEAR_INIT),
                    w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                ),
            ),
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(
                    5120, multiple_of=2048, ffn_dim_multiplier=1.2
                ),
                w1=Linear.Config(param_init=_LINEAR_INIT),
                w2w3=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
            ),
            attention=GQAttention.Config(
                n_heads=40,
                n_kv_heads=8,
                wqkv=Linear.Config(param_init=_LINEAR_INIT),
                wo=Linear.Config(param_init=_LINEAR_DEPTH_INIT),
                attn_backend="flex",
                attn_mask_type="block_causal",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=5120 // 40,
            max_seq_len=1048576,
            theta=500000,
            backend="complex",
            scaling="none",
        ),
    ),
}


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="llama4",
        flavor=flavor,
        model=llama4_configs[flavor],
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=Llama4StateDictAdapter,
    )
