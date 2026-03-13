# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.protocols.model_spec import ModelSpec

from .model import Nemotron3Config, Nemotron3Model
from .parallelize import parallelize_nemotron3
from .state_dict_adapter import Nemotron3StateDictAdapter

__all__ = [
    "Nemotron3Config",
    "Nemotron3Model",
    "Nemotron3StateDictAdapter",
    "parallelize_nemotron3",
    "nemotron3_configs",
    "model_registry",
]


# NemotronH model flavors
# Pattern key: M=Mamba2, *=Attention, E=MoE
nemotron3_configs = {
    # Debug model for testing
    "debugmodel": Nemotron3Config(
        vocab_size=131072,
        dim=1024,
        hidden_dim=4096,
        n_layers=3,
        hybrid_override_pattern="M*E",
        n_heads=16,
        head_dim=64,
        n_kv_heads=8,
        max_seq_len=4096,
        mamba_num_heads=16,
        mamba_head_dim=64,
        attn_type="sdpa",
    ),
    # NemotronH-nano-30B configuration
    # From https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/blob/main/config.json
    "nano-30B": Nemotron3Config(
        vocab_size=131072,
        dim=2688,  # hidden_size
        hidden_dim=1856,  # intermediate_size
        n_layers=52,
        hybrid_override_pattern="MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
        n_heads=32,
        head_dim=128,
        n_kv_heads=2,  # num_key_value_heads
        max_seq_len=262144,  # max_position_embeddings
        mlp_hidden_act="relu2",
        attn_bias=False,
        mlp_bias=False,
        initializer_range=0.02,
        norm_eps=1e-5,
        residual_in_fp32=False,
        # Mamba2 config
        ssm_state_size=128,
        mamba_num_heads=64,
        mamba_n_groups=8,
        mamba_head_dim=64,
        mamba_d_conv=4,
        mamba_hidden_act="silu",
        mamba_conv_bias=True,
        mamba_proj_bias=False,
        mamba_chunk_size=128,
        # MoE config
        n_routed_experts=128,
        moe_intermediate_size=1856,
        moe_shared_expert_intermediate_size=3712,
        num_experts_per_tok=6,
        routed_scaling_factor=2.5,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
        attn_type="sdpa",
    ),
}


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="nemotron3",
        flavor=flavor,
        model=nemotron3_configs[flavor],
        parallelize_fn=parallelize_nemotron3,
        pipelining_fn=None,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=Nemotron3StateDictAdapter,
    )
