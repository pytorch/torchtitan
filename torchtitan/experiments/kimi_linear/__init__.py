# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Kimi Linear model implementation for torchtitan.

Features:
- Multi-Latent Attention (MLA) for full attention layers
- Key-Delta Attention (KDA) for linear attention layers
- Optional MoE FFN layers
"""

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.validate import build_validator
from torchtitan.hf_datasets.dataloader import build_dataloader
from torchtitan.models.moe import MoEArgs
from torchtitan.protocols.train_spec import TrainSpec

from .infra.parallelize import parallelize_kimi_linear
from .model.args import KimiLinearModelArgs
from .model.model import KimiLinearModel
from .model.state_dict_adapter import KimiLinearStateDictAdapter
from .model.tokenizer import build_kimi_tokenizer

__all__ = [
    "parallelize_kimi_linear",
    "KimiLinearModelArgs",
    "KimiLinearModel",
    "kimi_linear_configs",
]

# Model configurations for different sizes
kimi_linear_configs = {
    # 48B-A3B model (from the reference config)
    "48B_A3B": KimiLinearModelArgs(
        dim=2304,
        n_layers=27,
        vocab_size=163840,
        hidden_dim=9216,
        n_heads=32,
        n_kv_heads=32,
        head_dim=72,  # q_head_dim (qk_nope_head_dim + qk_rope_head_dim for KDA)
        v_head_dim=128,  # value head dimension for MLA
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        norm_eps=1e-5,
        rope_theta=10000.0,
        max_seq_len=8192,
        # Linear attention config
        linear_attn_num_heads=32,
        linear_attn_head_dim=128,
        linear_attn_conv_kernel_size=4,
        # Full attention layers (1-based indexing for HF compatibility)
        full_attn_layers=[4, 8, 12, 16, 20, 24, 27],
        # MoE config
        moe_enabled=True,
        moe_inter_dim=1024,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        moe_args=MoEArgs(
            num_experts=256,
            num_shared_experts=1,
            top_k=8,
            score_func="sigmoid",
            route_norm=True,
            route_scale=2.446,  # routed_scaling_factor from config
            score_before_experts=False,  # HF applies weights to output, not input
        ),
    ),
    # 10B-A1B config (smaller model for testing)
    "10B_A1B": KimiLinearModelArgs(
        dim=1536,
        n_layers=32,
        vocab_size=163840,
        hidden_dim=6144,
        n_heads=24,
        n_kv_heads=24,
        head_dim=64,
        v_head_dim=96,  # value head dimension for MLA
        kv_lora_rank=384,
        qk_nope_head_dim=96,
        qk_rope_head_dim=48,
        norm_eps=1e-5,
        rope_theta=50000.0,
        max_seq_len=8192,
        # Linear attention config
        linear_attn_num_heads=24,
        linear_attn_head_dim=96,
        linear_attn_conv_kernel_size=4,
        # Full attention layers (1-based indexing for HF compatibility)
        full_attn_layers=[3, 7, 11, 15, 19],
        # MoE config
        moe_enabled=True,
        moe_inter_dim=512,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        moe_args=MoEArgs(
            num_experts=128,
            num_shared_experts=1,
            top_k=8,
            score_func="sigmoid",
            route_norm=True,
            route_scale=2.0,
            score_before_experts=False,  # HF applies weights to output, not input
        ),
    ),
    # Dense version without MoE (for testing/debugging)
    "3B_dense": KimiLinearModelArgs(
        dim=2048,
        n_layers=24,
        vocab_size=163840,
        hidden_dim=8192,
        n_heads=16,
        n_kv_heads=16,
        head_dim=128,
        v_head_dim=128,  # value head dimension for MLA
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        norm_eps=1e-5,
        rope_theta=10000.0,
        max_seq_len=8192,
        # Linear attention config
        linear_attn_num_heads=16,
        linear_attn_head_dim=128,
        linear_attn_conv_kernel_size=4,
        # Full attention layers (1-based indexing for HF compatibility)
        full_attn_layers=[5, 11, 17, 23],
        # No MoE
        moe_enabled=False,
    ),
}


def get_train_spec() -> TrainSpec:
    """Return the training specification for Kimi Linear."""
    return TrainSpec(
        model_cls=KimiLinearModel,
        model_args=kimi_linear_configs,
        parallelize_fn=parallelize_kimi_linear,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dataloader,
        build_tokenizer_fn=build_kimi_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=KimiLinearStateDictAdapter,
    )
