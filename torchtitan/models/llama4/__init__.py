# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers_with_moe_load_balancing
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import Validator
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.hf_datasets.text_datasets import build_text_dataloader
from torchtitan.models.common import (
    compute_ffn_hidden_dim,
    FeedForward,
    GQAttention,
    RoPE,
)
from torchtitan.models.common.moe import MoE
from torchtitan.protocols.train_spec import TrainSpec

from .infra.parallelize import parallelize_llama
from .model.model import compute_moe_hidden_dim, Transformer
from .model.state_dict_adapter import Llama4StateDictAdapter

__all__ = [
    "Transformer",
    "llama4_configs",
]


llama4_configs = {
    "debugmodel": Transformer.Config(
        dim=256,
        n_layers=6,
        vocab_size=2048,
        ff_config=FeedForward.Config(hidden_dim=compute_ffn_hidden_dim(256)),
        rope_config=RoPE.Config(
            dim=256 // 16,
            max_seq_len=1048576,
            theta=500000,
            format="complex",
            scaling="llama",
            scaling_factor=16.0,
            high_freq_factor=1.0,
        ),
        attn_config=GQAttention.Config(
            n_heads=16,
            attn_type="sdpa",
            rope_format="complex",
        ),
        moe_config=MoE.Config(hidden_dim=compute_moe_hidden_dim(256)),
    ),
    "17bx16e": Transformer.Config(
        dim=5120,
        n_layers=48,
        max_seq_len=10485760,
        moe_config=MoE.Config(
            num_experts=16,
            hidden_dim=compute_moe_hidden_dim(
                5120,
                multiple_of=2048,
                ffn_dim_multiplier=1.2,
                top_k=1,
                num_shared_experts=1,
            ),
        ),
        interleave_moe_layer_step=1,
        ff_config=FeedForward.Config(
            hidden_dim=compute_ffn_hidden_dim(
                5120, multiple_of=2048, ffn_dim_multiplier=1.2
            ),
        ),
        rope_config=RoPE.Config(
            dim=5120 // 40,
            max_seq_len=10485760,
            theta=500000,
            format="complex",
            scaling="llama",
            scaling_factor=16.0,
            high_freq_factor=1.0,
        ),
        attn_config=GQAttention.Config(
            n_heads=40,
            n_kv_heads=8,
            attn_type="sdpa",
            rope_format="complex",
        ),
    ),
    "17bx128e": Transformer.Config(
        dim=5120,
        n_layers=48,
        moe_config=MoE.Config(
            num_experts=128,
            hidden_dim=compute_moe_hidden_dim(
                5120,
                multiple_of=2048,
                ffn_dim_multiplier=1.2,
                top_k=1,
                num_shared_experts=1,
            ),
        ),
        ff_config=FeedForward.Config(
            hidden_dim=compute_ffn_hidden_dim(
                5120, multiple_of=2048, ffn_dim_multiplier=1.2
            ),
        ),
        rope_config=RoPE.Config(
            dim=5120 // 40,
            max_seq_len=1048576,
            theta=500000,
            format="complex",
            scaling="none",
        ),
        attn_config=GQAttention.Config(
            n_heads=40,
            n_kv_heads=8,
            attn_type="sdpa",
            rope_format="complex",
        ),
    ),
    "debugmodel_irope": Transformer.Config(
        dim=256,
        n_layers=6,
        vocab_size=2048,
        every_n_layers_nope=4,
        fixed_attn_block_size=256,
        attn_mask_type="block_causal",
        ff_config=FeedForward.Config(hidden_dim=compute_ffn_hidden_dim(256)),
        rope_config=RoPE.Config(
            dim=256 // 16,
            max_seq_len=1048576,
            theta=500000,
            format="complex",
            scaling="llama",
            scaling_factor=16.0,
            high_freq_factor=1.0,
        ),
        attn_config=GQAttention.Config(
            n_heads=16,
            attn_type="flex",
            rope_format="complex",
        ),
        moe_config=MoE.Config(hidden_dim=compute_moe_hidden_dim(256)),
    ),
    "17bx16e_irope": Transformer.Config(
        dim=5120,
        n_layers=48,
        max_seq_len=10485760,
        moe_config=MoE.Config(
            num_experts=16,
            hidden_dim=compute_moe_hidden_dim(
                5120,
                multiple_of=2048,
                ffn_dim_multiplier=1.2,
                top_k=1,
                num_shared_experts=1,
            ),
        ),
        interleave_moe_layer_step=1,
        every_n_layers_nope=4,
        attn_mask_type="block_causal",
        ff_config=FeedForward.Config(
            hidden_dim=compute_ffn_hidden_dim(
                5120, multiple_of=2048, ffn_dim_multiplier=1.2
            ),
        ),
        rope_config=RoPE.Config(
            dim=5120 // 40,
            max_seq_len=10485760,
            theta=500000,
            format="complex",
            scaling="llama",
            scaling_factor=16.0,
            high_freq_factor=1.0,
        ),
        attn_config=GQAttention.Config(
            n_heads=40,
            n_kv_heads=8,
            attn_type="flex",
            rope_format="complex",
        ),
    ),
    "17bx128e_irope": Transformer.Config(
        dim=5120,
        n_layers=48,
        moe_config=MoE.Config(
            num_experts=128,
            hidden_dim=compute_moe_hidden_dim(
                5120,
                multiple_of=2048,
                ffn_dim_multiplier=1.2,
                top_k=1,
                num_shared_experts=1,
            ),
        ),
        every_n_layers_nope=4,
        attn_mask_type="block_causal",
        ff_config=FeedForward.Config(
            hidden_dim=compute_ffn_hidden_dim(
                5120, multiple_of=2048, ffn_dim_multiplier=1.2
            ),
        ),
        rope_config=RoPE.Config(
            dim=5120 // 40,
            max_seq_len=1048576,
            theta=500000,
            format="complex",
            scaling="none",
        ),
        attn_config=GQAttention.Config(
            n_heads=40,
            n_kv_heads=8,
            attn_type="flex",
            rope_format="complex",
        ),
    ),
}


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_configs=llama4_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_optimizers_fn=build_optimizers_with_moe_load_balancing,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        validator_cls=Validator,
        state_dict_adapter=Llama4StateDictAdapter,
    )
