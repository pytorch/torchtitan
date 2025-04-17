# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.datasets.tokenizer.tiktoken import build_tiktoken_tokenizer
from torchtitan.models.llama3 import pipeline_llama
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from .infra.parallelize_llama import parallelize_llama
from .model.args import TransformerModelArgs
from .model.model import Transformer

__all__ = [
    "TransformerModelArgs",
    "Transformer",
    "llama4_configs",
]


llama4_configs = {
    "debugmodel": TransformerModelArgs(
        dim=256,
        n_layers=6,
        n_heads=16,
        rope_theta=500000,
    ),
    "17bx16e": TransformerModelArgs(
        dim=5120,
        n_layers=48,
        n_heads=40,
        n_kv_heads=8,
        ffn_dim_multiplier=1.2,
        multiple_of=2048,
        rope_theta=500000,
        num_experts=16,
        interleave_moe_layer_step=1,
    ),
    "17bx128e": TransformerModelArgs(
        dim=5120,
        n_layers=48,
        n_heads=40,
        n_kv_heads=8,
        ffn_dim_multiplier=1.2,
        multiple_of=2048,
        rope_theta=500000,
        num_experts=128,
    ),
    "debugmodel_irope": TransformerModelArgs(
        dim=256,
        n_layers=6,
        n_heads=16,
        rope_theta=500000,
        every_n_layers_nope=4,
        fixed_attn_block_size=256,
        use_flex_attn=True,
        attn_mask_type="block_causal",
    ),
    "17bx16e_irope": TransformerModelArgs(
        dim=5120,
        n_layers=48,
        n_heads=40,
        n_kv_heads=8,
        ffn_dim_multiplier=1.2,
        multiple_of=2048,
        rope_theta=500000,
        num_experts=16,
        interleave_moe_layer_step=1,
        every_n_layers_nope=4,
        use_flex_attn=True,
        attn_mask_type="block_causal",
    ),
    "17bx128e_irope": TransformerModelArgs(
        dim=5120,
        n_layers=48,
        n_heads=40,
        n_kv_heads=8,
        ffn_dim_multiplier=1.2,
        multiple_of=2048,
        rope_theta=500000,
        num_experts=128,
        every_n_layers_nope=4,
        use_flex_attn=True,
        attn_mask_type="block_causal",
    ),
}


register_train_spec(
    TrainSpec(
        name="llama4",
        cls=Transformer,
        config=llama4_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_tiktoken_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
)
