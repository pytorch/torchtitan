# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers_with_moe_load_balancing
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.hf_datasets.text_datasets import build_text_dataloader
from torchtitan.models.moe import MoEArgs
from torchtitan.protocols.train_spec import TrainSpec

from .infra.parallelize import parallelize_llama
from .model.args import RoPEScalingArgs, TransformerModelArgs
from .model.model import Transformer
from .model.state_dict_adapter import Llama4StateDictAdapter

__all__ = [
    "TransformerModelArgs",
    "Transformer",
    "llama4_args",
]


llama4_args = {
    "debugmodel": TransformerModelArgs(
        dim=256,
        n_layers=6,
        n_heads=16,
        vocab_size=2048,
        rope_theta=500000,
        rope_scaling_args=RoPEScalingArgs(),
    ),
    "17bx16e": TransformerModelArgs(
        dim=5120,
        n_layers=48,
        n_heads=40,
        n_kv_heads=8,
        ffn_dim_multiplier=1.2,
        multiple_of=2048,
        rope_theta=500000,
        rope_scaling_args=RoPEScalingArgs(),
        max_seq_len=10485760,
        moe_args=MoEArgs(num_experts=16),
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
        moe_args=MoEArgs(num_experts=128),
    ),
    "debugmodel_irope": TransformerModelArgs(
        dim=256,
        n_layers=6,
        n_heads=16,
        vocab_size=2048,
        rope_theta=500000,
        rope_scaling_args=RoPEScalingArgs(),
        every_n_layers_nope=4,
        fixed_attn_block_size=256,
        attn_type="flex",
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
        rope_scaling_args=RoPEScalingArgs(),
        max_seq_len=10485760,
        moe_args=MoEArgs(num_experts=16),
        interleave_moe_layer_step=1,
        every_n_layers_nope=4,
        attn_type="flex",
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
        moe_args=MoEArgs(num_experts=128),
        every_n_layers_nope=4,
        attn_type="flex",
        attn_mask_type="block_causal",
    ),
}


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=Transformer,
        model_args=llama4_args,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_optimizers_fn=build_optimizers_with_moe_load_balancing,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Llama4StateDictAdapter,
    )
