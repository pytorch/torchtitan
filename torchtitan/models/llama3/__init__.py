# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.hf_datasets.dataloader import build_dataloader
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.protocols.train_spec import TrainSpec

from .infra.parallelize import parallelize_llama
from .model.args import TransformerModelArgs
from .model.model import Transformer
from .model.state_dict_adapter import Llama3StateDictAdapter

__all__ = [
    "parallelize_llama",
    "TransformerModelArgs",
    "Transformer",
    "llama3_args",
]


llama3_args = {
    "debugmodel": TransformerModelArgs(
        dim=256, n_layers=6, n_heads=16, vocab_size=2048, rope_theta=500000
    ),
    "debugmodel_flex_attn": TransformerModelArgs(
        dim=256,
        n_layers=6,
        n_heads=16,
        vocab_size=2048,
        rope_theta=500000,
        use_flex_attn=True,
        attn_mask_type="block_causal",
    ),
    "8B": TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
    ),
    "8B_flex_attn": TransformerModelArgs(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=1024,
        rope_theta=500000,
        use_flex_attn=True,
        attn_mask_type="block_causal_by_sequence_lengths",
    ),
    "70B": TransformerModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        rope_theta=500000,
    ),
    "70B_flex_attn": TransformerModelArgs(
        dim=8192,
        n_layers=80,
        n_heads=64,
        n_kv_heads=8,
        ffn_dim_multiplier=1.3,
        multiple_of=4096,
        rope_theta=500000,
        use_flex_attn=True,
        attn_mask_type="block_causal_by_sequence_lengths",
    ),
    "405B": TransformerModelArgs(
        dim=16384,
        n_layers=126,
        n_heads=128,
        n_kv_heads=8,
        ffn_dim_multiplier=1.2,
        multiple_of=4096,
        rope_theta=500000,
    ),
    "36B_seed_flex_attn": TransformerModelArgs(
        dim=5120,
        n_layers=64,
        n_heads=80,
        n_kv_heads=8,
        ffn_dim_multiplier=2,
        multiple_of=432,
        rope_theta=10000000,
        use_flex_attn=True,
        attn_mask_type="block_causal_by_sequence_lengths",
        use_qkv_bias=True,
        vocab_size=155136,
        head_dim=128,
        rope_scaling_args=None, # no rope scaling
    ),
}


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=Transformer,
        model_args=llama3_args,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Llama3StateDictAdapter,
    )
