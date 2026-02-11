# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
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
from torchtitan.protocols.train_spec import TrainSpec

from .infra.parallelize import parallelize_llama
from .model.model import Transformer
from .model.state_dict_adapter import Llama3StateDictAdapter

__all__ = [
    "parallelize_llama",
    "Transformer",
    "llama3_configs",
]


llama3_configs = {
    "debugmodel": Transformer.Config(
        dim=256,
        n_layers=6,
        vocab_size=2048,
        ff_config=FeedForward.Config(hidden_dim=compute_ffn_hidden_dim(256)),
        rope_config=RoPE.Config(
            dim=256 // 16,
            max_seq_len=131072,
            theta=500000,
            format="complex",
            scaling="llama",
        ),
        attn_config=GQAttention.Config(
            n_heads=16, attn_type="sdpa", rope_format="complex"
        ),
    ),
    "debugmodel_flex_attn": Transformer.Config(
        dim=256,
        n_layers=6,
        vocab_size=2048,
        attn_mask_type="block_causal",
        ff_config=FeedForward.Config(hidden_dim=compute_ffn_hidden_dim(256)),
        rope_config=RoPE.Config(
            dim=256 // 16,
            max_seq_len=131072,
            theta=500000,
            format="complex",
            scaling="llama",
        ),
        attn_config=GQAttention.Config(
            n_heads=16, attn_type="flex", rope_format="complex"
        ),
    ),
    "debugmodel_varlen_attn": Transformer.Config(
        dim=256,
        n_layers=6,
        vocab_size=2048,
        attn_mask_type="block_causal",
        ff_config=FeedForward.Config(hidden_dim=compute_ffn_hidden_dim(256)),
        rope_config=RoPE.Config(
            dim=256 // 16,
            max_seq_len=131072,
            theta=500000,
            format="complex",
            scaling="llama",
        ),
        attn_config=GQAttention.Config(
            n_heads=16, attn_type="varlen", rope_format="complex"
        ),
    ),
    "8B": Transformer.Config(
        dim=4096,
        n_layers=32,
        ff_config=FeedForward.Config(
            hidden_dim=compute_ffn_hidden_dim(
                4096, multiple_of=1024, ffn_dim_multiplier=1.3
            )
        ),
        rope_config=RoPE.Config(
            dim=4096 // 32,
            max_seq_len=131072,
            theta=500000,
            format="complex",
            scaling="llama",
        ),
        attn_config=GQAttention.Config(
            n_heads=32, n_kv_heads=8, attn_type="sdpa", rope_format="complex"
        ),
    ),
    "8B_flex": Transformer.Config(
        dim=4096,
        n_layers=32,
        attn_mask_type="block_causal",
        ff_config=FeedForward.Config(
            hidden_dim=compute_ffn_hidden_dim(
                4096, multiple_of=1024, ffn_dim_multiplier=1.3
            )
        ),
        rope_config=RoPE.Config(
            dim=4096 // 32,
            max_seq_len=131072,
            theta=500000,
            format="complex",
            scaling="llama",
        ),
        attn_config=GQAttention.Config(
            n_heads=32, n_kv_heads=8, attn_type="flex", rope_format="complex"
        ),
    ),
    "8B_varlen": Transformer.Config(
        dim=4096,
        n_layers=32,
        attn_mask_type="block_causal",
        ff_config=FeedForward.Config(
            hidden_dim=compute_ffn_hidden_dim(
                4096, multiple_of=1024, ffn_dim_multiplier=1.3
            )
        ),
        rope_config=RoPE.Config(
            dim=4096 // 32,
            max_seq_len=131072,
            theta=500000,
            format="complex",
            scaling="llama",
        ),
        attn_config=GQAttention.Config(
            n_heads=32, n_kv_heads=8, attn_type="varlen", rope_format="complex"
        ),
    ),
    "70B": Transformer.Config(
        dim=8192,
        n_layers=80,
        ff_config=FeedForward.Config(
            hidden_dim=compute_ffn_hidden_dim(
                8192, multiple_of=4096, ffn_dim_multiplier=1.3
            )
        ),
        rope_config=RoPE.Config(
            dim=8192 // 64,
            max_seq_len=131072,
            theta=500000,
            format="complex",
            scaling="llama",
        ),
        attn_config=GQAttention.Config(
            n_heads=64, n_kv_heads=8, attn_type="sdpa", rope_format="complex"
        ),
    ),
    "405B": Transformer.Config(
        dim=16384,
        n_layers=126,
        ff_config=FeedForward.Config(
            hidden_dim=compute_ffn_hidden_dim(
                16384, multiple_of=4096, ffn_dim_multiplier=1.2
            )
        ),
        rope_config=RoPE.Config(
            dim=16384 // 128,
            max_seq_len=131072,
            theta=500000,
            format="complex",
            scaling="llama",
        ),
        attn_config=GQAttention.Config(
            n_heads=128, n_kv_heads=8, attn_type="sdpa", rope_format="complex"
        ),
    ),
}


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_configs=llama3_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        validator_cls=Validator,
        state_dict_adapter=Llama3StateDictAdapter,
    )
