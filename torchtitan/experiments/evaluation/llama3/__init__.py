# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.models.llama3 import pipeline_llama, parallelize_llama
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from .model import Transformer, TransformerModelArgs

__all__ = [
    "TransformerModelArgs",
    "Transformer",
    "llama3_configs",
]


llama3_configs = {
    "1b": TransformerModelArgs(  # Llama 3.2 1B
        dim=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        ffn_dim_multiplier=1.4,
        multiple_of=1024,
        rope_theta=500000,
        use_llama3_rope=True,
        factor=32.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=8192,
        tie_word_embeddings=True,
    ),
    "3b": TransformerModelArgs(  # Llama 3.2 3B
        dim=3072,
        n_layers=28,
        n_heads=24,
        n_kv_heads=8,
        ffn_dim_multiplier=1.,
        multiple_of=1024,
        rope_theta=500000,
        use_llama3_rope=True,
        factor=32.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=8192,
        tie_word_embeddings=True,
    ),
}


register_train_spec(
    TrainSpec(
        name="llama3.2",
        model_cls=Transformer,
        model_args=llama3_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
)
