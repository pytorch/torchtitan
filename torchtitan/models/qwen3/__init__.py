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
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from .infra.parallelize import parallelize_qwen3
from .infra.pipeline import pipeline_qwen3
from .model.args import TransformerModelArgs
from .model.model import Transformer

__all__ = [
    "parallelize_qwen3",
    "pipeline_qwen3",
    "TransformerModelArgs",
    "Transformer",
    "qwen3_configs",
]


qwen3_configs = {
    "0.6B": TransformerModelArgs(
        vocab_size=151_936,
        max_seq_len=40_960,
        dim=1024,
        n_layers=36,
        n_heads=16,
        n_kv_heads=8,
        qk_norm=True,
        hidden_dim=2560,
        rope_theta=1000000,
    )
}


register_train_spec(
    TrainSpec(
        name="qwen3",
        model_cls=Transformer,
        model_args=qwen3_configs,  # Change from dict to Mapping
        parallelize_fn=parallelize_qwen3,
        pipelining_fn=pipeline_qwen3,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
    )
)
