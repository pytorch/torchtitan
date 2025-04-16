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
from .model.args import DSV2ModelArgs
from .model.model import Transformer

__all__ = ["DSV2ModelArgs", "deepseek_configs"]


deepseek_configs = {
    "debugmodel": DSV2ModelArgs(
        dim=256,
        n_layers=8,
        n_heads=16,
        rope_theta=500000,
    ),
    "DeepSeekv2": DSV2ModelArgs(),
}


register_train_spec(
    TrainSpec(
        name="deepseekv2",
        cls=Transformer,
        config=deepseek_configs,
        parallelize_fn=parallelize_deepseek,
        pipelining_fn=pipeline_deepseek,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_tiktoken_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
)
