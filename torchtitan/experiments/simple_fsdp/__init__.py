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
from torchtitan.datasets.tokenizer.tiktoken import build_tiktoken_tokenizer
from torchtitan.models.llama3 import llama3_configs, pipeline_llama
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from .model import SimpleFSDPTransformer
from .parallelize_llama import parallelize_llama

register_train_spec(
    TrainSpec(
        name="llama3_simple_fsdp",
        cls=SimpleFSDPTransformer,
        config=llama3_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_tiktoken_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
)
