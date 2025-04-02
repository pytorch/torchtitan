# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from mm_dataset import build_mm_dataloader

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.datasets.tokenizer.tiktoken import build_tiktoken_tokenizer
from torchtitan.models.llama3 import parallelize_llama, pipeline_llama
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from .model import ModelArgs, MultimodalDecoder, VisionEncoder

__all__ = ["VisionEncoder", "ModelArgs", "MultimodalDecoder"]

llama4_mm_configs = {
    # TODO: add configs for llama4 multimodal
}

register_train_spec(
    TrainSpec(
        name="llama4_multimodal",
        cls=MultimodalDecoder,
        config=llama4_mm_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_mm_dataloader,
        build_tokenizer_fn=build_tiktoken_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
)
