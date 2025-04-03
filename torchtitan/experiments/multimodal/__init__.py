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

from torchtitan.models.llama import parallelize_llama, pipeline_llama

from torchtitan.models.llama_multimodal import llama3_2_configs, MultimodalDecoder
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

register_train_spec(
    TrainSpec(
        name="llama3",
        cls=MultimodalDecoder,  # TODO(tj.solergibert) Create VisionEncoder + MultimodalDecoder class?
        config=llama3_2_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_mm_dataloader,
        build_tokenizer_fn=build_tiktoken_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
)
