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
from torchtitan.hf_datasets.text_datasets import build_text_dataloader
from torchtitan.experiments.lfm2 import lfm2_args
from torchtitan.protocols.train_spec import TrainSpec

from .model import SimpleFSDPLFM2Model
from .parallelize import parallelize_lfm2


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=SimpleFSDPLFM2Model,
        model_args=lfm2_args,
        parallelize_fn=parallelize_lfm2,
        pipelining_fn=None,  # Pipeline parallelism not supported yet for LFM2
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
    )
