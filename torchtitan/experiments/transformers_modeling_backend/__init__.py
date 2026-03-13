# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.hf_datasets.text_datasets import build_text_dataloader
from torchtitan.protocols.train_spec import TrainSpec

from .infra.parallelize import parallelize_hf_transformers

from .infra.pipeline import pipeline_hf_transformers
from .model.args import HFTransformerModelArgs, TitanDenseModelArgs
from .model.model import HFTransformerModel

__all__ = [
    "HFTransformerModelArgs",
    "HFTransformerModel",
]


flavors = {
    "debugmodel": HFTransformerModelArgs(
        titan_dense_args=TitanDenseModelArgs(
            dim=256,
            n_layers=2,
            n_heads=16,
            n_kv_heads=16,
        ),
    ),
    "full": HFTransformerModelArgs(
        titan_dense_args=TitanDenseModelArgs(),
    ),
}


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=HFTransformerModel,
        model_args=flavors,
        parallelize_fn=parallelize_hf_transformers,
        pipelining_fn=pipeline_hf_transformers,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
