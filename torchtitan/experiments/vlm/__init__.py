# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import asdict, replace

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.models.llama3 import llama3_configs
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from .datasets.mm_datasets import build_mm_dataloader
from .infra.parallelize import parallelize_vlm
from .model.args import Llama3Siglip2ModelArgs, Siglip2ModelArgs
from .model.model import Llama3Siglip2Transformer

__all__ = [
    "parallelize_vlm",
    "Llama3Siglip2ModelArgs",
    "Llama3Siglip2Transformer",
    "llama3_siglip2_configs",
]


llama3_siglip2_configs = {
    "debugmodel": Llama3Siglip2ModelArgs(
        **asdict(replace(llama3_configs["debugmodel"], vocab_size=2048)),
        encoder=Siglip2ModelArgs(
            dim=128,
            ffn_dim=256,
            n_layers=4,
            n_heads=2,
        ),
    ),
}


register_train_spec(
    TrainSpec(
        name="llama3-siglip2",
        model_cls=Llama3Siglip2Transformer,
        model_args=llama3_siglip2_configs,
        parallelize_fn=parallelize_vlm,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_mm_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
    )
)
