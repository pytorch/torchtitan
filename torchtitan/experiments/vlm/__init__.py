# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields
from typing import Any

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.models.llama3 import llama3_args
from torchtitan.protocols.train_spec import TrainSpec

from .datasets.mm_datasets import build_mm_dataloader
from .infra.parallelize import parallelize_vlm
from .model.args import Llama3Siglip2ModelArgs, Siglip2ModelArgs
from .model.model import Llama3Siglip2Transformer

__all__ = [
    "parallelize_vlm",
    "Llama3Siglip2ModelArgs",
    "Llama3Siglip2Transformer",
    "llama3_siglip2_args",
]


def _get_dict(obj) -> dict[str, Any]:
    """Convert dataclass to dict, preserving nested dataclasses (unlike asdict)."""
    return {field.name: getattr(obj, field.name) for field in fields(obj)}


llama3_siglip2_args = {
    "debugmodel": Llama3Siglip2ModelArgs(
        **_get_dict(llama3_args["debugmodel_flex_attn"]),
        encoder=Siglip2ModelArgs(
            dim=128,
            ffn_dim=256,
            n_layers=4,
            n_heads=2,
        ),
    ),
}


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=Llama3Siglip2Transformer,
        model_args=llama3_siglip2_args,
        parallelize_fn=parallelize_vlm,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_mm_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
    )
