# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.distributed.pipeline_parallel import pipeline_llm

from torchtitan.experiments.simple_fsdp.llama3.model import SimpleFSDPTransformer
from torchtitan.hf_datasets.text_datasets import build_text_dataloader
from torchtitan.models.llama3 import llama3_configs
from torchtitan.protocols.train_spec import TrainSpec

from .parallelize import parallelize_llama


def _to_simple_fsdp_configs(base_configs):
    """Convert Transformer.Config instances to SimpleFSDPTransformer.Config."""
    return {
        k: SimpleFSDPTransformer.Config(
            **{f.name: getattr(v, f.name) for f in fields(v)}
        )
        for k, v in base_configs.items()
    }


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_configs=_to_simple_fsdp_configs(llama3_configs),
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
