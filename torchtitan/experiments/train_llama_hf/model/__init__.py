# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from loss import cross_entropy_loss_hf
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from torchtitan.components.optimizer import build_lr_schedulers, build_optimizers
from torchtitan.experiments.train_llama_hf.dataset import (
    build_pos_included_hf_dataloader,
)
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from .parallelize_llama import parallelize_llama
from .pipeline_llama import pipeline_llama

register_train_spec(
    TrainSpec(
        name="llama3_hf",
        cls=AutoModelForCausalLM,
        config=AutoConfig,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_pos_included_hf_dataloader,
        tokenizer_cls=AutoTokenizer,
        loss_fn=cross_entropy_loss_hf,
    )
)
