# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.distributed.pipeline_parallel import pipeline_llm

from torchtitan.experiments.sft.auto_tokenizer import build_auto_tokenizer

from torchtitan.experiments.sft.sft_text_datasets import (
    build_sft_text_dataloader,
    build_sft_validation_dataloader,
)

from torchtitan.experiments.vlm.infra.loss import build_token_imbalance_ce_loss

from torchtitan.models.llama3 import llama3_args, Transformer
from torchtitan.models.llama3.infra.parallelize import parallelize_llama
from torchtitan.models.llama3.model.state_dict_adapter import Llama3StateDictAdapter
from torchtitan.protocols.train_spec import TrainSpec


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=Transformer,
        model_args=llama3_args,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_sft_text_dataloader,
        build_tokenizer_fn=build_auto_tokenizer,
        # build_cross_entropy_loss will averaged on pad tokens unexpectedly
        build_loss_fn=build_token_imbalance_ce_loss,
        build_validator_fn=build_sft_validation_dataloader,
        state_dict_adapter=Llama3StateDictAdapter,
    )
