# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import Validator
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.hf_datasets.text_datasets import build_text_dataloader

from torchtitan.models.llama3 import llama3_configs
from torchtitan.models.llama3.model.state_dict_adapter import Llama3StateDictAdapter
from torchtitan.protocols.train_spec import TrainSpec

from .parallelize_llama import parallelize_llama


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_configs=llama3_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        validator_cls=Validator,
        state_dict_adapter=Llama3StateDictAdapter,
    )
