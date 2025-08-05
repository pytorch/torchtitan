# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.ft.diloco import FaultTolerantTrainSpec, fragment_llm
from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.protocols.train_spec import register_train_spec
from ..llama3 import (
    llama3_configs,
    Llama3StateDictAdapter,
    parallelize_llama,
    pipeline_llama,
    Transformer,
    TransformerModelArgs,
)

__all__ = [
    "parallelize_llama",
    "pipeline_llama",
    "TransformerModelArgs",
    "Transformer",
    "llama3_configs",
]


register_train_spec(
    FaultTolerantTrainSpec(
        name="llama3_ft",
        model_cls=Transformer,
        model_args=llama3_configs,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llama,
        fragment_fn=fragment_llm,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Llama3StateDictAdapter,
    )
)
