# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from torchtitan.components.ft.diloco import FaultTolerantTrainSpec, fragment_llm
from torchtitan.components.ft.manager import _FT_MANAGER, FTManager
from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers, OptimizersContainer
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.config import Optimizer as OptimizerConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.hf_datasets.text_datasets import build_text_dataloader
from torchtitan.protocols.train_spec import TrainSpec

from ..llama3 import llama3_args, Llama3StateDictAdapter, parallelize_llama, Transformer


def build_ft_optimizers(
    model_parts: list[nn.Module],
    optimizer_config: OptimizerConfig,
    parallel_dims: ParallelDims,
    ft_manager: FTManager | None = None,
) -> OptimizersContainer:
    return build_optimizers(
        model_parts=model_parts,
        optimizer_config=optimizer_config,
        parallel_dims=parallel_dims,
        ft_manager=_FT_MANAGER,
    )


def get_train_spec() -> TrainSpec:
    return FaultTolerantTrainSpec(
        model_cls=Transformer,
        model_args=llama3_args,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        build_optimizers_fn=build_ft_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=Llama3StateDictAdapter,
        fragment_fn=fragment_llm,
    )
