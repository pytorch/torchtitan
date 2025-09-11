# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Generator

import torch
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.pipelining.schedules import _PipelineSchedule

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.loss import LossFunction
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.components.validate import Validator
from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils

class BlendCorpusValidator(Validator):
    """
    Simple validator focused on correctness and integration.

    Args:
        job_config: Job configuration
        validation_dataloader: The validation dataloader
        loss_fn: Loss function to use for validation
        model: The model to validate (single model, no parallelism)
    """

    validation_dataloader: BaseDataLoader
    def blendcorpus_init(
        self,
        validation_dataloader: BaseDataLoader,
    ):
        self.validation_dataloader = validation_dataloader

def build_blendcorpus_validator(
    job_config: JobConfig,
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    parallel_dims: ParallelDims,
    loss_fn: LossFunction,
    validation_context: Generator[None, None, None],
    maybe_enable_amp: Generator[None, None, None],
    metrics_processor: MetricsProcessor | None = None,
    pp_schedule: _PipelineSchedule | None = None,
    pp_has_first_stage: bool | None = None,
    pp_has_last_stage: bool | None = None,
) -> BlendCorpusValidator:
    """Build a simple validator focused on correctness."""
    return BlendCorpusValidator(
        job_config=job_config,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
        parallel_dims=parallel_dims,
        loss_fn=loss_fn,
        validation_context=validation_context,
        maybe_enable_amp=maybe_enable_amp,
        metrics_processor=metrics_processor,
        pp_schedule=pp_schedule,
        pp_has_first_stage=pp_has_first_stage,
        pp_has_last_stage=pp_has_last_stage,
    )
