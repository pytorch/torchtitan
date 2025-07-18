# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Generator

import torch
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule
from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.loss import LossFunction
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config_manager import JobConfig
from torchtitan.datasets.hf_datasets import build_hf_validation_dataloader
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.tools import utils


class BaseValidator:
    def __init__(self, job_config: JobConfig):
        self.job_config = job_config

    def validate(self, model_parts: list[nn.Module]) -> dict[str, float]:
        raise NotImplementedError("validate method not implemented")

    def should_validate(self, step: int) -> bool:
        return step % self.job_config.validation.freq == 0


class Validator(BaseValidator):
    """
    Simple validator focused on correctness and integration.

    Args:
        job_config: Job configuration
        validation_dataloader: The validation dataloader
        loss_fn: Loss function to use for validation
        model: The model to validate (single model, no parallelism)
    """

    validation_dataloader: BaseDataLoader

    def __init__(
        self,
        job_config: JobConfig,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        parallel_dims: ParallelDims,
        loss_fn: LossFunction,
        validation_context: Generator[None, None, None],
        maybe_enable_amp: Generator[None, None, None],
        metrics_processor: MetricsProcessor,
    ):
        self.job_config = job_config
        self.parallel_dims = parallel_dims
        self.loss_fn = loss_fn
        self.validation_dataloader = build_hf_validation_dataloader(
            job_config=job_config,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            tokenizer=tokenizer,
        )
        self.validation_context = validation_context
        self.maybe_enable_amp = maybe_enable_amp
        self.metrics_processor = metrics_processor

    @torch.no_grad()
    def validate(
        self,
        model_parts: list[nn.Module],
        step: int,
    ) -> dict[str, float]:
        # Set model to eval mode
        # TODO: currently does not support pipeline parallelism
        model = model_parts[0]
        model.eval()

        parallel_dims = self.parallel_dims

        accumulated_losses = []
        device_type = utils.device_type
        num_steps = 0

        for input_dict, labels in self.validation_dataloader:
            if (
                self.job_config.validation.steps != -1
                and num_steps >= self.job_config.validation.steps
            ):
                break

            self.metrics_processor.ntokens_since_last_log += labels.numel()
            for k, v in input_dict.items():
                input_dict[k] = v.to(device_type)
            inputs = input_dict["input"]
            labels = labels.to(device_type)

            optional_context_parallel_ctx = (
                dist_utils.create_context_parallel_ctx(
                    cp_mesh=parallel_dims.world_mesh["cp"],
                    cp_buffers=[inputs, labels] + [m.freqs_cis for m in model_parts],
                    cp_seq_dims=[1, 1] + [0 for _ in model_parts],
                    cp_no_restore_buffers={inputs, labels},
                    cp_rotate_method=self.job_config.parallelism.context_parallel_rotate_method,
                )
                if parallel_dims.cp_enabled
                else None
            )

            with self.validation_context(optional_context_parallel_ctx):
                assert len(model_parts) == 1
                with self.maybe_enable_amp:
                    predictions = model(inputs)
                    loss = self.loss_fn(predictions, labels)

            accumulated_losses.append(loss.detach())

            num_steps += 1

        # Compute average loss
        loss = torch.sum(torch.stack(accumulated_losses))
        loss /= num_steps
        if parallel_dims.dp_cp_enabled:
            global_avg_loss = dist_utils.dist_mean(
                loss, parallel_dims.world_mesh["dp_cp"]
            )
        else:
            global_avg_loss = loss.item()

        self.metrics_processor.log_validation(loss=global_avg_loss, step=step)

        # Reshard after run forward pass
        # This is to ensure the model weights are sharded the same way for checkpoint saving.
        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.reshard()

        # Set model back to train mode
        model.train()


def build_validator(
    job_config: JobConfig,
    dp_world_size: int,
    dp_rank: int,
    tokenizer: BaseTokenizer,
    parallel_dims: ParallelDims,
    loss_fn: LossFunction,
    validation_context: Generator[None, None, None],
    maybe_enable_amp: Generator[None, None, None],
    metrics_processor: MetricsProcessor | None = None,
) -> BaseValidator:
    """Build a simple validator focused on correctness."""
    return Validator(
        job_config=job_config,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
        parallel_dims=parallel_dims,
        loss_fn=loss_fn,
        validation_context=validation_context,
        maybe_enable_amp=maybe_enable_amp,
        metrics_processor=metrics_processor,
    )
