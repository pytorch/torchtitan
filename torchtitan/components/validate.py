# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule

from torch.distributed.tensor import DTensor
from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.loss import LossFunction
from torchtitan.components.tokenizer import Tokenizer
from torchtitan.config_manager import JobConfig
from torchtitan.datasets.hf_datasets import build_hf_validation_dataloader
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.tools import utils
from torchtitan.tools.logging import logger


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
        tokenizer: Tokenizer,
        parallel_dims: ParallelDims,
        world_mesh: torch.distributed.DeviceMesh,
        loss_fn: LossFunction,
    ):
        self.job_config = job_config
        self.parallel_dims = parallel_dims
        self.world_mesh = world_mesh
        self.loss_fn = loss_fn
        self.validation_dataloader = build_hf_validation_dataloader(
            job_config=job_config,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            tokenizer=tokenizer,
        )

    @torch.no_grad()
    def validate(
        self,
        model_parts: list[nn.Module],
    ) -> dict[str, float]:
        # Set model to eval mode
        # TODO: currently does not support pipeline parallelism
        model = model_parts[0]
        model.eval()

        accumulated_losses = []
        device_type = utils.device_type
        num_val_steps = 0

        for input_dict, labels in self.validation_dataloader:
            if (
                self.job_config.validation.steps != -1
                and num_val_steps >= self.job_config.validation.steps
            ):
                break

            for k, v in input_dict.items():
                input_dict[k] = v.to(device_type)
            labels = labels.to(device_type)

            inputs = input_dict["input"]
            predictions = model(inputs)

            if self.parallel_dims.loss_parallel_enabled:
                if isinstance(predictions, torch.Tensor) and not isinstance(
                    predictions, DTensor
                ):
                    predictions = DTensor.from_local(predictions, self.world_mesh["tp"])
                if isinstance(labels, torch.Tensor) and not isinstance(labels, DTensor):
                    labels = DTensor.from_local(labels, self.world_mesh["tp"])
            loss = self.loss_fn(predictions, labels)

            accumulated_losses.append(loss.detach())

            num_val_steps += 1

        # Compute average loss
        loss = torch.sum(torch.stack(accumulated_losses))
        if self.parallel_dims.dp_cp_enabled:
            global_avg_loss = dist_utils.dist_mean(loss, self.world_mesh["dp_cp"])
        else:
            global_avg_loss = loss

        logger.info(
            f"Validation completed. Average loss: {global_avg_loss:.4f} over {num_val_steps} batches"
        )

        # Reshard after run forward pass
        # This is to ensure the model weights are sharded the same way for checkpoint saving.
        for module in model.modules():
            if isinstance(module, FSDPModule):
                module.reshard()

        # Set model back to train mode
        model.train()

        return {"validation_loss": global_avg_loss}


def build_validator(
    job_config: JobConfig,
    dp_world_size: int,
    dp_rank: int,
    tokenizer: Tokenizer,
    parallel_dims: ParallelDims,
    world_mesh: torch.distributed.DeviceMesh,
    loss_fn: LossFunction,
) -> BaseValidator:
    """Build a simple validator focused on correctness."""
    return Validator(
        job_config=job_config,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
        parallel_dims=parallel_dims,
        world_mesh=world_mesh,
        loss_fn=loss_fn,
    )
