# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.loss import LossFunction
from torchtitan.components.tokenizer import Tokenizer
from torchtitan.config_manager import JobConfig
from torchtitan.datasets.hf_datasets import build_hf_validation_dataloader
from torchtitan.tools import utils
from torchtitan.tools.logging import logger


class BaseValidator:
    def __init__(self, job_config: JobConfig):
        self.job_config = job_config

    def validate(self, model_parts: list[nn.Module]) -> dict[str, float]:
        raise NotImplementedError("validate method not implemented")


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
        loss_fn: LossFunction,
    ):
        self.job_config = job_config
        self.loss_fn = loss_fn
        self.validation_dataloader = build_hf_validation_dataloader(
            job_config=job_config,
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            tokenizer=tokenizer,
            infinite=False,
        )

    def should_validate(self, step: int) -> bool:
        return step % self.job_config.validation.val_freq == 0

    def validate(
        self,
        model_parts: list[nn.Module],
    ) -> dict[str, float]:
        # Set model to eval mode
        model = model_parts[0]
        model.eval()

        total_loss = 0.0
        num_batches = 0
        device_type = utils.device_type
        num_val_steps = 0

        with torch.no_grad():
            try:
                for input_dict, labels in self.validation_dataloader:

                    if (
                        self.job_config.validation.val_steps != -1
                        and num_val_steps >= self.job_config.validation.val_steps
                    ):
                        break

                    for k, v in input_dict.items():
                        input_dict[k] = v.to(device_type)
                    labels = labels.to(device_type)

                    inputs = input_dict["input"]
                    predictions = model(inputs)
                    loss = self.loss_fn(predictions, labels)

                    total_loss += loss.item()
                    num_batches += 1

                    num_val_steps += 1

            except StopIteration:
                logger.info("Validation dataloader exhausted")

        # Compute average loss
        if num_batches > 0:
            average_loss = total_loss / num_batches
        else:
            average_loss = 0.0
            logger.warning("No validation batches processed")

        logger.info(
            f"Validation completed. Average loss: {average_loss:.4f} over {num_batches} batches"
        )

        # Set model back to train mode
        model.train()

        return {"validation_loss": average_loss}


def build_validator(
    job_config: JobConfig,
    dp_world_size: int,
    dp_rank: int,
    tokenizer: Tokenizer,
    loss_fn: LossFunction,
) -> BaseValidator:
    """Build a simple validator focused on correctness."""
    return Validator(
        job_config=job_config,
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=tokenizer,
        loss_fn=loss_fn,
    )
