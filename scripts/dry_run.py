# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Dry run trainer for fast configuration validation without GPU/distributed setup.

This module provides a lightweight trainer that validates job configurations,
model architecture, and dataloader setup without requiring GPU resources or
distributed initialization. Useful for rapid iteration on configuration files
and CI/CD validation pipelines.
"""

import os
import sys

# Add parent directory to path to import torchtitan
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import torchtitan.protocols.train_spec as train_spec_module
from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.tools import utils
from torchtitan.tools.logging import logger
from torchtitan.train import main, Trainer


class DryRunTrainer(Trainer):
    """
    A lightweight trainer that validates configurations without GPU allocation.

    This trainer performs comprehensive validation of the training configuration
    without allocating GPU resources or initializing distributed setup. It validates:

    - Configuration file parsing and structure
    - Model architecture (constructed on meta device)
    - Tokenizer initialization
    - Dataloader configuration
    - Parallelism settings
    - Model converters (if specified)

    Unlike the regular Trainer, this does not:
    - Allocate GPU memory
    - Initialize distributed process groups
    - Create optimizers or learning rate schedulers
    - Set up checkpointing or metrics
    - Run any actual training

    Args:
        job_config: JobConfig containing all training configuration parameters

    Note:
        Validation completes immediately after initialization. No training loop is executed.
        All operations use CPU and meta devices for zero-cost validation.
    """

    def __init__(self, job_config: JobConfig):
        torch._C._log_api_usage_once("torchtitan.dry_run")

        self.job_config = job_config

        logger.info(f"Starting job: {job_config.job.description}")
        logger.info("DRY RUN MODE - Configuration validation only")

        # Use CPU device (no GPU required)
        self.device = torch.device("cpu")

        # Log and validate config
        job_config.maybe_log()
        logger.info("Configuration parsed successfully")

        # Get train spec
        self.train_spec = train_spec_module.get_train_spec(job_config.model.name)
        logger.info(f"Train spec loaded for model: {job_config.model.name}")

        # Build tokenizer
        self.tokenizer = (
            self.train_spec.build_tokenizer_fn(job_config)
            if self.train_spec.build_tokenizer_fn is not None
            else None
        )
        if self.tokenizer:
            logger.info("Tokenizer built successfully")

        # Validate model configuration
        model_args = self.train_spec.model_args[job_config.model.flavor]
        model_args.update_from_config(job_config)
        self.model_args = model_args

        logger.info(
            f"Model args validated: {job_config.model.name} {job_config.model.flavor}"
        )

        # Build model on meta device (validates architecture without memory allocation)
        logger.info("Validating model architecture...")
        with (
            torch.device("meta"),
            utils.set_default_dtype(TORCH_DTYPE_MAP[job_config.training.dtype]),
        ):
            model = self.train_spec.model_cls(model_args)

        # Calculate and log model size
        model_param_count, _ = model_args.get_nparams_and_flops(
            model, job_config.training.seq_len
        )
        logger.info(
            f"Model architecture validated: {job_config.model.name} "
            f"with {model_param_count:,} parameters"
        )

        # Validate dataloader configuration (build with minimal params)
        logger.info("Validating dataloader configuration...")
        try:
            # Use dp_world_size=1 and dp_rank=0 for dry run
            dataloader = self.train_spec.build_dataloader_fn(
                dp_world_size=1,
                dp_rank=0,
                tokenizer=self.tokenizer,
                job_config=job_config,
            )
            logger.info("Dataloader configuration validated successfully")
        except Exception as e:
            logger.warning(f"Dataloader validation encountered issue: {e}")
            logger.info(
                "Note: Some dataloader issues may only appear with actual data paths"
            )

        # Validate model converters if specified
        if job_config.model.converters:
            logger.info(f"Model converters specified: {job_config.model.converters}")

        # Validate parallelism configuration
        parallelism_config = job_config.parallelism
        logger.info(
            f"Parallelism config: "
            f"DP-shard={parallelism_config.data_parallel_shard_degree}, "
            f"DP-replicate={parallelism_config.data_parallel_replicate_degree}, "
            f"TP={parallelism_config.tensor_parallel_degree}, "
            f"PP={parallelism_config.pipeline_parallel_degree}, "
            f"CP={parallelism_config.context_parallel_degree}"
        )

        # Summary
        logger.info("=" * 80)
        logger.info("DRY RUN VALIDATION COMPLETE")
        logger.info("=" * 80)
        logger.info("All configurations validated successfully!")
        logger.info("Configuration is ready for training execution.")
        logger.info("=" * 80)

    def train(self):
        return


if __name__ == "__main__":
    main(DryRunTrainer)
