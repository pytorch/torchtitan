# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from typing import Any, cast, TypeAlias

import torch
import torch.nn as nn
from torch.distributed.pipelining.schedules import _PipelineSchedule
from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.loss import IGNORE_INDEX, LossFunction
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import Configurable, ParallelismConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.distributed.context_parallel import prepare_context_parallel_input
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.protocols import BaseModel
from torchtitan.tools import utils
from torchtitan.tools.logging import logger

ValidationContext: TypeAlias = Callable[[], AbstractContextManager[None]]


class BaseValidator(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        freq: int = 10
        """Frequency of validation"""

    def __init__(
        self,
        config: Config,
        **kwargs,
    ):
        self.config = config

    def validate(self, model_parts: list[nn.Module], step: int) -> None:
        raise NotImplementedError("validate method not implemented")

    def should_validate(self, step: int) -> bool:
        return step == 1 or step % self.config.freq == 0


class Validator(BaseValidator):
    """
    Simple validator focused on correctness and integration.

    Args:
        config: Validator.Config configuration
        parallelism: ParallelismConfig configuration
        dp_world_size: Data parallel world size
        dp_rank: Data parallel rank
        tokenizer: Tokenizer
        parallel_dims: Parallel dimensions
        loss_fn: Loss function to use for validation
        validation_context: Context manager for validation
        maybe_enable_amp: Context manager for AMP
        metrics_processor: Metrics processor
        pp_schedule: Pipeline schedule (optional)
        pp_has_first_stage: Whether this rank has the first PP stage (optional)
        pp_has_last_stage: Whether this rank has the last PP stage (optional)
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseValidator.Config):
        enable: bool = False
        """Enable validation to default run validation after each training loop"""

        steps: int = -1
        """
        Number of steps to take in the validation set, -1 means consuming
        all the data in the validation dataset.
        WARNING: When setting to -1 there could be hangs due to mismatch among ranks
        """

        dataloader: BaseDataLoader.Config = field(
            default_factory=lambda: HuggingFaceTextDataLoader.Config(
                dataset="c4_validation",
                infinite=False,
            )
        )
        """DataLoader configuration for validation"""

        def __post_init__(self):
            assert (
                self.steps > 0 or self.steps == -1
            ), "validation steps must be positive or -1"

    validation_dataloader: BaseDataLoader

    # TODO: improve the constructor signature
    def __init__(
        self,
        config: Config,
        *,
        parallelism: ParallelismConfig,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        parallel_dims: ParallelDims,
        loss_fn: LossFunction,
        validation_context: ValidationContext,
        maybe_enable_amp: AbstractContextManager[None],
        metrics_processor: MetricsProcessor,
        seq_len: int,
        local_batch_size: int,
        pp_schedule: _PipelineSchedule | None = None,
        pp_has_first_stage: bool | None = None,
        pp_has_last_stage: bool | None = None,
        **kwargs,
    ):
        super().__init__(config=config)
        self.parallelism = parallelism
        self.tokenizer = tokenizer
        self.parallel_dims = parallel_dims
        self.loss_fn = loss_fn
        # pyrefly: ignore [unexpected-keyword]
        dl_config = replace(config.dataloader, infinite=config.steps != -1)
        self.validation_dataloader = dl_config.build(
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            tokenizer=tokenizer,
            seq_len=seq_len,
            local_batch_size=local_batch_size,
        )
        self.validation_context = validation_context
        self.maybe_enable_amp = maybe_enable_amp
        self.metrics_processor = metrics_processor
        self.pp_schedule = pp_schedule
        self.pp_has_first_stage = pp_has_first_stage
        self.pp_has_last_stage = pp_has_last_stage

        if config.steps == -1:
            logger.warning(
                "Setting validation steps to -1 might cause hangs because of "
                "unequal sample counts across ranks when dataset is exhausted."
            )

    def post_dataloading_process(
        self,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        model_parts: list[nn.Module],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        """
        Post-processing hook after data loading and before model forward pass.

        This method processes the raw data from the dataloader and prepares it for
        the model's forward pass. It separates the main input tensor from auxiliary
        inputs and constructs additional keyword arguments (e.g., attention masks).

        Args:
            input_dict: Dictionary containing tensors from the dataloader. Must
                contain an "input" key with the main input tensor. May contain
                additional keys for auxiliary inputs (e.g., position ids).
            labels: Target labels for the batch.
            model_parts: List of model parts for accessing model methods.

        Returns:
            A tuple of (inputs, labels, extra_inputs, extra_kwargs) where:
                - inputs: Main input tensor extracted from input_dict["input"].
                - labels: Target labels (potentially modified by CP sharding).
                - extra_inputs: Dict of auxiliary input tensors (all keys except
                    "input" from input_dict). These are passed to the model forward
                    but are NOT forwarded across pipeline parallel stages.
                - extra_kwargs: Dict of additional keyword arguments for model forward.
                    These ARE forwarded across pipeline parallel stages. Contains
                    attention_masks if flex attention is enabled.

        Note:
            The distinction between extra_inputs and extra_kwargs is important for
            pipeline parallelism: extra_kwargs are forwarded to all pipeline stages,
            while extra_inputs are only available to the first stage.
        """
        inputs = input_dict["input"]
        extra_inputs = {k: v for k, v in input_dict.items() if k != "input"}
        # For arguments, like attention_masks, we have to put them in a separate
        # dict as extra_inputs are not forwarded to other stages in PP, but
        # extra_kwargs are.
        extra_kwargs: dict[str, Any] = {}

        try:
            # pyrefly: ignore [not-callable]
            extra_kwargs["attention_masks"] = cast(
                BaseModel, model_parts[0]
            ).get_attention_masks(
                input_batch=inputs,
                tokenizer=self.tokenizer,
                extra_inputs=extra_inputs,
            )
        except TypeError:
            pass

        if self.parallel_dims.cp_enabled:
            inputs, labels, extra_kwargs = prepare_context_parallel_input(
                inputs,
                labels,
                extra_kwargs,
                self.parallel_dims.get_mesh("cp"),
                inputs.device,
                self.parallelism.context_parallel_load_balancer,
            )

        return inputs, labels, extra_inputs, extra_kwargs

    @torch.no_grad()
    def validate(
        self,
        model_parts: list[nn.Module],
        step: int,
    ) -> None:
        # Set model to eval mode
        for model in model_parts:
            model.eval()

        parallel_dims = self.parallel_dims

        accumulated_losses = []
        device_type = utils.device_type
        num_steps = 0

        for input_dict, labels in self.validation_dataloader:
            # pyrefly: ignore [missing-attribute, unsupported-operation]
            if self.config.steps != -1 and num_steps >= self.config.steps:
                break

            self.metrics_processor.ntokens_since_last_log += labels.numel()
            for k, v in input_dict.items():
                input_dict[k] = v.to(device_type)
            labels = labels.to(device_type)

            # Process data (extract inputs, handle attention masks, CP sharding)
            inputs, labels, extra_inputs, extra_kwargs = self.post_dataloading_process(
                input_dict, labels, model_parts
            )

            # Count valid tokens for this batch
            local_valid_tokens = torch.tensor(0, dtype=torch.int64, device=device_type)
            local_valid_tokens += (labels != IGNORE_INDEX).sum()

            # All-reduce token count across DP ranks to get global token count
            if parallel_dims.dp_enabled:
                batch_mesh = parallel_dims.get_mesh("batch")
                global_valid_tokens = dist_utils.dist_sum(
                    local_valid_tokens, batch_mesh, None
                )
            else:
                global_valid_tokens = local_valid_tokens.float()

            if parallel_dims.pp_enabled:
                assert self.pp_schedule is not None
                assert self.pp_has_first_stage is not None
                assert self.pp_has_last_stage is not None
                # Pipeline Parallel forward inside eval() call
                with self.validation_context():
                    targets, losses = (
                        (labels, []) if self.pp_has_last_stage else (None, None)
                    )
                    if self.pp_has_first_stage:
                        self.pp_schedule.eval(
                            inputs,
                            **extra_inputs,
                            **extra_kwargs,
                            target=targets,
                            losses=losses,
                        )
                    else:
                        self.pp_schedule.eval(
                            **extra_kwargs,
                            target=targets,
                            losses=losses,
                        )

                # accumulate losses across pipeline microbatches
                # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
                loss_sum = (
                    # using sum because loss_fn already uses reduction='sum'
                    torch.sum(torch.stack(losses)).to(device_type)
                    if self.pp_has_last_stage
                    else torch.tensor([-1.0], device=device_type)
                )
            else:
                with self.validation_context():
                    assert len(model_parts) == 1
                    with self.maybe_enable_amp:
                        predictions = model_parts[0](
                            inputs, **extra_inputs, **extra_kwargs
                        )
                        loss_sum = self.loss_fn(predictions, labels)

            accumulated_losses.append(loss_sum.detach() / global_valid_tokens)
            num_steps += 1

        # Compute average loss
        loss = torch.sum(torch.stack(accumulated_losses))
        loss /= num_steps
        if parallel_dims.dp_cp_enabled:
            global_avg_loss = dist_utils.dist_sum(
                loss, parallel_dims.get_optional_mesh("loss")
            )
        else:
            global_avg_loss = loss.item()

        self.metrics_processor.log_validation(loss=global_avg_loss, step=step)

        # Set model back to train mode
        for model in model_parts:
            model.train()
