# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from typing import Any, cast, TypeAlias

import torch
import torch.nn as nn
from torch.distributed.pipelining.schedules import _PipelineSchedule
from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.collators import TrainerBatch
from torchtitan.components.data.loader import BaseDataLoader
from torchtitan.components.loss import LossFunction
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.config import Configurable, ParallelismConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols.model import BaseModel
from torchtitan.tools import utils

ValidationContext: TypeAlias = Callable[[], AbstractContextManager[None]]


class BaseValidator(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        freq: int = 10
        """Frequency of validation"""

        def __post_init__(self) -> None:
            if self.freq <= 0:
                raise ValueError(
                    f"validation frequency must be positive, got {self.freq}"
                )

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
        Number of validation steps. -1 consumes the finite dataset and therefore
        requires an effective data-parallel degree of one.
        """

        dataloader: BaseDataLoader.Config = field(
            default_factory=lambda: GrainDataLoader.Config(
                dataset=ConcatThenSplitPackingConfig(
                    dataset=DATASETS["c4_validation"],
                ),
                repeat=False,
            )
        )
        """DataLoader configuration for validation"""

        def __post_init__(self):
            BaseValidator.Config.__post_init__(self)
            assert (
                self.steps > 0 or self.steps == -1
            ), "validation steps must be positive or -1"

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
        metrics_processor: MetricsProcessor,
        seq_len: int,
        num_tokens_per_batch: int,
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
        # A bounded validation run repeats data; steps=-1 consumes one finite pass.
        self.dl_config = replace(config.dataloader, repeat=config.steps != -1)
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.seq_len = seq_len
        self.num_tokens_per_batch = num_tokens_per_batch
        self.validation_context = validation_context
        self.metrics_processor = metrics_processor
        self.pp_schedule = pp_schedule
        self.pp_has_first_stage = pp_has_first_stage
        self.pp_has_last_stage = pp_has_last_stage

    @sl.log_trace_span("eval")
    @torch.no_grad()
    def validate(
        self,
        model_parts: list[nn.Module],
        step: int,
    ) -> None:
        sl.add_step_tag("eval")
        # Set model to eval mode
        for model in model_parts:
            model.eval()

        parallel_dims = self.parallel_dims

        accumulated_loss: torch.Tensor | None = None
        device_type = utils.device_type
        total_global_valid_tokens = torch.zeros(
            (), dtype=torch.int64, device=device_type
        )
        num_steps = 0
        num_pp_microbatches = (
            self.parallelism.num_pp_microbatches if parallel_dims.pp_enabled else 1
        )

        validation_dataloader = self.dl_config.build(
            dp_world_size=self.dp_world_size,
            dp_rank=self.dp_rank,
            tokenizer=self.tokenizer,
            max_context_length=self.seq_len,
            num_tokens_per_batch=self.num_tokens_per_batch,
        )

        validation_iterator = iter(iterate_and_close_dataloader(validation_dataloader))
        while True:
            # pyrefly: ignore [missing-attribute, unsupported-operation]
            if self.config.steps != -1 and num_steps >= self.config.steps:
                break

            try:
                microbatches = []
                local_valid_tokens = 0
                for _ in range(num_pp_microbatches):
                    input_dict, labels = next(validation_iterator)
                    # Popped so the batch reaching the model holds only its kwargs.
                    local_valid_tokens += input_dict.pop("num_valid_tokens")
                    self.metrics_processor.ntokens_since_last_log += labels.numel()
                    for k, v in input_dict.items():
                        input_dict[k] = v.to(device_type)
                    labels = labels.to(device_type)
                    microbatches.append((input_dict, labels))
            except StopIteration:
                break

            # All-reduce token count across DP ranks while keeping it on device.
            local_valid_tokens_tensor = torch.tensor(
                local_valid_tokens, dtype=torch.int64, device=device_type
            )
            if parallel_dims.dp_enabled:
                batch_mesh = parallel_dims.get_mesh("batch")
                global_valid_tokens = dist_utils.dist_sum_tensor(
                    local_valid_tokens_tensor, batch_mesh, None
                )
            else:
                global_valid_tokens = local_valid_tokens_tensor

            if parallel_dims.pp_enabled:
                assert self.pp_schedule is not None
                assert self.pp_has_first_stage is not None
                assert self.pp_has_last_stage is not None

                arg_mbs: list[tuple[torch.Tensor, ...]] = []
                kwarg_mbs: list[dict[str, Any]] = []
                target_mbs: list[torch.Tensor] | None = (
                    [] if self.pp_has_last_stage else None
                )

                for input_dict, labels in microbatches:
                    inputs, labels, extra_kwargs = cast(
                        BaseModel, model_parts[0]
                    ).preprocess_inputs(
                        {**input_dict, "labels": labels},
                        parallel_dims=self.parallel_dims,
                        parallelism=self.parallelism,
                    )
                    if self.pp_has_first_stage:
                        arg_mbs.append((inputs,))
                    kwarg_mbs.append(extra_kwargs)
                    if target_mbs is not None:
                        target_mbs.append(labels)

                with self.validation_context():
                    losses = [] if self.pp_has_last_stage else None
                    self.pp_schedule.eval(
                        arg_mbs=arg_mbs if self.pp_has_first_stage else None,
                        kwarg_mbs=kwarg_mbs,
                        target_mbs=target_mbs,
                        losses=losses,
                    )

                # accumulate losses across pipeline microbatches
                # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
                if self.pp_has_last_stage:
                    assert losses is not None
                    # using sum because loss_fn already uses reduction='sum'
                    loss_sum = torch.sum(torch.stack(losses)).to(device_type)
                else:
                    loss_sum = torch.tensor([-1.0], device=device_type)
            else:
                assert len(microbatches) == 1
                input_dict, labels = microbatches[0]
                inputs, labels, extra_kwargs = cast(
                    BaseModel, model_parts[0]
                ).preprocess_inputs(
                    {**input_dict, "labels": labels},
                    parallel_dims=self.parallel_dims,
                    parallelism=self.parallelism,
                )
                with self.validation_context():
                    assert len(model_parts) == 1
                    predictions = model_parts[0](inputs, **extra_kwargs)
                    loss_sum, _ = self.loss_fn(predictions, labels)

            loss_sum = loss_sum.detach()
            if accumulated_loss is None:
                accumulated_loss = loss_sum.clone()
            else:
                accumulated_loss.add_(loss_sum)
            total_global_valid_tokens.add_(global_valid_tokens)
            num_steps += 1

        assert accumulated_loss is not None
        num_global_valid_tokens = int(total_global_valid_tokens.item())
        if parallel_dims.dp_cp_enabled:
            global_loss_sum = dist_utils.dist_sum(
                accumulated_loss, parallel_dims.get_optional_mesh("loss")
            )
        else:
            global_loss_sum = float(accumulated_loss.item())
        global_avg_loss = global_loss_sum / num_global_valid_tokens

        self.metrics_processor.log_validation(loss=global_avg_loss, step=step)

        # Set model back to train mode
        for model in model_parts:
            model.train()


def iterate_and_close_dataloader(
    dataloader: BaseDataLoader,
) -> Iterator[TrainerBatch]:
    """Close a temporary dataloader when its consumer stops iterating."""
    try:
        yield from dataloader
    finally:
        dataloader.close()
