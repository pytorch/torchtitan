# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from collections.abc import Iterable
from datetime import timedelta
from typing import Any

import torch
from torch.distributed.elastic.multiprocessing.errors import record

from torchtitan.components.dataloader import BaseDataLoader, DataloaderExhaustedError
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.components.validate import Validator
from torchtitan.config import ConfigManager
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.context_parallel import prepare_context_parallel_input
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.profiling import (
    maybe_enable_memory_snapshot,
    maybe_enable_profiling,
)
from torchtitan.trainer import Trainer as TitanTrainer

from .engine import ForgeEngine


class Trainer(ForgeEngine):
    tokenizer: HuggingFaceTokenizer | None
    dataloader: BaseDataLoader
    validator: Validator
    metrics_processor: MetricsProcessor

    # additional training states
    step: int

    # Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
    @record
    def __init__(self, config: TitanTrainer.Config):
        if config.debug.print_config:
            logger.info(f"Running with args: {config.to_dict()}")

        # NOTE: Here we are passing in Trainer.Config as a superset of ForgeEngine.Config
        super().__init__(config)

        # build tokenizer
        self.tokenizer = (
            config.tokenizer.build(tokenizer_path=config.hf_assets_path)
            if config.tokenizer is not None
            else None
        )

        # build dataloader
        self.dataloader = config.dataloader.build(
            dp_world_size=self.dp_degree,
            dp_rank=self.dp_rank,
            tokenizer=self.tokenizer,
            seq_len=config.training.seq_len,
            local_batch_size=config.training.local_batch_size,
        )

        model_args = self.model_config
        logger.info(
            f"Built {config.model_spec.name} {config.model_spec.flavor} with {model_args}"
        )

        # metrics logging
        self.metrics_processor = config.metrics.build(
            parallel_dims=self.parallel_dims,
            dump_folder=config.dump_folder,
            pp_schedule=config.parallelism.pipeline_parallel_schedule,
            config_dict=config.to_dict(),
        )
        color = self.metrics_processor.color

        self.metrics_processor.num_flops_per_token = self.num_flops_per_token

        logger.info(
            f"{color.blue}Model {config.model_spec.name} {config.model_spec.flavor} "
            f"{color.red}size: {self.model_param_count:,} total parameters{color.reset}"
        )

        # initialize device memory monitor and get peak flops for MFU calculation
        device_memory_monitor = self.metrics_processor.device_memory_monitor
        gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
        logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
        device_mem_stats = device_memory_monitor.get_peak_stats()
        logger.info(
            f"{utils.device_type.upper()} memory usage for model: "
            f"{device_mem_stats.max_reserved_gib:.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)"
        )

        self.metrics_processor.optimizers = self.optimizers

        # Initialize trainer states that will be saved in checkpoint.
        # These attributes must be initialized before checkpoint loading.
        self.step = 0

        # Build validator if validation is configured
        if config.validator.enable:
            pp_schedule, pp_has_first_stage, pp_has_last_stage = (
                (
                    self.pp_schedule,
                    self.pp_has_first_stage,
                    self.pp_has_last_stage,
                )
                if self.parallel_dims.pp_enabled
                else (None, None, None)
            )

            self.validator = config.validator.build(
                parallelism=config.parallelism,
                dp_world_size=self.dp_degree,
                dp_rank=self.dp_rank,
                tokenizer=self.tokenizer,
                parallel_dims=self.parallel_dims,
                loss_fn=self.loss_fn,
                validation_context=self.train_context,
                maybe_enable_amp=self.maybe_enable_amp,
                metrics_processor=self.metrics_processor,
                seq_len=config.training.seq_len,
                local_batch_size=config.training.local_batch_size,
                pp_schedule=pp_schedule,
                pp_has_first_stage=pp_has_first_stage,
                pp_has_last_stage=pp_has_last_stage,
            )

        logger.info(
            "Trainer is initialized with "
            f"local batch size {config.training.local_batch_size}, "
            f"global batch size {self.global_batch_size}, "
            f"gradient accumulation steps {self.gradient_accumulation_steps}, "
            f"sequence length {config.training.seq_len}, "
            f"total steps {config.training.steps} "
            f"(warmup {config.lr_scheduler.warmup_steps})."
        )

    def batch_generator(
        self, data_iterable: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ) -> Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        """Returns an iterator that processes batches from the data iterator."""
        device_type = utils.device_type
        data_iterator = iter(data_iterable)

        while True:
            try:
                batch = next(data_iterator)
            except StopIteration as ex:
                # If data runs out during gradient accumulation, that
                # entire step will not be executed.
                raise DataloaderExhaustedError() from ex
            data_load_start = time.perf_counter()
            input_dict, labels = batch
            self.metrics_processor.ntokens_since_last_log += labels.numel()
            self.metrics_processor.data_loading_times.append(
                time.perf_counter() - data_load_start
            )

            # Tensors stay on CPU; moved to GPU per-microbatch during training
            yield input_dict, labels

    def post_dataloading_process(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        inputs = input_dict["input"]
        extra_inputs = {k: v for k, v in input_dict.items() if k != "input"}
        # For arguments, like attention_masks, we have to put them in a separate
        # dict as extra_inputs are not forwarded to other stages in PP, but
        # extra_kwargs are.
        extra_kwargs: dict[str, Any] = {}

        try:
            # pyrefly: ignore [not-callable]
            extra_kwargs["attention_masks"] = self.model_parts[0].get_attention_masks(
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
                self.device,
                self.config.parallelism.context_parallel_load_balancer,
            )

        return inputs, labels, extra_inputs, extra_kwargs

    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        model_parts = self.model_parts
        parallel_dims = self.parallel_dims

        inputs, labels, extra_inputs, extra_kwargs = self.post_dataloading_process(
            input_dict, labels
        )

        if parallel_dims.pp_enabled:
            # Pipeline Parallel forward / backward inside step() call
            with self.train_context():
                targets, losses = (
                    (labels, []) if self.pp_has_last_stage else (None, None)
                )
                if self.pp_has_first_stage:
                    self.pp_schedule.step(
                        inputs,
                        **extra_inputs,
                        **extra_kwargs,
                        target=targets,
                        losses=losses,
                    )
                else:
                    self.pp_schedule.step(
                        **extra_kwargs,
                        target=targets,
                        losses=losses,
                    )

            # accumulate losses across pipeline microbatches
            # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
            loss = (
                # Rescale PP loss to be "local loss sum / global valid tokens"
                # because each microbatch could have different number of valid tokens
                (torch.sum(torch.stack(losses)) / global_valid_tokens).to(self.device)
                if self.pp_has_last_stage
                else torch.tensor([-1.0], device=self.device)
            )
        else:
            # Non-PP forward / backward
            with self.train_context():
                assert len(model_parts) == 1
                with self.maybe_enable_amp:
                    pred = model_parts[0](inputs, **extra_inputs, **extra_kwargs)
                    # Compute loss sum (reduction='sum')
                    loss_sum = self.loss_fn(pred, labels)

                    # Scale the loss by the inverse of the total weight denominator before backward
                    # This ensures gradients are properly normalized across all microbatches
                    loss = loss_sum / global_valid_tokens

                # need to free pred before bwd to avoid peaking memory
                del pred
                loss.backward()

        # The returned loss here is local SUM loss / global_valid_tokens
        return loss

    def train_step(
        self, data_iterator: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        self.optimizers.zero_grad()

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        parallel_dims = self.parallel_dims

        # Collect all microbatches on CPU and count total valid tokens
        # Here we assume the inputs/labels are on GPU
        microbatches = []
        local_valid_tokens = torch.tensor(0, dtype=torch.int64)
        for _microbatch in range(self.gradient_accumulation_steps):
            input_dict, labels = next(data_iterator)
            local_valid_tokens += (labels != IGNORE_INDEX).sum()
            microbatches.append((input_dict, labels))

        # All-reduce to get global token count across DP ranks
        # Move to GPU for distributed communication
        local_valid_tokens = local_valid_tokens.to(self.device)
        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            global_valid_tokens = dist_utils.dist_sum(local_valid_tokens, batch_mesh)
        else:
            global_valid_tokens = local_valid_tokens.float()

        # Process each microbatch: move to GPU, forward/backward, then free
        accumulated_losses = []
        for input_dict, labels in microbatches:
            # Move tensors to GPU
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor):
                    input_dict[k] = v.to(self.device)
            labels = labels.to(self.device)

            loss = self.forward_backward_step(
                input_dict=input_dict,
                labels=labels,
                global_valid_tokens=global_valid_tokens,
            )
            accumulated_losses.append(loss.detach())

        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=parallel_dims.get_optional_mesh("pp"),
            ep_enabled=parallel_dims.ep_enabled,
        )
        self.checkpointer.maybe_wait_for_staging()
        self.optimizers.step()
        self.lr_schedulers.step()

        # Reduce the data collected over gradient accumulation steps.
        loss = torch.sum(torch.stack(accumulated_losses))

        # log metrics
        if not self.metrics_processor.should_log(self.step):
            return

        if parallel_dims.dp_cp_enabled:
            loss = loss.detach()
            global_avg_loss, global_max_loss = (
                dist_utils.dist_sum(loss, parallel_dims.get_optional_mesh("loss")),
                dist_utils.dist_max(loss, parallel_dims.get_optional_mesh("loss")),
            )
        else:
            global_avg_loss = global_max_loss = loss.detach().item()

        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            grad_norm.item(),
        )

    @record
    def train(self):
        config = self.config

        self.checkpointer.load(step=config.checkpoint.load_step)
        logger.info(f"Training starts at step {self.step + 1}.")

        with (
            maybe_enable_profiling(
                config.profiling,
                global_step=self.step,
                base_folder=config.dump_folder,
            ) as torch_profiler,
            maybe_enable_memory_snapshot(
                config.profiling,
                global_step=self.step,
                base_folder=config.dump_folder,
            ) as memory_profiler,
        ):
            data_iterator = self.batch_generator(self.dataloader)
            while self.step < config.training.steps:
                self.step += 1
                self.gc_handler.run(self.step)
                try:
                    self.train_step(data_iterator)
                except DataloaderExhaustedError:
                    logger.warning("Ran out of data; last step was canceled.")
                    break

                # Run validation if validator is available
                if config.validator.enable and self.validator.should_validate(
                    self.step
                ):
                    self.validator.validate(self.model_parts, self.step)

                self.checkpointer.save(
                    self.step, last_step=(self.step == config.training.steps)
                )

                # signal the profiler that the next profiling step has started
                if torch_profiler:
                    torch_profiler.step()
                if memory_profiler:
                    memory_profiler.step()

                # reduce timeout after first train step for faster signal
                # (assuming lazy init and compilation are finished)
                if self.step == 1:
                    dist_utils.set_pg_timeouts(
                        timeout=timedelta(seconds=config.comm.train_timeout_seconds),
                        parallel_dims=self.parallel_dims,
                    )

        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)

        logger.info("Training completed")

    def state_dict(self) -> dict[str, Any]:
        return {"step": self.step}

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.step = state_dict["step"]

    def close(self) -> None:
        if self.metrics_processor:
            self.metrics_processor.close()
        super().close()


def main(custom_trainer_class: type[Trainer] | None = None) -> None:
    """Main entry point for training."""
    init_logger()

    import torchtitan

    logger.info(
        "torchtitan version: %s (0.0.0 means __version__ is not defined correctly).",
        torchtitan.__version__,
    )

    config_manager = ConfigManager()
    config = config_manager.parse_args()
    trainer: Trainer | None = None

    try:
        # TODO(local_tensor): Remove this special case once LocalTensor supports
        # init_weights() and foreach_allgather. In local tensor mode, skip
        # training/checkpointing as the # model is not fully initialized
        # pyrefly: ignore [missing-attribute]
        if config.comm.mode == "local_tensor":
            logger.info("Local tensor mode enabled - skipping training execution")
            return

        # pyrefly: ignore [missing-attribute]
        if custom_trainer_class is not None:
            trainer = custom_trainer_class(config)
        else:
            trainer = config.build()

        # pyrefly: ignore [missing-attribute]
        if config.checkpoint.create_seed_checkpoint:
            assert (
                int(os.environ["WORLD_SIZE"]) == 1
            ), "Must create seed checkpoint using a single device, to disable sharding."
            assert (
                # pyrefly: ignore [missing-attribute]
                config.checkpoint.enable
            ), "Must enable checkpointing when creating a seed checkpoint."
            trainer.checkpointer.save(curr_step=0, last_step=True)
            logger.info("Created seed checkpoint")
        else:
            trainer.train()
    except Exception:
        if trainer:
            trainer.close()
        raise
    else:
        trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        logger.info("Process group destroyed")


if __name__ == "__main__":
    main(Trainer)
