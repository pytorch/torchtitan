# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import time
from datetime import timedelta
from typing import Any, Iterable

import torch
from torch.distributed.elastic.multiprocessing.errors import record

import torchtitan.protocols.train_spec as train_spec_module
from torchtitan.components.dataloader import DataloaderExhaustedError
from torchtitan.components.metrics import build_metrics_processor
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.config import JobConfig
from torchtitan.distributed import utils as dist_utils
from torchtitan.hf_datasets.text_datasets import build_text_dataloader
from torchtitan.tools import utils
from torchtitan.tools.logging import logger
from torchtitan.tools.profiling import (
    maybe_enable_memory_snapshot,
    maybe_enable_profiling,
)
from torchtitan.train import main

from .engine import ForgeEngine


class Trainer(ForgeEngine):
    tokenizer: train_spec_module.BaseTokenizer | None
    dataloader: train_spec_module.BaseDataLoader
    validator: train_spec_module.BaseValidator
    metrics_processor: train_spec_module.MetricsProcessor

    # additional training states
    step: int

    # Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
    @record
    def __init__(self, job_config: JobConfig):
        logger.info(f"Starting job: {job_config.job.description}")

        if job_config.job.print_config:
            logger.info(f"Running with args: {job_config.to_dict()}")

        if job_config.experimental.custom_import:
            importlib.import_module(job_config.experimental.custom_import)

        # NOTE: Here we are passing in JobConfig as a superset of ForgeJobConfig
        super().__init__(job_config)

        # build tokenizer
        self.tokenizer = build_hf_tokenizer(job_config)

        # build dataloader
        self.dataloader = build_text_dataloader(
            dp_world_size=self.dp_degree,
            dp_rank=self.dp_rank,
            tokenizer=self.tokenizer,
            job_config=job_config,
        )

        model_args = self.model_args
        logger.info(
            f"Built {job_config.model.name} {job_config.model.flavor} with {model_args}"
        )

        # metrics logging
        self.metrics_processor = build_metrics_processor(
            job_config, self.parallel_dims, model_args
        )
        color = self.metrics_processor.color

        self.metrics_processor.num_flops_per_token = self.num_flops_per_token

        logger.info(
            f"{color.blue}Model {job_config.model.name} {job_config.model.flavor} "
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
        if job_config.validation.enable:
            self.validator = build_validator(
                job_config=job_config,
                dp_world_size=self.dp_degree,
                dp_rank=self.dp_rank,
                tokenizer=self.tokenizer,
                parallel_dims=self.parallel_dims,
                loss_fn=self.loss_fn,
                validation_context=self.train_context,
                maybe_enable_amp=self.maybe_enable_amp,
            )

        logger.info(
            "Trainer is initialized with "
            f"local batch size {job_config.training.local_batch_size}, "
            f"global batch size {self.global_batch_size}, "
            f"gradient accumulation steps {self.gradient_accumulation_steps}, "
            f"sequence length {job_config.training.seq_len}, "
            f"total steps {job_config.training.steps} "
            f"(warmup {job_config.lr_scheduler.warmup_steps})."
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

            # Move tensors to the appropriate device
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor):
                    input_dict[k] = v.to(device_type)
            labels = labels.to(device_type)

            yield input_dict, labels

    def forward_backward_step(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        model_parts = self.model_parts
        parallel_dims = self.parallel_dims

        inputs = input_dict["input"]
        extra_kwargs = {}

        if getattr(self.model_args, "attn_type", "sdpa") == "flex":
            extra_kwargs["attention_masks"] = model_parts[0].get_attention_masks(
                input_batch=inputs,
                tokenizer=self.tokenizer,
            )

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

        if parallel_dims.pp_enabled:
            # Pipeline Parallel forward / backward inside step() call
            with self.train_context(optional_context_parallel_ctx):
                targets, losses = (
                    (labels, []) if self.pp_has_last_stage else (None, None)
                )
                if self.pp_has_first_stage:
                    self.pp_schedule.step(
                        inputs,
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
                # using sum instead of mean because we already rescale the
                # loss_fn down by a factor of n_microbatches in
                # torchtitan/distributed/pipeline_parallel.py
                torch.sum(torch.stack(losses)).to(self.device)
                if self.pp_has_last_stage
                else torch.tensor([-1.0], device=self.device)
            )
        else:
            # Non-PP forward / backward
            with self.train_context(optional_context_parallel_ctx):
                assert len(model_parts) == 1
                with self.maybe_enable_amp:
                    pred = model_parts[0](inputs, **extra_kwargs)
                    loss = self.loss_fn(pred, labels)
                # need to free to before bwd to avoid peaking memory
                del pred
                loss.backward()

        return loss

    def train_step(
        self, data_iterator: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        self.optimizers.zero_grad()

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        parallel_dims = self.parallel_dims

        accumulated_losses = []
        # If data runs out during gradient accumulation, that
        # entire step will not be executed.
        for microbatch in range(self.gradient_accumulation_steps):
            input_dict, labels = next(data_iterator)
            loss = self.forward_backward_step(input_dict, labels)
            accumulated_losses.append(loss.detach())

        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.job_config.training.max_norm,
            foreach=True,
            pp_mesh=(
                parallel_dims.world_mesh["pp"] if parallel_dims.pp_enabled else None
            ),
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
                dist_utils.dist_mean(loss, parallel_dims.world_mesh["dp_cp"]),
                dist_utils.dist_max(loss, parallel_dims.world_mesh["dp_cp"]),
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
        job_config = self.job_config

        self.checkpointer.load(step=job_config.checkpoint.load_step)
        logger.info(f"Training starts at step {self.step + 1}.")

        with (
            maybe_enable_profiling(
                job_config.profiling,
                global_step=self.step,
                base_folder=job_config.job.dump_folder,
            ) as torch_profiler,
            maybe_enable_memory_snapshot(
                job_config.profiling,
                global_step=self.step,
                base_folder=job_config.job.dump_folder,
            ) as memory_profiler,
        ):
            data_iterator = self.batch_generator(self.dataloader)
            while self.step < job_config.training.steps:
                self.step += 1
                self.gc_handler.run(self.step)
                try:
                    self.train_step(data_iterator)
                except DataloaderExhaustedError:
                    logger.warning("Ran out of data; last step was canceled.")
                    break

                # Run validation if validator is available
                if (
                    self.job_config.validation.enable
                    and self.validator.should_validate(self.step)
                ):
                    with self.loss_fn.no_rescale():
                        self.validator.validate(self.model_parts, self.step)

                self.checkpointer.save(
                    self.step, last_step=(self.step == job_config.training.steps)
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
                        timeout=timedelta(
                            seconds=job_config.comm.train_timeout_seconds
                        ),
                        world_mesh=self.parallel_dims.world_mesh,
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


if __name__ == "__main__":
    main(Trainer)
