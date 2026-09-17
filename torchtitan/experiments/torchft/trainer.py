# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Annotated, Any

import torch
import tyro
from torch.distributed.elastic.multiprocessing.errors import record

from torchtitan.components.data.loader import BaseDataLoader, DataloaderExhaustedError
from torchtitan.components.data.types import TrainingMicrobatch
from torchtitan.config import apply_overrides, CompileConfig, Configurable
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.experiments.torchft.checkpoint import TorchFTCheckpointManager
from torchtitan.experiments.torchft.config.job_config import FaultTolerance
from torchtitan.experiments.torchft.manager import maybe_semi_sync_training
from torchtitan.experiments.torchft.optimizer import TorchFTOptimizersContainer
from torchtitan.models.common.aux_loss import collect_aux_loss_metrics
from torchtitan.observability.metrics import (
    build_device_memory_monitor,
    ensure_pp_loss_visible,
)
from torchtitan.protocols import BaseModel
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools import utils
from torchtitan.trainer import Trainer
from torchtitan.training_engine import TrainingEngine


logger = logging.getLogger(__name__)


class FaultTolerantTrainingEngine(TrainingEngine):
    """Training engine with TorchFT process groups and component adapters."""

    def __init__(
        self,
        config: TrainingEngine.Config,
        *,
        model_config: BaseModel.Config,
        max_num_documents: int | None,
        output_dir: str,
        fault_tolerance: FaultTolerance,
    ) -> None:
        # The base constructor invokes the distributed-runtime hook.
        self.fault_tolerance = fault_tolerance
        super().__init__(
            config,
            model_config=model_config,
            max_num_documents=max_num_documents,
            output_dir=output_dir,
        )

    def _initialize_distributed_runtime(self) -> None:
        device_module = utils.device_module
        # pyrefly: ignore [read-only]
        self.device = utils.get_local_device()
        device_module.set_device(self.device)

        global_ranks = []
        if self.fault_tolerance.enable:
            first_rank = (
                self.fault_tolerance.replica_id * self.fault_tolerance.group_size
            )
            last_rank = first_rank + self.fault_tolerance.group_size - 1
            global_ranks = list(range(first_rank, last_rank + 1))

        config = self.config
        dist_utils.init_distributed(
            config.comm,
            enable_cpu_backend=config.training.enable_cpu_offload,
            base_folder=self.output_dir,
            ranks=global_ranks,
        )
        self.ft_manager = self.fault_tolerance.build()
        self.parallel_dims = ParallelDims.from_config(
            config.parallelism, int(os.environ["WORLD_SIZE"])
        )
        self.gc_handler = utils.GarbageCollection(
            gc_freq=config.training.gc_freq,
            debug=config.training.gc_debug,
        )
        dist_utils.set_determinism(
            self.parallel_dims,
            self.device,
            config.debug,
            distinct_seed_mesh_dims=["pp"],
        )
        self.device_memory_monitor = build_device_memory_monitor()

    def _initialize_model(
        self,
        model_spec: ModelSpec,
        *,
        compile_config: CompileConfig | None,
        create_seed_checkpoint: bool = False,
    ) -> None:
        super()._initialize_model(
            model_spec,
            compile_config=compile_config,
            create_seed_checkpoint=create_seed_checkpoint,
        )
        self.ft_manager.maybe_set_all_reduce_hook(self.model_parts)

    def _initialize_optimizer(self, model_spec: ModelSpec) -> None:
        if isinstance(self.config.optimizer, TorchFTOptimizersContainer.Config):
            self.optimizers = self.config.optimizer.build(
                model_parts=self.model_parts,
                ft_manager=self.ft_manager,
            )
        else:
            self.optimizers = self.config.optimizer.build(model_parts=self.model_parts)
        if model_spec.post_optimizer_build_fn is not None:
            model_spec.post_optimizer_build_fn(
                self.optimizers,
                self.model_parts,
                self.parallel_dims,
            )
        self.lr_schedulers = self.config.lr_scheduler.build(
            optimizers=self.optimizers,
            training_steps=self.config.training.steps,
        )

    def _initialize_checkpointer(
        self,
        *,
        dataloader: BaseDataLoader | None,
        sd_adapter: Any | None,
    ) -> None:
        checkpointer_config = self.config.checkpointer
        if checkpointer_config is None:
            return
        self.checkpointer = checkpointer_config.build(
            dataloader=dataloader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={"train_state": self},
            sd_adapter=sd_adapter,
            base_folder=self.output_dir,
            ft_manager=self.ft_manager,
        )


class FaultTolerantTrainer(Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        checkpointer: Annotated[
            TorchFTCheckpointManager.Config | None, tyro.conf.AvoidSubcommands
        ] = None
        fault_tolerance: FaultTolerance = field(default_factory=FaultTolerance)

    engine: FaultTolerantTrainingEngine

    @record
    def __init__(self, config: Config):
        self.config = config
        model_spec = config.model_spec
        model_config = model_spec.model
        model_config.update_from_config(config=config)
        if config.override.imports:
            apply_overrides(config.override, config)
        config.__post_init__()

        self.engine = FaultTolerantTrainingEngine(
            config,
            model_config=model_config,
            max_num_documents=config.dataloader.max_num_documents,
            output_dir=config.dump_folder,
            fault_tolerance=config.fault_tolerance,
        )
        engine = self.engine
        parallel_dims = engine.parallel_dims

        # Logging needs to happen after distributed initialization.
        config.maybe_log()

        if parallel_dims.dp_enabled:
            dp_mesh = parallel_dims.get_mesh("dp")
            dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
        else:
            dp_degree, dp_rank = 1, 0
        dp_degree, dp_rank = engine.ft_manager.get_dp_info(dp_degree, dp_rank)

        # build tokenizer
        self.tokenizer = (
            config.tokenizer.build(tokenizer_path=config.hf_assets_path)
            if config.tokenizer is not None
            else None
        )

        num_pp_microbatches = (
            config.parallelism.num_pp_microbatches if parallel_dims.pp_enabled else 1
        )
        # build dataloader
        num_tokens_per_microbatch = (
            config.training.num_tokens_per_microbatch_per_dp_rank
        )
        self.dataloader = config.dataloader.build(
            dp_world_size=dp_degree,
            dp_rank=dp_rank,
            tokenizer=self.tokenizer,
            max_context_length=config.training.max_context_length,
            num_tokens_per_microbatch=num_tokens_per_microbatch,
        )

        # metrics logging (FT addition: ft_enable, ft_replica_id)
        self.metrics_processor = config.metrics.build(
            parallel_dims=parallel_dims,
            device_memory_monitor=engine.device_memory_monitor,
            dump_folder=config.dump_folder,
            pp_schedule=config.parallelism.pipeline_parallel_schedule,
            ft_enable=config.fault_tolerance.enable,
            ft_replica_id=config.fault_tolerance.replica_id,
            config_dict=config.to_dict(),
        )
        color = self.metrics_processor.color

        self.num_pp_microbatches = num_pp_microbatches
        num_tokens_per_dp_rank = (
            config.training.num_tokens_per_microbatch_per_dp_rank
            * self.num_pp_microbatches
        )
        num_tokens_per_train_step = config.training.num_tokens_per_train_step
        if num_tokens_per_train_step < 0:
            num_tokens_per_train_step = num_tokens_per_dp_rank * dp_degree
        if num_tokens_per_train_step % (num_tokens_per_dp_rank * dp_degree) != 0:
            raise ValueError(
                "training.num_tokens_per_train_step "
                f"({num_tokens_per_train_step}) must be divisible by the number "
                "of tokens processed globally in one gradient accumulation "
                f"iteration ({num_tokens_per_dp_rank * dp_degree})."
            )
        self.gradient_accumulation_steps = num_tokens_per_train_step // (
            num_tokens_per_dp_rank * dp_degree
        )

        engine.initialize(
            model_spec,
            compile_config=config.compile,
            dataloader=self.dataloader,
            sd_adapter=(
                model_spec.state_dict_adapter(model_config, config.hf_assets_path)
                if model_spec.state_dict_adapter
                else None
            ),
            create_seed_checkpoint=config.create_seed_checkpoint,
        )

        if parallel_dims.pp_enabled:
            ensure_pp_loss_visible(
                parallel_dims=parallel_dims,
                pp_schedule=config.parallelism.pipeline_parallel_schedule,
                color=color,
            )
        self.metrics_processor.num_flops_per_token = engine.num_flops_per_token

        # initialize device memory monitor and get peak flops for MFU calculation
        device_memory_monitor = engine.device_memory_monitor
        gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
        logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
        logger.info(
            f"{engine.device.type.upper()} memory usage for model: "
            f"{engine.model_device_mem_stats.max_reserved_gib:.2f}GiB"
            f"({engine.model_device_mem_stats.max_reserved_pct:.2f}%)"
        )

        self.metrics_processor.optimizers = engine.optimizers
        self.metrics_processor.model_parts = engine.model_parts

        # Build validator if validation is configured
        if config.validator is not None:
            pp_schedule, pp_has_first_stage, pp_has_last_stage = (
                (
                    engine.pp_schedule,
                    engine.pp_has_first_stage,
                    engine.pp_has_last_stage,
                )
                if parallel_dims.pp_enabled
                else (None, None, None)
            )

            self.validator = config.validator.build(
                parallelism=config.parallelism,
                job_config=config,
                dp_world_size=dp_degree,
                dp_rank=dp_rank,
                tokenizer=self.tokenizer,
                parallel_dims=parallel_dims,
                loss_fn=engine.loss_fn,
                validation_context=engine.train_context,
                metrics_processor=self.metrics_processor,
                seq_len=config.training.max_context_length,
                num_tokens_per_microbatch=num_tokens_per_microbatch,
                pp_schedule=pp_schedule,
                pp_has_first_stage=pp_has_first_stage,
                pp_has_last_stage=pp_has_last_stage,
            )

        logger.info(
            "Trainer is initialized with "
            f"{num_tokens_per_dp_rank} tokens per DP rank, "
            f"{num_tokens_per_train_step} tokens per train step, "
            f"gradient accumulation steps {self.gradient_accumulation_steps}, "
            f"maximum context length {config.training.max_context_length}, "
            f"total steps {config.training.steps} "
            f"(warmup {config.lr_scheduler.warmup_steps})"
        )

    def microbatch_generator(
        self, data_iterable: Iterable[TrainingMicrobatch]
    ) -> Iterator[TrainingMicrobatch]:
        data_iterator = iter(data_iterable)
        while True:
            data_load_start = time.perf_counter()
            try:
                microbatch = next(data_iterator)
            except StopIteration as ex:
                raise DataloaderExhaustedError() from ex
            ntokens_microbatch = (
                self.config.training.num_tokens_per_microbatch_per_dp_rank
            )
            self.metrics_processor.ntokens_since_last_log += ntokens_microbatch
            self.metrics_processor.data_loading_times.append(
                time.perf_counter() - data_load_start
            )
            yield microbatch

    def train_step(self, data_iterator: Iterator[TrainingMicrobatch]):
        engine = self.engine
        current_step = engine.num_completed_steps + 1
        # Save the current step learning rate for logging
        lr = engine.lr_schedulers.schedulers[0].get_last_lr()[0]
        should_log = self.metrics_processor.should_log(current_step)

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        parallel_dims = engine.parallel_dims
        # All groups form one optimizer step. Each microbatch group forms one
        # complete PP step, or one local forward/backward when PP is disabled.
        microbatch_groups: list[list[TrainingMicrobatch]] = []
        local_valid_tokens = 0
        for _ in range(self.gradient_accumulation_steps):
            microbatch_group = []
            for _ in range(self.num_pp_microbatches):
                microbatch = next(data_iterator)
                local_valid_tokens += microbatch.num_valid_tokens
                microbatch_group.append(microbatch)
            microbatch_groups.append(microbatch_group)

        # Keep the global token count on device so loss normalization does not
        # introduce a CPU synchronization in the training path.
        global_valid_tokens = torch.tensor(
            local_valid_tokens,
            dtype=torch.int64,
            device=engine.device,
        )
        if parallel_dims.dp_enabled:
            dp_mesh = parallel_dims.get_mesh("dp")
            global_valid_tokens = dist_utils.dist_sum_tensor(
                global_valid_tokens, dp_mesh
            )

        global_valid_tokens = engine.prepare_step(
            global_valid_tokens,
            num_accumulation_steps=self.gradient_accumulation_steps,
        )

        accumulated_loss: torch.Tensor | None = None
        for fwd_bwd_index, microbatch_group in enumerate(microbatch_groups):
            detached_loss = engine.forward_backward_microbatch(
                microbatch_group=microbatch_group,
                global_valid_tokens=global_valid_tokens,
                accumulation_index=fwd_bwd_index,
            )
            if should_log:
                if accumulated_loss is None:
                    # Take ownership before the next replay overwrites the
                    # graph-owned output. Later losses accumulate in place.
                    accumulated_loss = detached_loss.clone()
                else:
                    accumulated_loss.add_(detached_loss)

        grad_norm = engine.optimizer_step()

        # log metrics
        if not should_log:
            return

        assert accumulated_loss is not None

        if parallel_dims.dp_cp_enabled:
            # FT addition: use ft_manager.loss_sync_pg for extra process group
            ft_pg = engine.ft_manager.loss_sync_pg
            loss_mesh = parallel_dims.get_optional_mesh("loss")

            # For global_avg_loss, we want the average loss across all ranks:
            # accumulated_loss = local_loss_sum / global_valid_tokens
            # global_avg_loss = sum(local_loss_sum) / global_valid_tokens
            #                 = sum(accumulated_loss)
            #
            # For global_max_loss, we want the max of local average losses across ranks:
            # local_avg_loss = local_loss_sum / local_valid_tokens
            #                = (accumulated_loss * global_valid_tokens) / local_valid_tokens
            # global_max_loss = max(local_avg_loss)
            local_avg_loss = accumulated_loss * global_valid_tokens / local_valid_tokens
            global_avg_loss, global_max_loss, global_ntokens_seen = (
                dist_utils.dist_sum(accumulated_loss, loss_mesh, ft_pg),
                dist_utils.dist_max(local_avg_loss, loss_mesh, ft_pg),
                dist_utils.dist_sum(
                    torch.tensor(
                        engine.ntokens_seen,
                        dtype=torch.int64,
                        device=engine.device,
                    ),
                    loss_mesh,
                    ft_pg,
                ),
            )
            # ft_pg is None in semi-sync training.
            if ft_pg is not None:
                # Avoid artificial jumps in logged loss when replicas leave or rejoin.
                global_avg_loss /= ft_pg.size()
        else:
            global_avg_loss = global_max_loss = accumulated_loss.item()
            global_ntokens_seen = engine.ntokens_seen

        extra_metrics = {
            "n_tokens_seen": global_ntokens_seen,
            "lr": lr,
            **collect_aux_loss_metrics(engine.parallel_dims),
        }
        self.metrics_processor.log(
            engine.num_completed_steps,
            global_avg_loss,
            global_max_loss,
            grad_norm.item(),
            extra_metrics=extra_metrics,
        )

    @record
    def train(self):
        config = self.config
        engine = self.engine

        engine.load_checkpoint()
        logger.info(f"Training starts at step {engine.num_completed_steps + 1}")

        # FT addition: per-replica profiling leaf folder
        leaf_folder = (
            ""
            if not engine.ft_manager.enabled
            else f"replica_{engine.ft_manager.replica_id}"
        )
        with (
            config.profiler.build(
                global_step=engine.num_completed_steps,
                base_folder=config.dump_folder,
                leaf_folder=leaf_folder,
            ) as profiler,
            # FT addition: maybe_semi_sync_training context manager
            maybe_semi_sync_training(
                config.fault_tolerance,
                ft_manager=engine.ft_manager,
                model=engine.model_parts[0],
                n_layers=(
                    len(engine.model_config.layers)
                    if hasattr(engine.model_config, "layers")
                    else 0
                ),
                optimizer=engine.optimizers,
                fragment_fn=(
                    config.model_spec.fragment_fn
                    if hasattr(config.model_spec, "fragment_fn")
                    else None
                ),
            ),
        ):
            data_iterator = self.microbatch_generator(self.dataloader)
            while self.should_continue_training():
                try:
                    self.train_step(data_iterator)
                except DataloaderExhaustedError:
                    logger.warning("Ran out of data; last step was canceled.")
                    break

                engine.save_checkpoint(
                    last_step=(engine.num_completed_steps == config.training.steps),
                )

                # Run validation if validator is available
                if (
                    self.config.validator is not None
                    and self.validator.should_validate(engine.num_completed_steps)
                ):
                    self.validator.validate(
                        engine.model_parts, engine.num_completed_steps
                    )

                # signal the profiler that the next profiling step has started
                profiler.step()

                # reduce timeout after first train step for faster signal
                # (assuming lazy init and compilation are finished)
                if engine.num_completed_steps == 1:
                    dist_utils.set_pg_timeouts(
                        timeout=timedelta(seconds=config.comm.train_timeout_seconds),
                        parallel_dims=engine.parallel_dims,
                    )

        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)

        logger.info("Training completed")

    def should_continue_training(self) -> bool:
        return self.engine.num_completed_steps < self.config.training.steps

    def close(self) -> None:
        if self.dataloader:
            self.dataloader.close()
        self.engine.close()
        if self.metrics_processor:
            self.metrics_processor.close()
