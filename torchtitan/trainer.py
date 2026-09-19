# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import json
import logging
import os
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from typing import Annotated, Any

import torch
import tyro
from torch.distributed.elastic.multiprocessing.errors import record

from torchtitan.components.data.loader import BaseDataLoader, DataloaderExhaustedError
from torchtitan.components.data.types import TrainingMicrobatch
from torchtitan.components.tokenizer import BaseTokenizer, HuggingFaceTokenizer
from torchtitan.components.validate import BaseValidator, Validator
from torchtitan.config import Configurable
from torchtitan.config.configs import CompileConfig
from torchtitan.config.override import apply_overrides
from torchtitan.config.validation import validate_model_training_config
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.cuda_graph import cuda_graphs_supported
from torchtitan.models.common.aux_loss import collect_aux_loss_metrics
from torchtitan.observability import structured_logger as sl
from torchtitan.observability.metrics import ensure_pp_loss_visible, MetricsProcessor
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.training_engine import TrainingEngine


logger = logging.getLogger(__name__)


class Trainer(Configurable):
    """Dataset-driven training application that owns a :class:`TrainingEngine`.

    The trainer owns input, validation, and reporting policy. Its training engine
    owns model execution, optimization, checkpointing, profiling, and replay.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(TrainingEngine.Config):
        """
        Default container for training configuration.
        """

        # model_spec is always set by the registry. The unused string constructor
        # keeps Tyro from traversing the model config before applying Suppress.
        model_spec: Annotated[
            ModelSpec,
            tyro.conf.Suppress,
            tyro.conf.arg(constructor=str),
        ]

        hf_assets_path: str = "./tests/assets/tokenizer"
        """
        Path to HF assets folder. This folder contains local copies of Hugging Face assets,
        including model weights in .safetensors format, the model.safetensor.index.json file
        (fqn to file mapping), the config.json file, generation_config.json, and tokenizer files.
        """

        metrics: MetricsProcessor.Config = field(
            default_factory=MetricsProcessor.Config
        )
        tokenizer: BaseTokenizer.Config = field(
            default_factory=HuggingFaceTokenizer.Config
        )
        dataloader: BaseDataLoader.Config = field(default_factory=BaseDataLoader.Config)
        compile: Annotated[CompileConfig | None, tyro.conf.AvoidSubcommands] = None
        validator: Annotated[Validator.Config | None, tyro.conf.AvoidSubcommands] = None
        dump_folder: str = "./outputs"

        create_seed_checkpoint: Annotated[bool, tyro.conf.Suppress] = False
        """Initialize and save an unsharded model-only checkpoint, then exit."""

        def __post_init__(self):
            TrainingEngine.Config.__post_init__(self)
            if self.debug.batch_invariant:
                raise ValueError(
                    "Batch-invariant mode is not needed in supervised learning."
                )

            if (
                not self.training.disable_cuda_graphs
                and cuda_graphs_supported()
                and self.parallelism.pipeline_parallel_degree > 1
                and self.validator is not None
            ):
                raise ValueError(
                    "CUDA graphs with pipeline parallelism do not support "
                    "validation because validation reinitializes the shared "
                    "pipeline schedule. Disable validation or CUDA graphs."
                )

            if self.model_spec is not None:
                validate_model_training_config(
                    self.model_spec.model,
                    parallelism=self.parallelism,
                    training=self.training,
                    debug=self.debug,
                    activation_checkpoint=self.activation_checkpoint,
                    compile_config=self.compile,
                    max_num_documents=self.dataloader.max_num_documents,
                )

        def to_dict(self) -> dict[str, Any]:
            d = {}
            for f in dataclasses.fields(self):
                if f.name == "model_spec":
                    # ModelSpec contains callables that can't be serialized
                    d["model_spec"] = {
                        "name": self.model_spec.name,
                        "flavor": self.model_spec.flavor,
                        "model": self.model_spec.model.to_dict(),
                    }
                else:
                    val = getattr(self, f.name)
                    if hasattr(val, "to_dict"):
                        d[f.name] = val.to_dict()
                    elif dataclasses.is_dataclass(val):
                        d[f.name] = asdict(val)
                    else:
                        d[f.name] = val
            return d

        def maybe_log(self) -> None:
            if self.debug.print_config:
                logger.info(
                    f"Running with configs: {json.dumps(self.to_dict(), indent=2, ensure_ascii=False)}"
                )

            if self.debug.save_config_file is not None:
                config_file = os.path.join(
                    self.dump_folder, self.debug.save_config_file
                )
                if torch.distributed.is_initialized():
                    if torch.distributed.get_rank() == 0:
                        os.makedirs(os.path.dirname(config_file), exist_ok=True)
                        with open(config_file, "w") as f:
                            json.dump(self.to_dict(), f, indent=2)
                    logger.info(f"Saved job configs to {config_file}")
                else:
                    logger.warning(
                        "Job configs logging is disabled due to torch.distributed not initialized."
                    )

    config: Config
    engine: TrainingEngine
    engine_cls: type[TrainingEngine] = TrainingEngine

    tokenizer: BaseTokenizer
    dataloader: BaseDataLoader
    validator: BaseValidator
    metrics_processor: MetricsProcessor
    gradient_accumulation_steps: int
    num_pp_microbatches: int

    # Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
    @record
    def __init__(self, config: Config):
        self.config = config
        model_spec = config.model_spec
        model_config = model_spec.model
        model_config.update_from_config(config=config)

        # Apply overrides to the full config tree, before any component is
        # built. The model config is reached via ModelSpec.traverse. Model
        # overrides must run after update_from_config above (it sets sharding
        # config on the pre-override modules); all other components (optimizer,
        # loss, dataloader, …) are built later in __init__.
        if config.override.imports:
            apply_overrides(config.override, config)
        # Overrides may change any config field; re-run the full validation.
        # __post_init__ only raises (no mutation), so re-running is safe.
        config.__post_init__()

        self.engine = self.engine_cls(
            config,
            model_config=model_config,
            max_num_documents=config.dataloader.max_num_documents,
            output_dir=config.dump_folder,
        )
        engine = self.engine
        parallel_dims = engine.parallel_dims

        # Logging needs to happen after distributed initialized
        config.maybe_log()

        if parallel_dims.dp_enabled:
            dp_mesh = parallel_dims.get_mesh("dp")
            dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
        else:
            dp_degree, dp_rank = 1, 0

        # metrics logging
        self.metrics_processor = config.metrics.build(
            parallel_dims=parallel_dims,
            device_memory_monitor=engine.device_memory_monitor,
            dump_folder=config.dump_folder,
            pp_schedule=config.parallelism.pipeline_parallel_schedule,
            config_dict=config.to_dict(),
            has_quantization=engine.has_quantization,
        )
        color = self.metrics_processor.color

        self.num_pp_microbatches = (
            config.parallelism.num_pp_microbatches if parallel_dims.pp_enabled else 1
        )
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

        self.tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)
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
        self.metrics_processor.optimizers = engine.optimizers
        self.metrics_processor.model_parts = engine.model_parts

        logger.info(
            "Peak FLOPS used for computing MFU: "
            f"{self.metrics_processor.gpu_peak_flops:.3e}"
        )
        logger.info(
            f"{engine.device.type.upper()} memory usage for model: "
            f"{engine.model_device_mem_stats.max_reserved_gib:.2f}GiB"
            f"({engine.model_device_mem_stats.max_reserved_pct:.2f}%)"
        )

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
        """Return microbatches while recording data-loading metrics.

        Note: Tensors are yielded on CPU. The caller is responsible for moving
        them to GPU when needed. This allows for more efficient memory usage
        when doing gradient accumulation.
        """
        data_iterator = iter(data_iterable)

        while True:
            data_load_start = time.perf_counter()
            try:
                microbatch = next(data_iterator)
            except StopIteration as ex:
                # If data runs out during gradient accumulation, that
                # entire step will not be executed.
                raise DataloaderExhaustedError() from ex
            ntokens_microbatch = (
                self.config.training.num_tokens_per_microbatch_per_dp_rank
            )
            self.metrics_processor.ntokens_since_last_log += ntokens_microbatch
            self.metrics_processor.data_loading_times.append(
                time.perf_counter() - data_load_start
            )

            # Tensors stay on CPU; moved to GPU per-microbatch during training
            yield microbatch

    def train_step(self, data_iterator: Iterator[TrainingMicrobatch]):
        engine = self.engine
        current_step = engine.num_completed_steps + 1
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
                with sl.log_trace_span("fetching_batch"):
                    microbatch = next(data_iterator)
                local_valid_tokens += microbatch.num_valid_tokens
                microbatch_group.append(microbatch)
            microbatch_groups.append(microbatch_group)
        sl.log_trace_scalar({"local_valid_tokens": local_valid_tokens})

        # Keep the global token count on device so loss normalization does not
        # introduce a CPU synchronization in the training path.
        local_valid_tokens_tensor = torch.tensor(
            local_valid_tokens,
            dtype=torch.int64,
            device=engine.device,
        )
        if parallel_dims.dp_enabled:
            dp_mesh = parallel_dims.get_mesh("dp")
            global_valid_tokens = dist_utils.dist_sum_tensor(
                local_valid_tokens_tensor, dp_mesh
            )
        else:
            global_valid_tokens = local_valid_tokens_tensor

        # Auxiliary losses normalize by the same per-step token count as the
        # main loss, so their scale is independent of parallelism degrees.
        global_valid_tokens = engine.prepare_step(
            global_valid_tokens,
            num_accumulation_steps=self.gradient_accumulation_steps,
        )

        # Process each gradient accumulation step, then free its inputs.
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

        # Capture the learning rates used by this optimizer update before the
        # scheduler advances in engine.optimizer_step().
        lr_metrics = engine.lr_schedulers.get_metrics() if should_log else {}
        grad_norm = engine.optimizer_step()

        # log metrics
        if not should_log:
            return

        assert accumulated_loss is not None

        with sl.log_trace_span("collect_dist_metrics"):
            sl.log_trace_scalar({"global_valid_tokens": int(global_valid_tokens)})

            if parallel_dims.dp_cp_enabled:
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
                local_avg_loss = (
                    accumulated_loss * global_valid_tokens / local_valid_tokens
                )
                global_avg_loss, global_max_loss, global_ntokens_seen = (
                    dist_utils.dist_sum(accumulated_loss, loss_mesh),
                    dist_utils.dist_max(local_avg_loss, loss_mesh),
                    dist_utils.dist_sum(
                        torch.tensor(
                            engine.ntokens_seen,
                            dtype=torch.int64,
                            device=engine.device,
                        ),
                        loss_mesh,
                    ),
                )
            else:
                global_avg_loss = global_max_loss = float(accumulated_loss.item())
                global_ntokens_seen = engine.ntokens_seen

        extra_metrics = {
            "n_tokens_seen": global_ntokens_seen,
            **lr_metrics,
            **collect_aux_loss_metrics(parallel_dims),
        }
        self.metrics_processor.log(
            engine.num_completed_steps,
            global_avg_loss,
            global_max_loss,
            float(grad_norm.item()),
            extra_metrics=extra_metrics,
        )

    @record
    def train(self):
        config = self.config
        engine = self.engine

        sl.log_trace_instant("training_start")

        engine.load_checkpoint()

        # Capture loaded step for relative_step calculation.
        # After checkpoint load, this is the restored number of completed updates.
        loaded_step = engine.num_completed_steps

        logger.info(f"Training starts at step {engine.num_completed_steps + 1}")

        engine.start_profiler()
        try:
            data_iterator = self.microbatch_generator(self.dataloader)
            while self.should_continue_training():
                current_step = engine.num_completed_steps + 1
                sl.set_step(current_step, relative_step=current_step - loaded_step)

                with sl.log_trace_span("step"):
                    try:
                        self.train_step(data_iterator)
                    except DataloaderExhaustedError:
                        logger.warning("Ran out of data; last step was canceled.")
                        break

                    engine.save_checkpoint(
                        last_step=(engine.num_completed_steps == config.training.steps)
                    )

                    # Run validation if validator is available
                    if (
                        self.config.validator is not None
                        and self.validator.should_validate(engine.num_completed_steps)
                    ):
                        self.validator.validate(
                            engine.model_parts, engine.num_completed_steps
                        )

                    engine.step_profiler()

                    # Reduce timeout after the first train step of THIS process
                    # (assuming lazy init and compilation are finished). Use the
                    # relative step so this fires on resumed runs too.
                    if engine.num_completed_steps - loaded_step == 1:
                        dist_utils.set_pg_timeouts(
                            timeout=timedelta(
                                seconds=config.comm.train_timeout_seconds
                            ),
                            parallel_dims=engine.parallel_dims,
                        )
        finally:
            # The entry point also calls close() for checkpoint and graph
            # cleanup; close the profiler here so direct train() callers get
            # balanced lifecycle handling.
            engine.close_profiler()

        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)

        logger.info("Training completed")

    def should_continue_training(self) -> bool:
        return self.engine.num_completed_steps < self.config.training.steps

    def close(self) -> None:
        if hasattr(self, "dataloader") and self.dataloader:
            self.dataloader.close()
        if hasattr(self, "engine"):
            self.engine.close()
        if hasattr(self, "metrics_processor") and self.metrics_processor:
            self.metrics_processor.close()
