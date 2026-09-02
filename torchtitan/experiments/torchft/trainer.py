# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import time
from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import timedelta
from typing import cast

import torch
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.tensor import DTensor

from torchtitan.components.data.loader import DataloaderExhaustedError
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.quantization.utils import has_quantization
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.config.override import apply_overrides
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.distributed.cudagraph import wrap_with_cuda_graph
from torchtitan.experiments.torchft.config.job_config import FaultTolerance
from torchtitan.experiments.torchft.manager import (
    maybe_semi_sync_training,
    TorchFTManager,
)
from torchtitan.experiments.torchft.optimizer import TorchFTOptimizersContainer
from torchtitan.observability import structured_logger as sl
from torchtitan.observability.sdc_replayer import ScalarStateAccessor, SDCReplayer
from torchtitan.protocols import BaseModel
from torchtitan.tools import utils
from torchtitan.tools.logging import logger
from torchtitan.trainer import Trainer


class FaultTolerantTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        fault_tolerance: FaultTolerance = field(default_factory=FaultTolerance)

    ft_manager: TorchFTManager

    @record
    def __init__(self, config: Config):
        torch._C._log_api_usage_once("torchtitan.train")

        self.config = config
        assert (
            config.model_spec is not None
        ), "model_spec must be set before creating Trainer"
        model_spec = config.model_spec

        device_module, device_type = utils.device_module, utils.device_type
        # pyrefly: ignore [read-only]
        self.device = utils.get_local_device()
        # Device has to be set before creating TorchFT manager.
        device_module.set_device(self.device)

        # init distributed and build meshes (FT override handles ft_manager creation)
        self.parallel_dims = parallel_dims = self.init_distributed()

        # Validate dense activation token-count evenness.
        num_tokens_per_pp_microbatch = (
            config.training.num_tokens_per_microbatch_per_dp_rank
        )
        seq_len_divisor = (
            parallel_dims.tp if config.parallelism.enable_sequence_parallel else 1
        ) * (2 * parallel_dims.cp if parallel_dims.cp > 1 else 1)
        if num_tokens_per_pp_microbatch % seq_len_divisor != 0:
            raise ValueError(
                "The number of tokens per pipeline microbatch "
                f"({num_tokens_per_pp_microbatch}) must be "
                f"divisible by {seq_len_divisor} for the configured "
                "sequence/context parallelism."
            )

        # TODO(pianpwk): Transitional until the local-SPMD and full-DTensor
        # backends share one runtime mesh/type mechanism.
        dist_utils.set_spmd_backend(config.parallelism.spmd_backend)

        # Logging needs to happen after distributed initialized
        config.maybe_log()

        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            batch_degree, batch_rank = batch_mesh.size(), batch_mesh.get_local_rank()
        else:
            batch_degree, batch_rank = 1, 0

        # FT addition: adjust dp info via ft_manager
        batch_degree, batch_rank = self.ft_manager.get_dp_info(batch_degree, batch_rank)

        # take control of garbage collection to avoid stragglers
        self.gc_handler = utils.GarbageCollection(
            gc_freq=config.training.gc_freq, debug=config.training.gc_debug
        )

        # Set random seed, and maybe enable deterministic mode
        # (mainly for debugging, expect perf loss).
        dist_utils.set_determinism(
            parallel_dims,
            self.device,
            config.debug,
            distinct_seed_mesh_dims=["pp"],
        )

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
        num_tokens_per_batch = config.training.num_tokens_per_microbatch_per_dp_rank
        self.dataloader = config.dataloader.build(
            dp_world_size=batch_degree,
            dp_rank=batch_rank,
            tokenizer=self.tokenizer,
            max_context_length=config.training.max_context_length,
            num_tokens_per_batch=num_tokens_per_batch,
        )

        # build model (using meta init)
        model_config = model_spec.model
        # set the model args from training job configs
        model_config.update_from_config(
            config=config,
        )
        self.model_config = model_config

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

        logger.info(
            f"Building {model_spec.name} {model_spec.flavor} "
            f"with {json.dumps(model_config.to_dict(), indent=2, ensure_ascii=False)}"
        )
        with (
            torch.device("meta"),
            utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]),
        ):
            model = model_config.build()

        # Verify all submodules satisfy the Module protocol
        model.verify_module_protocol()

        # metrics logging (FT addition: ft_enable, ft_replica_id)
        self.metrics_processor = config.metrics.build(
            parallel_dims=parallel_dims,
            dump_folder=config.dump_folder,
            pp_schedule=config.parallelism.pipeline_parallel_schedule,
            ft_enable=config.fault_tolerance.enable,
            ft_replica_id=config.fault_tolerance.replica_id,
            config_dict=config.to_dict(),
            has_quantization=has_quantization(model_config),
        )
        color = self.metrics_processor.color

        # calculate model size and flops per token
        (
            model_param_count,
            self.metrics_processor.num_flops_per_token,
        ) = model_config.get_nparams_and_flops(
            model, config.training.max_context_length
        )

        logger.info(
            f"{color.blue}Model {model_spec.name} {model_spec.flavor} "
            f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
        )

        # move sharded model to CPU/GPU and initialize weights via DTensor
        buffer_device: torch.device | None
        if config.checkpoint.create_seed_checkpoint:
            init_device = "cpu"
            buffer_device = None
        elif config.training.enable_cpu_offload:
            init_device = "cpu"
            buffer_device = torch.device(device_type)
        else:
            init_device = device_type
            buffer_device = None

        self.loss_fn = config.loss.build(
            compile_config=config.compile,
        )

        self.num_pp_microbatches = num_pp_microbatches
        num_tokens_per_dp_rank = (
            config.training.num_tokens_per_microbatch_per_dp_rank
            * self.num_pp_microbatches
        )
        num_tokens_per_train_step = config.training.num_tokens_per_train_step
        if num_tokens_per_train_step < 0:
            num_tokens_per_train_step = num_tokens_per_dp_rank * batch_degree
        if num_tokens_per_train_step % (num_tokens_per_dp_rank * batch_degree) != 0:
            raise ValueError(
                "training.num_tokens_per_train_step "
                f"({num_tokens_per_train_step}) must be divisible by the number "
                "of tokens processed globally in one gradient accumulation "
                f"iteration ({num_tokens_per_dp_rank * batch_degree})."
            )
        self.gradient_accumulation_steps = num_tokens_per_train_step // (
            num_tokens_per_dp_rank * batch_degree
        )

        # apply parallelisms and initialization
        with sl.log_trace_span("model_parallelism_init"):
            if parallel_dims.pp_enabled:
                from torchtitan.components.metrics import ensure_pp_loss_visible

                if not model_spec.pipelining_fn:
                    raise RuntimeError(
                        f"Pipeline Parallel is enabled but {model_spec.name} "
                        f"does not support pipelining"
                    )

                # apply both PT-D Pipeline Parallel and SPMD-style PT-D techniques
                (
                    self.pp_schedule,
                    self.model_parts,
                    self.pp_has_first_stage,
                    self.pp_has_last_stage,
                ) = model_spec.pipelining_fn(
                    model,
                    parallel_dims=parallel_dims,
                    training=config.training,
                    parallelism=config.parallelism,
                    compile_config=config.compile,
                    ac_config=config.activation_checkpoint,
                    dump_folder=config.dump_folder,
                    device=self.device,
                    model_config=model_config,
                    parallelize_fn=model_spec.parallelize_fn,
                    loss_fn=self.loss_fn,
                )
                # when PP is enabled, `model` obj is no longer used after this point,
                # model_parts is used instead
                del model

                for m in self.model_parts:
                    m.to_empty(device=init_device)
                    with torch.no_grad():
                        cast(BaseModel, m).init_weights(buffer_device=buffer_device)
                    m.train()

                # confirm that user will be able to view loss metrics on the console
                ensure_pp_loss_visible(
                    parallel_dims=parallel_dims,
                    pp_schedule=config.parallelism.pipeline_parallel_schedule,
                    color=color,
                )
            else:
                if not config.checkpoint.create_seed_checkpoint:
                    # Skip parallelize_fn for seed checkpoints — nothing from
                    # it is needed (AC, compile, nD parallelism, mixed precision, etc.).
                    model = model_spec.parallelize_fn(
                        model,
                        parallel_dims=parallel_dims,
                        training=config.training,
                        parallelism=config.parallelism,
                        compile_config=config.compile,
                        ac_config=config.activation_checkpoint,
                        dump_folder=config.dump_folder,
                    )

                model.to_empty(device=init_device)
                with torch.no_grad():
                    cast(BaseModel, model).init_weights(buffer_device=buffer_device)
                model.train()

                self.model_parts = [model]

        # Set lm_head reference for ChunkedLossWrapper after model construction.
        # Non-PP: single model part always has lm_head.
        # PP: only the last stage has lm_head; non-last stages skip this.
        if isinstance(self.loss_fn, ChunkedLossWrapper):
            if parallel_dims.pp_enabled:
                if self.pp_has_last_stage:
                    lm_head = self.model_parts[-1].lm_head
                    assert (
                        lm_head is not None
                    ), "Last PP stage must have lm_head for ChunkedLossWrapper"
                    self.loss_fn.set_lm_head(
                        lm_head  # pyrefly: ignore[bad-argument-type]
                    )
                    self.model_parts[
                        -1
                    ]._skip_lm_head = True  # pyrefly: ignore[bad-argument-type]
            else:
                assert len(self.model_parts) == 1
                lm_head = self.model_parts[0].lm_head
                assert (
                    lm_head is not None
                ), "Model must have lm_head for ChunkedLossWrapper"
                self.loss_fn.set_lm_head(lm_head)  # pyrefly: ignore[bad-argument-type]
                self.model_parts[
                    0
                ]._skip_lm_head = True  # pyrefly: ignore[bad-argument-type]

        # FT addition: set all reduce hook
        self.ft_manager.maybe_set_all_reduce_hook(self.model_parts)

        # initialize device memory monitor and get peak flops for MFU calculation
        device_memory_monitor = self.metrics_processor.device_memory_monitor
        gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
        logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
        device_mem_stats = device_memory_monitor.get_peak_stats()
        logger.info(
            f"{device_type.upper()} memory usage for model: "
            f"{device_mem_stats.max_reserved_gib:.2f}GiB"
            f"({device_mem_stats.max_reserved_pct:.2f}%)"
        )

        # build optimizer after applying parallelisms to the model
        # FT addition: pass ft_manager for TorchFTOptimizersContainer
        if isinstance(config.optimizer, TorchFTOptimizersContainer.Config):
            self.optimizers = config.optimizer.build(
                model_parts=self.model_parts, ft_manager=self.ft_manager
            )
        else:
            self.optimizers = config.optimizer.build(model_parts=self.model_parts)
        if model_spec.post_optimizer_build_fn is not None:
            model_spec.post_optimizer_build_fn(
                self.optimizers, self.model_parts, parallel_dims
            )
        self.lr_schedulers = config.lr_scheduler.build(
            optimizers=self.optimizers,
            training_steps=config.training.steps,
        )
        self.metrics_processor.optimizers = self.optimizers
        self.metrics_processor.model_parts = self.model_parts

        # Initialize trainer states that will be saved in checkpoint.
        # These attributes must be initialized before checkpoint loading.
        self.step = 0
        self.ntokens_seen = 0

        # SDC replay state is process-local and not checkpointed; its check
        # schedule restarts after every checkpoint load (see load_state_dict).
        self.sdc_replayer: SDCReplayer | None = None
        if config.sdc_replayer is not None:
            self.sdc_replayer = config.sdc_replayer.build(
                modules=self.model_parts,
                device=self.device,
                # ntokens_seen is the only trainer scalar the replayed
                # forward/backward mutates; self.step is incremented outside
                # the replay boundary and needs no capture.
                scalar_state={
                    "ntokens_seen": ScalarStateAccessor(
                        get=lambda: self.ntokens_seen,
                        set=lambda value: setattr(self, "ntokens_seen", value),
                    )
                },
            )

        # FT addition: pass ft_manager to CheckpointManager
        self.checkpointer = config.checkpoint.build(
            dataloader=self.dataloader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={"train_state": self},
            sd_adapter=(
                model_spec.state_dict_adapter(model_config, config.hf_assets_path)
                if model_spec.state_dict_adapter
                else None
            ),
            base_folder=config.dump_folder,
            ft_manager=self.ft_manager,
        )

        self.train_context = dist_utils.get_spmd_context(
            parallel_dims=parallel_dims,
            spmd_typechecking=(
                config.parallelism.spmd_backend == "spmd_types"
                and config.debug.spmd_typechecking
            ),
        )
        self.fwd_bwd_fn = self._forward_backward_body
        if not config.training.disable_cuda_graphs:
            self.fwd_bwd_fn = wrap_with_cuda_graph(self.fwd_bwd_fn)

        # Build validator if validation is configured
        if config.validator.enable:
            pp_schedule, pp_has_first_stage, pp_has_last_stage = (
                (
                    self.pp_schedule,
                    self.pp_has_first_stage,
                    self.pp_has_last_stage,
                )
                if parallel_dims.pp_enabled
                else (None, None, None)
            )

            self.validator = config.validator.build(
                parallelism=config.parallelism,
                dp_world_size=batch_degree,
                dp_rank=batch_rank,
                tokenizer=self.tokenizer,
                parallel_dims=parallel_dims,
                loss_fn=self.loss_fn,
                validation_context=self.train_context,
                metrics_processor=self.metrics_processor,
                seq_len=config.training.max_context_length,
                num_tokens_per_batch=num_tokens_per_batch,
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

    def init_distributed(self) -> ParallelDims:
        config = self.config

        # determine the global ranks when fault tolerance is enabled
        global_ranks = []
        ft_config = config.fault_tolerance
        if ft_config.enable:
            group_size = ft_config.group_size
            replica_id = ft_config.replica_id
            first_rank = replica_id * group_size
            last_rank = first_rank + group_size - 1
            global_ranks = list(range(first_rank, last_rank + 1))

        # init distributed and build meshes
        dist_utils.init_distributed(
            config.comm,
            enable_cpu_backend=config.training.enable_cpu_offload,
            base_folder=config.dump_folder,
            ranks=global_ranks,
        )

        # FT addition: build TorchFTManager
        self.ft_manager = config.fault_tolerance.build()

        world_size = int(os.environ["WORLD_SIZE"])

        return ParallelDims.from_config(config.parallelism, world_size)

    def train_step(
        self, data_iterator: Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        self.optimizers.zero_grad(set_to_none=self.config.training.disable_cuda_graphs)
        # Save per-optimizer-group learning rates for logging
        lr_metrics = self.lr_schedulers.get_metrics()
        should_log = self.metrics_processor.should_log(self.step)

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        parallel_dims = self.parallel_dims
        # All groups form one optimizer step; each group feeds one fwd-bwd call.
        microbatch_groups: list[list[tuple[dict[str, torch.Tensor], torch.Tensor]]] = []
        local_valid_tokens = torch.tensor(0, dtype=torch.int64)
        for _ in range(self.gradient_accumulation_steps):
            microbatches = []
            for _ in range(self.num_pp_microbatches):
                with sl.log_trace_span("fetching_batch"):
                    input_dict, labels = next(data_iterator)
                # Popped so the batch reaching the model holds only its kwargs.
                local_valid_tokens += input_dict.pop("num_valid_tokens")
                microbatches.append((input_dict, labels))
            microbatch_groups.append(microbatches)
        sl.log_trace_scalar({"local_valid_tokens": int(local_valid_tokens)})

        # Keep the global token count on device so loss normalization does not
        # introduce a CPU synchronization in the training path.
        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            global_valid_tokens = dist_utils.dist_sum_tensor(
                local_valid_tokens.to(self.device), batch_mesh
            )
        else:
            global_valid_tokens = local_valid_tokens.to(self.device)

        # Process each gradient accumulation step, then free its inputs.
        accumulated_loss: torch.Tensor | None = None
        # int32 is supported by NCCL reductions, unlike bool.
        loss_is_finite = torch.ones((), dtype=torch.int32, device=self.device)
        for fwd_bwd_index, microbatches in enumerate(microbatch_groups):
            input_dict_mbs = []
            label_mbs = []
            for input_dict, labels in microbatches:
                for key, value in input_dict.items():
                    if isinstance(value, torch.Tensor):
                        input_dict[key] = value.to(self.device)
                input_dict_mbs.append(input_dict)
                label_mbs.append(labels.to(self.device))

            if parallel_dims.pp_enabled:
                fwd_bwd_input_dict = input_dict_mbs
                fwd_bwd_labels = label_mbs
            else:
                assert len(input_dict_mbs) == len(label_mbs) == 1
                fwd_bwd_input_dict = input_dict_mbs[0]
                fwd_bwd_labels = label_mbs[0]

            def fwd_bwd() -> torch.Tensor:
                return self.forward_backward_step(
                    input_dict=fwd_bwd_input_dict,
                    labels=fwd_bwd_labels,
                    global_valid_tokens=global_valid_tokens,
                )

            if self.sdc_replayer is not None and fwd_bwd_index == 0:
                # Only the step's first gradient-accumulation group is
                # replay-checked; under PP one group is a complete pipeline
                # schedule. Later groups exercise the same compute and
                # communication paths, so checking them too would only add
                # overhead.
                loss = self.sdc_replayer.run_fwd_bwd(fwd_bwd, step=self.step)
            else:
                loss = fwd_bwd()
            detached_loss = loss.detach()
            local_loss = (
                detached_loss.to_local()
                if isinstance(detached_loss, DTensor)
                else detached_loss
            )
            loss_is_finite.logical_and_(torch.isfinite(local_loss).all())
            if should_log:
                if accumulated_loss is None:
                    # Take ownership before the next replay overwrites the
                    # graph-owned output. Later losses accumulate in place.
                    accumulated_loss = detached_loss.clone()
                else:
                    accumulated_loss.add_(detached_loss)

        with sl.log_trace_span("optim"):
            grad_norm = dist_utils.clip_grad_norm_(
                [p for m in self.model_parts for p in m.parameters()],
                self.config.training.max_norm,
                foreach=True,
                pp_mesh=parallel_dims.get_optional_mesh("pp"),
                ep_enabled=parallel_dims.ep_enabled,
            )
            # Only the last PP stage owns the loss. First combine its DP/CP
            # replicas, then propagate the result across PP. TP replicas have
            # identical loss values, and grad_norm is already world-reduced by
            # clip_grad_norm_.
            if not parallel_dims.pp_enabled or self.pp_has_last_stage:
                loss_mesh = parallel_dims.get_optional_mesh("loss")
                if loss_mesh is not None:
                    torch.distributed.all_reduce(
                        loss_is_finite,
                        op=torch.distributed.ReduceOp.MIN,
                        group=loss_mesh.get_group(),
                    )
            pp_mesh = parallel_dims.get_optional_mesh("pp")
            if pp_mesh is not None:
                torch.distributed.all_reduce(
                    loss_is_finite,
                    op=torch.distributed.ReduceOp.MIN,
                    group=pp_mesh.get_group(),
                )

            step_is_finite = loss_is_finite.logical_and(torch.isfinite(grad_norm).all())
            # Keep the check and optimizer kernels ordered on the device without
            # synchronizing the host on every step. The RuntimeError is catchable
            # on CPU, while a failed CUDA assertion invalidates the process.
            torch._assert_async(
                step_is_finite,
                "Loss or gradient norm is not finite on at least one rank at "
                f"step {self.step}. Stopping training before the optimizer update.",
            )
            self.checkpointer.maybe_wait_for_staging()
            self.optimizers.step()
            self.lr_schedulers.step()

        # log metrics
        if not should_log:
            return

        assert accumulated_loss is not None

        with sl.log_trace_span("collect_dist_metrics"):
            sl.log_trace_scalar({"global_valid_tokens": int(global_valid_tokens)})

            if parallel_dims.dp_cp_enabled:
                # FT addition: use ft_manager.loss_sync_pg for extra process group
                ft_pg = self.ft_manager.loss_sync_pg
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
                    dist_utils.dist_sum(accumulated_loss, loss_mesh, ft_pg),
                    dist_utils.dist_max(local_avg_loss, loss_mesh, ft_pg),
                    dist_utils.dist_sum(
                        torch.tensor(
                            self.ntokens_seen, dtype=torch.int64, device=self.device
                        ),
                        loss_mesh,
                        ft_pg,
                    ),
                )
            else:
                global_avg_loss = global_max_loss = float(accumulated_loss.item())
                global_ntokens_seen = self.ntokens_seen

        extra_metrics = {
            "n_tokens_seen": global_ntokens_seen,
            **lr_metrics,
        }
        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            float(grad_norm.item()),
            extra_metrics=extra_metrics,
        )

    @record
    def train(self):
        config = self.config

        sl.log_trace_instant("training_start")

        self.checkpointer.load(step=config.checkpoint.load_step)

        # Capture loaded step for relative_step calculation.
        # After checkpoint load: self.step = restored step (e.g. 100), or 0 if fresh.
        loaded_step = self.step

        logger.info(f"Training starts at step {self.step + 1}")

        # FT addition: per-replica profiling leaf folder
        leaf_folder = (
            ""
            if not self.ft_manager.enabled
            else f"replica_{self.ft_manager.replica_id}"
        )
        # Semi-sync fragment replication is only implemented for the
        # non-pipeline case; under PP fall back to async-quorum training.
        if self.parallel_dims.pp_enabled and (
            getattr(config.fault_tolerance, "semi_sync_method", None) is not None
        ):
            logger.warning(
                "Semi-sync training (fault_tolerance.semi_sync_method) is not "
                "supported with pipeline parallelism; continuing without it."
            )
            semi_sync_ctx = nullcontext()
        else:
            # FT addition: maybe_semi_sync_training context manager
            semi_sync_ctx = maybe_semi_sync_training(
                config.fault_tolerance,
                ft_manager=self.ft_manager,
                model=self.model_parts[0],
                n_layers=(
                    len(self.model_config.layers)
                    if hasattr(self.model_config, "layers")
                    else 0
                ),
                optimizer=self.optimizers,
                fragment_fn=(
                    config.model_spec.fragment_fn
                    if hasattr(config.model_spec, "fragment_fn")
                    else None
                ),
            )
        with (
            config.profiler.build(
                global_step=self.step,
                base_folder=config.dump_folder,
                leaf_folder=leaf_folder,
            ) as profiler,
            semi_sync_ctx,
        ):
            data_iterator = self.batch_generator(self.dataloader)
            while self.should_continue_training():
                self.step += 1
                sl.set_step(self.step, relative_step=self.step - loaded_step)

                with sl.log_trace_span("step"):
                    self.gc_handler.run(self.step)

                    try:
                        self.train_step(data_iterator)
                    except DataloaderExhaustedError:
                        logger.warning("Ran out of data; last step was canceled.")
                        break

                    self.checkpointer.save(
                        self.step,
                        last_step=(self.step == config.training.steps),
                    )

                    # Run validation if validator is available
                    if self.config.validator.enable and self.validator.should_validate(
                        self.step
                    ):
                        self.validator.validate(self.model_parts, self.step)

                    # signal the profiler that the next profiling step has started
                    profiler.step()

                    # Reduce timeout after the first train step of THIS process
                    # (assuming lazy init and compilation are finished). Use the
                    # relative step so this fires on resumed runs too.
                    if self.step - loaded_step == 1:
                        dist_utils.set_pg_timeouts(
                            timeout=timedelta(
                                seconds=config.comm.train_timeout_seconds
                            ),
                            parallel_dims=self.parallel_dims,
                        )

        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)

        logger.info("Training completed")
