# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Generator

import torch
from torch.distributed.elastic.multiprocessing.errors import record

import torchtitan.protocols.train_spec as train_spec_module
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import rescale_accumulated_loss
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.protocols import BaseModelArgs
from torchtitan.tools import utils

from .job_config import ForgeJobConfig
from .train_spec import ForgeTrainSpec, get_train_spec


class ForgeEngine(torch.distributed.checkpoint.stateful.Stateful):
    # core configs
    job_config: ForgeJobConfig
    parallel_dims: ParallelDims
    train_spec: ForgeTrainSpec

    # swappable training components in ForgeTrainSpec
    model_parts: list[torch.nn.Module]
    loss_fn: train_spec_module.LossFunction
    optimizers: train_spec_module.OptimizersContainer
    lr_schedulers: train_spec_module.LRSchedulersContainer

    # non-swappable training components
    checkpointer: CheckpointManager

    # runtime utilities
    device: torch.device
    gc_handler: utils.GarbageCollection
    gradient_accumulation_steps: int
    train_context: Generator[None, None, None]
    pp_has_first_stage: bool
    pp_has_last_stage: bool

    # Fields in ForgeEngine which are not in original Trainer
    # for dataloading
    dp_degree: int
    dp_rank: int
    # for logging
    model_args: BaseModelArgs
    num_flops_per_token: float
    model_param_count: int
    global_batch_size: int

    # Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
    @record
    def __init__(self, job_config: ForgeJobConfig):
        torch._C._log_api_usage_once("torchtitan.train")

        self.job_config = job_config

        device_module, device_type = utils.device_module, utils.device_type
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        # Device has to be set before creating TorchFT manager.
        device_module.set_device(self.device)

        # init distributed and build meshes
        dist_utils.init_distributed(
            job_config.comm,
            enable_cpu_backend=job_config.training.enable_cpu_offload,
        )
        world_size = int(os.environ["WORLD_SIZE"])
        parallelism_config = job_config.parallelism
        self.parallel_dims = parallel_dims = ParallelDims(
            dp_shard=parallelism_config.data_parallel_shard_degree,
            dp_replicate=parallelism_config.data_parallel_replicate_degree,
            cp=parallelism_config.context_parallel_degree,
            tp=parallelism_config.tensor_parallel_degree,
            pp=parallelism_config.pipeline_parallel_degree,
            ep=parallelism_config.expert_parallel_degree,
            world_size=world_size,
        )

        world_mesh = parallel_dims.world_mesh
        if parallel_dims.dp_enabled:
            dp_mesh = world_mesh["dp"]
            dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
        else:
            dp_degree, dp_rank = 1, 0
        self.dp_degree, self.dp_rank = dp_degree, dp_rank

        # take control of garbage collection to avoid stragglers
        self.gc_handler = utils.GarbageCollection(
            gc_freq=job_config.training.gc_freq, debug=job_config.training.gc_debug
        )

        # Set random seed, and maybe enable deterministic mode
        # (mainly for debugging, expect perf loss).
        dist_utils.set_determinism(
            world_mesh,
            self.device,
            job_config.training.seed,
            job_config.training.deterministic,
        )
        self.train_spec = get_train_spec(job_config.model.name)

        # build model (using meta init)
        self.model_args = model_args = self.train_spec.model_args[
            job_config.model.flavor
        ]
        # set the model args from training job configs
        model_args.update_from_config(job_config)

        with torch.device("meta"):
            model = self.train_spec.model_cls(model_args)

        # calculate model size and flops per token
        (
            self.model_param_count,
            self.num_flops_per_token,
        ) = model_args.get_nparams_and_flops(model, job_config.training.seq_len)

        # move sharded model to CPU/GPU and initialize weights via DTensor
        if job_config.training.enable_cpu_offload:
            init_device = "cpu"
            buffer_device = device_type
        else:
            init_device = device_type
            buffer_device = None

        self.loss_fn = self.train_spec.build_loss_fn(job_config)

        # verify batch sizes
        global_batch_size = job_config.training.global_batch_size
        if global_batch_size < 0:
            # This global batch size results in 1 gradient accumulation
            # step.
            global_batch_size = job_config.training.local_batch_size * dp_degree
        assert global_batch_size > 0
        assert (
            global_batch_size % (job_config.training.local_batch_size * dp_degree) == 0
        ), (
            f"global batch size must be multiple of local batch size times "
            f"data-parallel degree ({global_batch_size} "
            f"% ({job_config.training.local_batch_size} * {dp_degree}) != 0)"
        )
        self.global_batch_size = global_batch_size

        # calculate gradient accumulation steps
        self.gradient_accumulation_steps = global_batch_size // (
            job_config.training.local_batch_size * dp_degree
        )
        assert self.gradient_accumulation_steps > 0
        self.loss_fn = rescale_accumulated_loss(
            self.loss_fn, self.gradient_accumulation_steps
        )

        # apply parallelisms and initialization
        if parallel_dims.pp_enabled:
            if not self.train_spec.pipelining_fn:
                raise RuntimeError(
                    f"Pipeline Parallel is enabled but {self.train_spec.name} "
                    f"does not support pipelining"
                )

            # apply both PT-D Pipeline Parallel and SPMD-style PT-D techniques
            (
                self.pp_schedule,
                self.model_parts,
                self.pp_has_first_stage,
                self.pp_has_last_stage,
            ) = self.train_spec.pipelining_fn(
                model,
                parallel_dims,
                job_config,
                self.device,
                model_args,
                self.train_spec.parallelize_fn,
                self.loss_fn,
            )
            # when PP is enabled, `model` obj is no longer used after this point,
            # model_parts is used instead
            del model

            for m in self.model_parts:
                m.to_empty(device=init_device)
                with torch.no_grad():
                    m.init_weights(buffer_device=buffer_device)
                m.train()
        else:
            # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
            model = self.train_spec.parallelize_fn(model, parallel_dims, job_config)

            model.to_empty(device=init_device)
            with torch.no_grad():
                model.init_weights(buffer_device=buffer_device)
            model.train()

            self.model_parts = [model]

        # build optimizer after applying parallelisms to the model
        self.optimizers = self.train_spec.build_optimizers_fn(
            self.model_parts, job_config.optimizer, parallel_dims
        )
        self.lr_schedulers = self.train_spec.build_lr_schedulers_fn(
            self.optimizers, job_config.lr_scheduler, job_config.training.steps
        )

        self.checkpointer = CheckpointManager(
            dataloader=None,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={"train_state": self},
            checkpoint_config=job_config.checkpoint,
            sd_adapter=(
                self.train_spec.state_dict_adapter(model_args)
                if self.train_spec.state_dict_adapter
                else None
            ),
        )

        loss_parallel_enabled = (
            parallel_dims.tp_enabled and not parallelism_config.disable_loss_parallel
        )
        self.train_context = dist_utils.get_train_context(
            loss_parallel_enabled,
            parallelism_config.enable_compiled_autograd,
        )
        self.maybe_enable_amp = dist_utils.maybe_enable_amp(
            parallel_dims,
            job_config.training.mixed_precision_param,
            device_type,
        )

    def close(self) -> None:
        if self.checkpointer:
            self.checkpointer.close()
