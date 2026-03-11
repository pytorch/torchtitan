# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections.abc import Generator
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch.distributed.elastic.multiprocessing.errors import record

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import LossFunction
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import Configurable, TORCH_DTYPE_MAP
from torchtitan.config.configs import (
    ActivationCheckpointConfig,
    CommConfig,
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.protocols import BaseModel
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools import utils


class ForgeEngine(torch.distributed.checkpoint.stateful.Stateful, Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        hf_assets_path: str = "./tests/assets/tokenizer"
        dump_folder: str = "./outputs"
        model_spec: ModelSpec = field(default_factory=ModelSpec)
        optimizer: OptimizersContainer.Config = field(
            default_factory=OptimizersContainer.Config
        )
        lr_scheduler: LRSchedulersContainer.Config = field(
            default_factory=LRSchedulersContainer.Config
        )
        training: TrainingConfig = field(default_factory=TrainingConfig)
        parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
        checkpoint: CheckpointManager.Config = field(
            default_factory=CheckpointManager.Config
        )
        activation_checkpoint: ActivationCheckpointConfig = field(
            default_factory=ActivationCheckpointConfig
        )
        compile: CompileConfig = field(default_factory=CompileConfig)
        model_converters: ModelConvertersContainer.Config = field(
            default_factory=ModelConvertersContainer.Config
        )
        comm: CommConfig = field(default_factory=CommConfig)
        debug: DebugConfig = field(default_factory=DebugConfig)

        def to_dict(self) -> dict[str, Any]:
            return asdict(self)

    # core configs
    config: Config
    parallel_dims: ParallelDims
    train_spec: ModelSpec

    # swappable training components in ModelSpec
    model_parts: list[torch.nn.Module]
    loss_fn: LossFunction
    optimizers: OptimizersContainer
    lr_schedulers: LRSchedulersContainer

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
    model_config: BaseModel.Config
    num_flops_per_token: float
    model_param_count: int
    global_batch_size: int

    # Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
    @record
    def __init__(self, config: Config):
        torch._C._log_api_usage_once("torchtitan.train")

        self.config = config

        device_module, device_type = utils.device_module, utils.device_type
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        # Device has to be set before creating TorchFT manager.
        device_module.set_device(self.device)

        # init distributed and build meshes
        dist_utils.init_distributed(
            config.comm,
            enable_cpu_backend=config.training.enable_cpu_offload,
        )
        world_size = int(os.environ["WORLD_SIZE"])
        parallelism_config = config.parallelism
        self.parallel_dims = parallel_dims = ParallelDims(
            dp_shard=parallelism_config.data_parallel_shard_degree,
            dp_replicate=parallelism_config.data_parallel_replicate_degree,
            cp=parallelism_config.context_parallel_degree,
            tp=parallelism_config.tensor_parallel_degree,
            pp=parallelism_config.pipeline_parallel_degree,
            ep=parallelism_config.expert_parallel_degree,
            etp=parallelism_config.expert_tensor_parallel_degree,
            world_size=world_size,
        )

        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            dp_degree, dp_rank = batch_mesh.size(), batch_mesh.get_local_rank()
        else:
            dp_degree, dp_rank = 1, 0
        self.dp_degree, self.dp_rank = dp_degree, dp_rank

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
            distinct_seed_mesh_dims=["pp"],  # same as `torchtitan/train.py`
        )
        self.train_spec = config.model_spec

        # build model (using meta init)
        self.model_config = model_config = self.train_spec.model
        # set the model args from training configs
        model_config.update_from_config(
            trainer_config=config,
        )

        with (
            torch.device("meta"),
            utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]),
        ):
            model = model_config.build()

        # calculate model size and flops per token
        (
            self.model_param_count,
            self.num_flops_per_token,
        ) = model_config.get_nparams_and_flops(model, config.training.seq_len)

        # move sharded model to CPU/GPU and initialize weights via DTensor
        if config.training.enable_cpu_offload:
            init_device = "cpu"
            buffer_device = device_type
        else:
            init_device = device_type
            buffer_device = None

        self.loss_fn = self.train_spec.build_loss_fn(
            config.compile, parallel_dims=parallel_dims
        )

        # verify batch sizes
        global_batch_size = config.training.global_batch_size
        if global_batch_size < 0:
            # This global batch size results in 1 gradient accumulation
            # step.
            global_batch_size = config.training.local_batch_size * dp_degree
        assert global_batch_size > 0
        assert (
            global_batch_size % (config.training.local_batch_size * dp_degree) == 0
        ), (
            f"global batch size must be multiple of local batch size times "
            f"data-parallel degree ({global_batch_size} "
            f"% ({config.training.local_batch_size} * {dp_degree}) != 0)"
        )
        self.global_batch_size = global_batch_size

        # calculate gradient accumulation steps
        self.gradient_accumulation_steps = global_batch_size // (
            config.training.local_batch_size * dp_degree
        )
        assert self.gradient_accumulation_steps > 0

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
                parallel_dims=parallel_dims,
                training=config.training,
                model_converters=config.model_converters,
                parallelism=config.parallelism,
                compile_config=config.compile,
                ac_config=config.activation_checkpoint,
                dump_folder=config.dump_folder,
                device=self.device,
                model_config=model_config,
                parallelize_fn=self.train_spec.parallelize_fn,
                loss_fn=self.loss_fn,
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
            model = self.train_spec.parallelize_fn(
                model,
                parallel_dims=parallel_dims,
                training=config.training,
                model_converters=config.model_converters,
                parallelism=config.parallelism,
                compile_config=config.compile,
                ac_config=config.activation_checkpoint,
                dump_folder=config.dump_folder,
            )

            model.to_empty(device=init_device)
            with torch.no_grad():
                model.init_weights(buffer_device=buffer_device)
            model.train()

            self.model_parts = [model]

        # build optimizer after applying parallelisms to the model
        self.optimizers = config.optimizer.build(
            model_parts=self.model_parts,
        )
        if self.train_spec.post_optimizer_build_fn is not None:
            self.train_spec.post_optimizer_build_fn(
                self.optimizers, self.model_parts, parallel_dims
            )
        self.lr_schedulers = config.lr_scheduler.build(
            optimizers=self.optimizers,
            training_steps=config.training.steps,
        )

        self.checkpointer = config.checkpoint.build(
            dataloader=None,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={"train_state": self},
            sd_adapter=(
                self.train_spec.state_dict_adapter(model_config, config.hf_assets_path)
                if self.train_spec.state_dict_adapter
                else None
            ),
            base_folder=config.dump_folder,
        )

        loss_parallel_enabled = (
            parallel_dims.tp_enabled and not parallelism_config.disable_loss_parallel
        )
        self.train_context = dist_utils.get_train_context(loss_parallel_enabled)
        self.maybe_enable_amp = dist_utils.maybe_enable_amp(
            parallel_dims,
            config.training.mixed_precision_param,
            device_type,
        )

    def close(self) -> None:
        if self.checkpointer:
            self.checkpointer.close()
