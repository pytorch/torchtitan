# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import torch
import torchstore as ts
from monarch.actor import Actor, endpoint
from torch.distributed.checkpoint.stateful import Stateful
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import CommConfig, Configurable, TORCH_DTYPE_MAP
from torchtitan.config.configs import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.experiments.rl.types import (
    ForwardBackwardResult,
    OptimStepResult,
    TrainBatch,
)
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools import utils

logger = logging.getLogger(__name__)


class PolicyTrainer(Actor, Stateful, Configurable):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Trainer configuration for optimizer, training, and parallelism."""

        model_spec: ModelSpec | None = None
        hf_assets_path: str = ""
        dump_folder: str = "./outputs/rl"
        optimizer: OptimizersContainer.Config = field(
            default_factory=OptimizersContainer.Config
        )
        lr_scheduler: LRSchedulersContainer.Config = field(
            default_factory=LRSchedulersContainer.Config
        )
        training: TrainingConfig = field(default_factory=TrainingConfig)
        parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
        compile: CompileConfig = field(default_factory=CompileConfig)
        comm: CommConfig = field(default_factory=CommConfig)
        checkpoint: CheckpointManager.Config = field(
            default_factory=CheckpointManager.Config
        )

    weight_sync_transfer_dtype: Literal["bfloat16", "float32"] = "bfloat16"
    _cast_dtypes_for_weight_sync: bool = False

    def __init__(
        self,
        config: Config,
        *,
        loss_fn: Callable,
    ):
        self.config = config
        self.loss_fn = loss_fn

        assert (
            config.model_spec is not None
        ), "model_spec must be set before creating Trainer"
        model_spec = config.model_spec

        device_module, device_type = utils.device_module, utils.device_type
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        device_module.set_device(self.device)

        self.parallel_dims = parallel_dims = self.init_distributed()

        if parallel_dims.dp_enabled:
            batch_mesh = parallel_dims.get_mesh("batch")
            self.dp_degree = batch_mesh.size()
            self.dp_rank = batch_mesh.get_local_rank()
        else:
            self.dp_degree = 1
            self.dp_rank = 0

        # Build model (meta init -> parallelize -> materialize -> init weights)
        model_config = model_spec.model
        model_config.update_from_config(trainer_config=config)
        self.model_config = model_config

        with torch.device("meta"):
            with utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]):
                model = model_config.build()

        model = model_spec.parallelize_fn(
            model,
            parallel_dims=parallel_dims,
            parallelism=config.parallelism,
            compile_config=config.compile,
        )

        model.to_empty(device=device_type)
        with torch.no_grad():
            model.init_weights(buffer_device=None)
        model.train()

        self.model = model
        self.model_parts = [model]

        # Build optimizer and LR scheduler (after parallelize, matches main Trainer)
        self.optimizers = config.optimizer.build(model_parts=self.model_parts)
        self.lr_schedulers = config.lr_scheduler.build(
            optimizers=self.optimizers,
            training_steps=config.training.steps,
        )

        # RL-specific state
        self.policy_version = 0

        # Checkpoint manager: handles both HF initial load and DCP save/load
        sd_adapter = (
            model_spec.state_dict_adapter(model_config, config.hf_assets_path)
            if model_spec.state_dict_adapter
            else None
        )
        self.checkpointer = config.checkpoint.build(
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states={"train_state": self},
            sd_adapter=sd_adapter,
            base_folder=config.dump_folder,
        )
        self.checkpointer.load()

        # Weight sync dtype: only cast if different from training dtype
        if self.weight_sync_transfer_dtype != config.training.dtype:
            self._cast_dtypes_for_weight_sync = True

        logger.info(
            f"PolicyTrainer initialized "
            f"(dp_rank={self.dp_rank}, dp_degree={self.dp_degree}, "
            f"total steps={config.training.steps}, "
            f"warmup={config.lr_scheduler.warmup_steps})"
        )

    def init_distributed(self) -> ParallelDims:
        config = self.config
        world_size = dist_utils.init_distributed(config.comm)
        parallelism_config = config.parallelism
        return ParallelDims(
            dp_shard=parallelism_config.data_parallel_shard_degree,
            dp_replicate=parallelism_config.data_parallel_replicate_degree,
            cp=parallelism_config.context_parallel_degree,
            tp=parallelism_config.tensor_parallel_degree,
            pp=parallelism_config.pipeline_parallel_degree,
            ep=parallelism_config.expert_parallel_degree,
            etp=parallelism_config.expert_tensor_parallel_degree,
            world_size=world_size,
        )

    @endpoint
    async def get_dp_info(self) -> dict:
        """Return DP topology so the controller can collate correctly."""
        return {
            "dp_degree": self.dp_degree,
            "dp_rank": self.dp_rank,
        }

    @endpoint
    async def forward_backward(
        self, batches: list[TrainBatch]
    ) -> ForwardBackwardResult:
        """Run model forward, compute loss, and backpropagate.

        Runs a standard [B, L] forward pass with causal SDPA attention,
        extracts per-token logprobs, builds a response mask, and delegates
        to the loss function.

        Args:
            batches: Pre-sharded list of TrainBatch, one per DP rank.
                The controller's collate function is responsible for splitting
                data across DP ranks. Each rank indexes by dp_rank.

        Returns:
            ForwardBackwardResult with loss and metrics.
        """
        from torchtitan.experiments.rl.actors.utils import (
            build_response_mask,
            extract_logprobs_batched,
        )

        fwd_bwd_start = time.perf_counter()
        self.optimizers.zero_grad()

        local_batch = batches[self.dp_rank]
        device = next(self.model.parameters()).device

        token_ids = local_batch.token_ids.to(device)  # [B, L]
        logits = self.model(token_ids)  # [B, L, V]

        policy_logprobs = extract_logprobs_batched(logits, token_ids)  # [B, L]
        response_mask = build_response_mask(
            local_batch.prompt_lens.to(device),
            local_batch.response_lens.to(device),
            token_ids.shape[1],
            device,
        )

        loss, loss_metrics = self.loss_fn(
            policy_logprobs=policy_logprobs,
            response_mask=response_mask,
            **vars(local_batch),
        )

        loss.backward()

        logger.debug(
            f"{os.getpid()=} PolicyTrainer finished forward_backward "
            f"for (step {self.policy_version}) in "
            f"{time.perf_counter() - fwd_bwd_start:.3f}s"
        )

        return ForwardBackwardResult(
            loss=loss.item(),
            metrics=loss_metrics,
        )

    @endpoint
    async def optim_step(self) -> OptimStepResult:
        """Clip gradients, step optimizer, update LR scheduler.

        Must be called after forward_backward(). Gradients persist on the
        model parameters until this method consumes them.
        """
        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=self.parallel_dims.get_optional_mesh("pp"),
        )

        self.optimizers.step()
        self.lr_schedulers.step()

        self.policy_version += 1

        return OptimStepResult(
            grad_norm=(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm),
            policy_version=self.policy_version,
        )

    @endpoint
    async def push_weights(self) -> None:
        """Publish model weights for generator consumption via TorchStore.

        When ``direct_rdma=True``, weights are transferred directly from
        GPU to GPU via one-sided RDMA reads, bypassing StorageVolumes
        entirely. When ``False``, data goes through StorageVolumes
        (which may themselves use RDMA as a transport internally).

        Note: we couple ``is_rdma_available()`` with ``direct_rdma`` here,
        but the two concepts are not identical — StorageVolumes can also
        use RDMA as their transport layer. ``direct_rdma`` specifically
        means "skip StorageVolumes and let the destination read directly
        from the source's GPU memory".
        """
        from monarch.rdma import is_rdma_available

        await ts.put_state_dict(
            self.model.state_dict(),
            key="model_state_dict",
            direct_rdma=is_rdma_available(),
            transfer_dtype=TORCH_DTYPE_MAP[self.weight_sync_transfer_dtype]
            if self._cast_dtypes_for_weight_sync
            else None,
        )

    def state_dict(self) -> dict:
        """Mostly stubs for now to satisfy ``Stateful`` class"""
        return {"policy_version": self.policy_version}

    def load_state_dict(self, state_dict: dict) -> None:
        self.policy_version = state_dict["policy_version"]
