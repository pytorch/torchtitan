# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torchstore as ts
from monarch.actor import Actor, endpoint
from torch.distributed.checkpoint.state_dict import (
    set_model_state_dict,
    StateDictOptions,
)
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import CommConfig, Configurable, TORCH_DTYPE_MAP
from torchtitan.config.configs import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.experiments.rl.actors.utils import (
    build_response_mask,
    extract_logprobs_batched,
    verify_logprob_identity,
)
from torchtitan.experiments.rl.types import TrainBatch
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools import utils

logger = logging.getLogger(__name__)


class PolicyTrainer(Actor, Configurable):
    """
    Updates policy based on collected Episode using TorchTitan components.

    Uses ModelSpec for model construction, parallelization, and weight loading.

    Args:
        config: PolicyTrainer.Config for model/optimizer/parallelism settings.
        model_spec: Model specification (model config, parallelize_fn, state_dict_adapter).
        hf_assets_path: Path to HF assets folder for checkpoint loading.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """PolicyTrainer configuration for optimizer, training, and parallelism."""

        optimizer: OptimizersContainer.Config = field(
            default_factory=OptimizersContainer.Config
        )
        lr_scheduler: LRSchedulersContainer.Config = field(
            default_factory=LRSchedulersContainer.Config
        )
        training: TrainingConfig = field(default_factory=TrainingConfig)
        parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
        comm: CommConfig = field(default_factory=CommConfig)
        """Communication configuration for distributed initialization."""
        compile: CompileConfig = field(default_factory=CompileConfig)

    def __init__(
        self,
        config: Config,
        *,
        loss_fn: Callable,
        model_spec: ModelSpec,
        batch_invariant_mode: bool,
        hf_assets_path: str = "",
        transfer_dtype: str = "",
    ):
        self.config = config
        self.loss_fn = loss_fn
        self.model_spec = model_spec
        # Only cast if transfer dtype differs from training dtype, otherwise
        # staging buffers would be allocated for a no-op cast.
        training_dtype = TORCH_DTYPE_MAP[config.training.dtype]
        requested = TORCH_DTYPE_MAP[transfer_dtype] if transfer_dtype else None
        self._transfer_dtype = requested if requested != training_dtype else None

        # The policy and ref models share code objects, so dynamo's
        # per-code-object cache must hold entries for both grad modes
        # (grad for policy, no_grad for ref). The default limit of 8
        # is not enough; 16 accommodates both without recompile storms.
        # TODO: @Lucaskabela fix recompiles in general as these increase startup
        torch._dynamo.config.cache_size_limit = 16

        # Device setup
        device_module, device_type = utils.device_module, utils.device_type
        self.device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
        device_module.set_device(self.device)

        world_size = dist_utils.init_distributed(config.comm)

        # Build parallel dims
        parallelism_config = config.parallelism
        self.parallel_dims = ParallelDims(
            dp_shard=parallelism_config.data_parallel_shard_degree,
            dp_replicate=parallelism_config.data_parallel_replicate_degree,
            cp=parallelism_config.context_parallel_degree,
            tp=parallelism_config.tensor_parallel_degree,
            pp=parallelism_config.pipeline_parallel_degree,
            ep=parallelism_config.expert_parallel_degree,
            etp=parallelism_config.expert_tensor_parallel_degree,
            world_size=world_size,
        )

        # Initialize state dict adapter for HF checkpoint loading
        if model_spec.state_dict_adapter is not None:
            self.sd_adapter = model_spec.state_dict_adapter(
                model_spec.model, hf_assets_path
            )
        else:
            self.sd_adapter = None

        # Create training policy model
        model = self._build_model(
            model_spec, config, device_type, batch_invariant_mode, hf_assets_path
        )
        model.train()
        self.model = model
        self.model_parts = [model]

        # Create reference model for KL divergence (frozen copy of initial policy)
        # TODO: Move ref_model to a separate actor so it can live on different GPUs
        ref_model = self._build_model(
            model_spec, config, device_type, batch_invariant_mode, hf_assets_path
        )
        for p in ref_model.parameters():
            p.requires_grad = False
        ref_model.eval()
        self.ref_model = ref_model

        # Build optimizer and LR scheduler
        self.optimizers = config.optimizer.build(model_parts=self.model_parts)
        self.lr_schedulers = config.lr_scheduler.build(
            optimizers=self.optimizers,
            training_steps=config.training.steps,
        )

        self.policy_version = 0
        self.generator: Any | None = None

        # Data parallelism: determine this rank's shard of the batch.
        self.dp_size = self.parallel_dims.dp_replicate * self.parallel_dims.dp_shard
        self.dp_rank = dist.get_rank() // self.parallel_dims.non_data_parallel_size
        self.dp_enabled = self.parallel_dims.dp_enabled

        logger.debug(
            f"PolicyTrainer initialized (dp_rank={self.dp_rank}, dp_size={self.dp_size})"
        )

    def _load_initial_hf_weights(self, model, checkpoint_path: str) -> None:
        """Load model weights from HF checkpoint using DCP and state_dict_adapter.

        Args:
            model: The model to load weights into.
            checkpoint_path: Path to HF checkpoint directory.
        """
        if self.sd_adapter is None:
            logger.warning(
                "No state_dict_adapter available, skipping initial weight load"
            )
            return

        if not os.path.isdir(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint path '{checkpoint_path}' does not exist. "
                "Please provide a valid path to a HuggingFace checkpoint directory."
            )

        storage_reader = self.sd_adapter.get_hf_storage_reader(checkpoint_path)
        hf_state_dict = self.sd_adapter.to_hf(model.state_dict())
        dcp.load(hf_state_dict, storage_reader=storage_reader)
        torchtitan_state_dict = self.sd_adapter.from_hf(hf_state_dict)

        set_model_state_dict(
            model=model,
            model_state_dict=torchtitan_state_dict,
            options=StateDictOptions(strict=True),
        )
        logger.info(
            f"Loaded initial weights from {checkpoint_path} "
            f"({len(torchtitan_state_dict)} parameters)"
        )

    def _build_model(
        self,
        model_spec: ModelSpec,
        config: Config,
        device_type: str,
        batch_invariant_mode: bool,
        hf_assets_path: str,
    ):
        """Build, parallelize, and initialize a model from checkpoint.
        Will be used to build trainer's policy model and reference model.

        Args:
            model_spec: Model specification for building and parallelizing.
            config: Trainer config (used for dtype, parallelism, checkpoint path, etc.).
            device_type: Device type string (e.g. "cuda").
            batch_invariant_mode: Whether to patch attention for vLLM compatibility.
            hf_assets_path: Path to HF assets folder for checkpoint loading.

        Returns:
            Initialized model with weights loaded from checkpoint.
        """

        # TODO Also support flex attention backend later.
        from torchtitan.models.common.attention import VarlenAttention

        assert isinstance(
            model_spec.model.layer.attention.inner_attention, VarlenAttention.Config
        ), "Only varlen attention backend is allowed."

        with torch.device("meta"):
            with utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]):
                model = model_spec.model.build()

        model = model_spec.parallelize_fn(
            model,
            parallel_dims=self.parallel_dims,
            parallelism=config.parallelism,
            compile_config=config.compile,
        )

        model.to_empty(device=device_type)
        with torch.no_grad():
            model.init_weights(buffer_device=None)

        # Load initial weights from HF
        self._load_initial_hf_weights(model, hf_assets_path)

        return model

    @endpoint
    async def push_model_state_dict(self) -> None:
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
            "model_state_dict",
            direct_rdma=is_rdma_available(),
            transfer_dtype=self._transfer_dtype,
        )

    @endpoint
    async def step(self, batches: list[TrainBatch]) -> dict:
        """Perform one training step.

        Expects one pre-collated batch per DP rank.

        Args:
            batches: Pre-sharded batches, one per DP rank.

        Returns:
            Training metrics
        """
        logger.debug(
            f"{os.getpid()=} PolicyTrainer starting step {self.policy_version} "
        )

        local_batch = batches[self.dp_rank]
        device = self.device

        token_ids = local_batch.token_ids.to(device)
        prompt_lens = local_batch.prompt_lens.to(device)
        response_lens = local_batch.response_lens.to(device)
        advantages = local_batch.advantages.to(device)

        logits = self.model(token_ids)
        policy_logprobs = extract_logprobs_batched(logits, token_ids)

        with torch.no_grad():
            ref_logits = self.ref_model(token_ids)
            ref_logprobs = extract_logprobs_batched(ref_logits, token_ids)

        response_mask = build_response_mask(
            prompt_lens,
            response_lens,
            token_ids.shape[1],
            device,
        )
        loss, loss_metrics = self.loss_fn(
            policy_logprobs=policy_logprobs,
            response_mask=response_mask,
            advantages=advantages,
            ref_logprobs=ref_logprobs,
        )

        rollout_log_probs = []
        batch_token_log_probs = []
        for i in range(token_ids.shape[0]):
            start = prompt_lens[i].item()
            end = start + response_lens[i].item()
            rollout_log_probs.append(local_batch.token_log_probs[i, start:end].tolist())
            batch_token_log_probs.append(policy_logprobs[i, start:end])

        # Verify logprob identity (local shard)
        verification_result = verify_logprob_identity(
            rollout_log_probs,
            batch_token_log_probs,
        )
        logger.debug(
            f"Logprob verification: bitwise_identical={verification_result['logprob_bitwise_identical']}, "
            f"max_delta={verification_result['logprob_max_delta']:.6e}, "
            f"diff_mean={verification_result['logprob_diff_mean']:.6e}, "
            f"diff_max={verification_result['logprob_diff_max']:.6e}, "
            f"tokens_checked={verification_result['total_tokens_checked']}"
        )

        # Update weights
        self.optimizers.zero_grad()
        loss.backward()

        # All-reduce gradients across DP ranks so all ranks have consistent
        # weight updates despite processing different data shards.
        if self.dp_enabled:
            for param in self.model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)

        # Gradient clipping
        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=self.parallel_dims.get_optional_mesh("pp"),
        )

        self.optimizers.step()
        self.lr_schedulers.step()

        self.policy_version += 1

        # Return metrics
        metrics = {
            "loss": loss.item(),
            "advantage_mean": advantages.mean().item(),
            "advantage_std": advantages.std().item(),
            "policy_version": self.policy_version,
            "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else grad_norm,
            # Trainer vs generator log prob divergence
            "logprob_diff_mean": verification_result["logprob_diff_mean"],
            "logprob_diff_max": verification_result["logprob_diff_max"],
            "logprob_max_delta": verification_result["logprob_max_delta"],
            "logprob_bitwise_identical": verification_result[
                "logprob_bitwise_identical"
            ],
            **loss_metrics,
        }
        logger.debug(
            f"{os.getpid()=} PolicyTrainer finished step {self.policy_version}"
        )
        return metrics
