# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from monarch.actor import Actor, endpoint
from torch.distributed._tensor import DTensor
from torch.distributed.checkpoint.state_dict import (
    set_model_state_dict,
    StateDictOptions,
)
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import CommConfig, Configurable, TORCH_DTYPE_MAP
from torchtitan.config.configs import ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.experiments.rl.unified.actors.utils import (
    compute_policy_gradient_loss,
    compute_token_log_probs,
    verify_logprob_identity,
)
from torchtitan.experiments.rl.unified.models.attention import (
    replace_with_vllm_compatible_flash_attention,
)
from torchtitan.experiments.rl.unified.types import Episode
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

    def __init__(
        self,
        config: Config,
        *,
        model_spec: ModelSpec,
        batch_invariant_mode: bool,
        hf_assets_path: str = "",
    ):
        self.config = config
        self.model_spec = model_spec

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
        with torch.device("meta"):
            with utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]):
                model = model_spec.model.build()

        # Replace attention with vLLM compatible attention for RL training.
        # NOTE: Long-term this will be replaced by pytorch attention
        # supporting paged attention / kv cache.
        if batch_invariant_mode:
            replace_with_vllm_compatible_flash_attention(
                model, tp_size=self.parallel_dims.tp
            )

        model = model_spec.parallelize_fn(
            model,
            parallel_dims=self.parallel_dims,
            parallelism=config.parallelism,
        )

        model.to_empty(device=device_type)
        with torch.no_grad():
            model.init_weights(buffer_device=None)

        # Load initial weights from HF
        self._load_initial_hf_weights(model, hf_assets_path)

        return model

    @endpoint
    async def get_weights(self) -> dict:
        """Get model weights for generator.

        Returns:
            model state dict with plain local tensors (DTensors unwrapped
            to avoid cross-mesh issues when transferring through Monarch).
        """
        titan_state = self.model.state_dict()

        return {
            k: v.full_tensor() if isinstance(v, DTensor) else v
            for k, v in titan_state.items()
        }

    @endpoint
    async def step(self, episodes: list[Episode]) -> dict:
        """Perform one training step.

        Expects a flat list of Episodes with ``advantage`` already computed
        by the controller. Shards episodes across DP ranks so each rank
        processes a unique slice of the data.

        Args:
            episodes: Flat list of Episodes with advantages set.

        Returns:
            Training metrics
        """
        logger.debug(
            f"{os.getpid()=} PolicyTrainer starting step {self.policy_version} "
        )

        advantages = torch.tensor([ep.advantage for ep in episodes])

        all_token_ids: list[list[int]] = [ep.token_ids for ep in episodes]
        all_prompt_token_ids: list[list[int]] = [ep.prompt_token_ids for ep in episodes]
        all_token_log_probs: list[list[float]] = [ep.token_log_probs for ep in episodes]

        all_rewards_tensor = torch.tensor([ep.reward for ep in episodes])

        # Shard flattened completions across DP ranks so each rank processes
        # a unique subset of the data.
        total_samples = len(all_token_ids)
        my_indices = list(range(self.dp_rank, total_samples, self.dp_size))
        my_token_ids = [all_token_ids[i] for i in my_indices]
        my_prompt_token_ids = [all_prompt_token_ids[i] for i in my_indices]
        my_token_log_probs = [all_token_log_probs[i] for i in my_indices]
        my_advantages = advantages[my_indices]

        # Compute reference log probs using frozen ref_model (local shard only)
        ref_token_log_probs = []
        device = next(self.model.parameters()).device
        with torch.no_grad():
            for prompt_toks, gen_toks in zip(my_prompt_token_ids, my_token_ids):
                token_lps = compute_token_log_probs(
                    self.ref_model, prompt_toks, gen_toks, device
                )
                ref_token_log_probs.append(token_lps)

        # Compute loss on this rank's shard
        loss, loss_metrics, batch_token_log_probs = compute_policy_gradient_loss(
            self.model,
            my_token_ids,
            my_prompt_token_ids,
            my_advantages,
            ref_token_log_probs,
            kl_coef=0.1,
        )

        # Verify logprob identity (local shard)
        verification_result = verify_logprob_identity(
            my_token_log_probs,
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
            "reward_mean": all_rewards_tensor.mean().item(),
            "reward_std": all_rewards_tensor.std().item(),
            "advantage_mean": advantages.mean().item(),
            "advantage_std": advantages.std().item(),
            "sample_completion": episodes[0].text[:80],
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
