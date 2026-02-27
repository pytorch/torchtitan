# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import glob
import logging
import os
import shutil

from dataclasses import dataclass, field
from typing import List

import torch
from monarch.actor import Actor, endpoint
from safetensors.torch import save_file
from torchtitan.config import CommConfig, Configurable, ParallelismConfig
from torchtitan.distributed import utils as dist_utils

from torchtitan.experiments.rl.unified.configs import (
    PolicyOptimizationConfig,
    VLLMSamplingConfig,
)

from torchtitan.experiments.rl.vllm_compat.simple_rl import (
    compute_grpo_advantages,
    compute_grpo_advantages_stable,
    trivial_reward_function,
)
from torchtitan.protocols.model_spec import ModelSpec
from vllm import EngineArgs, LLMEngine, SamplingParams

from vllm.config import AttentionConfig
from vllm.model_executor.layers.batch_invariant import init_batch_invariance
from vllm.sampling_params import RequestOutputKind

from vllm.v1.attention.backends.registry import AttentionBackendEnum

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryData:
    """
    Data from one generation batch.

    Attributes:
        policy_version: Version of policy that produced this batch
        completions: List of completion strings
        vllm_token_ids: List of token ID lists for each completion
        vllm_token_log_probs: List of per-token log prob lists
        prompt_token_ids: List of prompt token ID lists
        rewards: Computed rewards for each completion
        advantages: Computed advantages for each completion
    """

    policy_version: int
    completions: list[str]
    vllm_token_ids: list[list[int]]
    vllm_token_log_probs: list[list[float]]
    prompt_token_ids: list[list[int]]
    rewards: torch.Tensor
    advantages: torch.Tensor


class Generator(Actor, Configurable):
    """
    Generates rollouts using vLLM engine.

    Maintains a vLLM engine that is synchronized with the Trainer
    via weight sync. Generates completions for given prompts and
    computes rewards/advantages.

    Args:
        config: Generator-specific configuration.
        model_path: Path to the HF model checkpoint.
        dump_folder: Root output folder for RL artifacts.
        batch_invariant_mode: Enable batch-invariant mode for deterministic ops.
        policy_optimization: GRPO hyperparameters.
        prompt_texts: List of prompt strings.
        expected_answers: List of expected answer strings.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Generator actor configuration."""

        dtype: str = "bfloat16"
        """Data type for model weights, passed directly to vLLM (auto, float16, bfloat16, float32)."""

        gpu_memory_limit: float = 0.5
        """Fraction of GPU memory to use for the vLLM engine (0.0 to 1.0)."""

        enforce_eager: bool = True
        """Disable CUDA graphs in vLLM (use eager execution)."""

        seed: int | None = None
        """Random seed for reproducible generation. None means no fixed seed."""

        parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
        """Parallelism configuration for the vLLM engine."""

        sampling: VLLMSamplingConfig = field(default_factory=VLLMSamplingConfig)
        """Default sampling parameters for generation."""

        vllm_attention_backend: str = "FLASH_ATTN"
        """vLLM attention backend to use (e.g., FLASH_ATTN, XFORMERS)."""

    def __init__(
        self,
        config: Config,
        *,
        model_spec: ModelSpec,
        model_path: str,
        dump_folder: str,
        batch_invariant_mode: bool,
        policy_optimization: PolicyOptimizationConfig,
        prompt_texts: list[str],
        expected_answers: list[str],
    ):
        self.config = config
        self.model_spec = model_spec

        # Register TorchTitan model with vLLM before any engine creation
        from torchtitan.experiments.rl.unified.plugin import (
            register_model_to_vllm_model_registry,
            VLLM_MODEL_NAME,
        )

        register_model_to_vllm_model_registry(model_spec)
        self._vllm_model_name = VLLM_MODEL_NAME

        # Set vLLM environment variables from config before any vLLM initialization
        if batch_invariant_mode:
            os.environ["VLLM_BATCH_INVARIANT"] = "1"
            init_batch_invariance(AttentionBackendEnum.FLASH_ATTN)

        os.environ["VLLM_ATTENTION_BACKEND"] = config.vllm_attention_backend

        self.prompt_texts = prompt_texts
        self.expected_answers = expected_answers

        # Extract needed fields from configs
        self.model_path = model_path
        self.max_new_tokens = config.sampling.max_tokens
        self.temperature = config.sampling.temperature
        self.group_size = policy_optimization.group_size
        self.grpo_beta = policy_optimization.beta
        self.use_stable_grpo = policy_optimization.use_stable_grpo

        # Initialize distributed environment for SPMD generator
        world_size = dist_utils.init_distributed(CommConfig())

        # Set up temp model directory for vLLM weight loading
        self._base_model_path = model_path
        self._temp_model_dir = os.path.abspath(
            os.path.join(dump_folder, "vllm_temp_model")
        )
        os.makedirs(self._temp_model_dir, exist_ok=True)

        # Copy config/tokenizer files from base model to temp dir
        for file in [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "merges.txt",
            "vocab.json",
        ]:
            src = os.path.join(model_path, file)
            if os.path.exists(src):
                shutil.copy2(src, self._temp_model_dir)

        # Copy the original model shard files if they exist
        # We'll overwrite these with our single model.safetensors later
        for shard_file in glob.glob(os.path.join(model_path, "model-*.safetensors")):
            dst = os.path.join(self._temp_model_dir, os.path.basename(shard_file))
            shutil.copy2(shard_file, dst)

        # Copy index file if it exists
        index_file = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index_file):
            shutil.copy2(index_file, self._temp_model_dir)

        self._engine: LLMEngine | None = None

        self.policy_version = 0

        # Reward function. TODO: Move reward calculation out of generator
        self.reward_fn = trivial_reward_function

        logger.debug("Generator initialized (vLLM engine will load on first use)")

    def _update_vllm_model_weights(self, vllm_state: dict) -> None:
        """
        Update vLLM model weights from vLLM model state dict. This function is used
        when updating vLLM model's weights from trainer's updated weights.

        Args:
            vllm_state: vLLM model state dict, a map from vLLM model's fqn names to weights
        """
        # Save to temp model directory
        checkpoint_path = os.path.join(self._temp_model_dir, "model.safetensors")

        # Update the shard files that vLLM will actually load
        # We need to split our weights to match the original 2-shard structure
        shard_files = sorted(
            glob.glob(os.path.join(self._temp_model_dir, "model-*.safetensors"))
        )
        index_file = os.path.join(self._temp_model_dir, "model.safetensors.index.json")

        # TODO: need to replace this with Torchtitan's checkpoint save and load
        # right now we hardcoded to work with 1 safetensor files which we only
        # tested on Qwen3 0.6B model. In the longer term, need to use TorchStore
        # to achieve the weight communication.
        if torch.distributed.get_rank() == 0:
            logger.debug(f"Saving weights to {checkpoint_path}")

            # TODO: Check the detail of vLLM's dtype conversion journey
            # Currently converting float32 to bfloat16 to match vLLM's attention and kv cache dtype
            vllm_state = {
                k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
                for k, v in vllm_state.items()
            }
            save_file(vllm_state, checkpoint_path)

        # Synchronize all ranks before reloading to ensure rank 0 finished writing
        torch.distributed.barrier()
        logger.debug(
            f"[Rank {torch.distributed.get_rank()}] Synchronized after weight save"
        )

        # First time: create the engine using LLMEngine and EngineArgs
        if self._engine is None:
            cfg = self.config

            engine_args = EngineArgs(
                # Model configuration
                model=self._temp_model_dir,
                trust_remote_code=True,
                dtype=cfg.dtype,
                # Parallelism configuration
                tensor_parallel_size=cfg.parallelism.tensor_parallel_degree,
                # Use external_launcher because Monarch already spawns the worker processes
                distributed_executor_backend="external_launcher",
                # Memory and performance
                gpu_memory_utilization=cfg.gpu_memory_limit,
                enforce_eager=cfg.enforce_eager,
                # Seed
                seed=cfg.seed,
                # HuggingFace overrides to use registered TorchTitan model
                hf_overrides={"architectures": [self._vllm_model_name]},
                attention_config=AttentionConfig(
                    backend=AttentionBackendEnum.FLASH_ATTN,
                ),
            )

            logger.debug("Initializing LLMEngine from EngineArgs...")
            self._engine = LLMEngine.from_engine_args(engine_args)
            logger.debug("Created new vLLM LLMEngine")
        else:
            # Direct parameter copy into model tensors.
            # This bypasses vLLM's reload_weights() which uses a layerwise
            # reload mechanism that moves params to meta device
            from torchtitan.experiments.rl.vllm_compat.weights import vllm_to_torchtitan

            titan_state = vllm_to_torchtitan(vllm_state)
            model = self._engine.model_executor.driver_worker.get_model()
            params = dict(model.named_parameters())

            for name, new_weight in titan_state.items():
                # TorchTitanVLLMModelWrapper stores the model as self.model,
                # so parameters have a "model." prefix
                param_name = f"model.{name}"
                if param_name in params:
                    param = params[param_name]
                    new_w = new_weight.to(device=param.device, dtype=param.dtype)
                    param.data.copy_(new_w)

    def _compute_rewards_and_advantages(
        self, completions: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute rewards and GRPO advantages for generated completions.
        TODO: Move this function out of generator for encapsulation.

        Args:
            completions: List of completion strings.

        Returns:
            rewards: Raw rewards tensor.
            advantages: GRPO advantage tensor.
        """
        logger.debug(
            f"Computing rewards: {len(completions)} completions, "
            f"{len(self.expected_answers)} expected answers, "
            f"group_size={self.group_size}"
        )
        rewards = self.reward_fn(completions, self.expected_answers, self.group_size)

        # Normalize rewards
        reward_mean = rewards.mean()
        reward_std = rewards.std()
        if reward_std > 1e-8:
            rewards_normalized = (rewards - reward_mean) / reward_std
        else:
            rewards_normalized = rewards - reward_mean

        # Compute advantages using GRPO
        if self.use_stable_grpo:
            advantages = compute_grpo_advantages_stable(
                rewards_normalized, self.group_size
            )
        else:
            advantages = compute_grpo_advantages(
                rewards_normalized, self.group_size, beta=self.grpo_beta
            )

        return rewards, advantages

    @endpoint
    async def generate(self) -> TrajectoryData:
        """Generate trajectories and compute rewards/advantages.
        Called by the orchestrator (simple_grpo.py).
        """
        logger.debug(
            f"{os.getpid()=} Generating start generate (policy v{self.policy_version})..."
        )

        with torch.no_grad():
            # Generate samples using vLLM
            sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                n=self.group_size,
                seed=self.config.seed,
                logprobs=1,
                prompt_logprobs=1,  # Also get prompt log probs to access prompt token IDs
                output_kind=RequestOutputKind.FINAL_ONLY,  # Only return completed outputs
            )

            for request_id, prompt in enumerate(self.prompt_texts):
                self._engine.add_request(str(request_id), prompt, sampling_params)

            # Step through engine until all requests are finished
            all_outputs = []
            while self._engine.has_unfinished_requests():
                request_outputs = self._engine.step()
                all_outputs.extend(request_outputs)

            # Extract completions and log probs
            completions = []
            token_ids_list = []
            token_log_probs_list = []
            prompt_token_ids_list = []

            for output in all_outputs:
                prompt_token_ids = output.prompt_token_ids

                for sample in output.outputs:
                    completions.append(sample.text)
                    prompt_token_ids_list.append(
                        prompt_token_ids
                    )  # Store prompt tokens for this sample
                    token_ids_list.append(
                        sample.token_ids
                    )  # Extract token IDs (generated tokens only)
                    per_token_log_probs = [
                        list(logprob_dict.values())[0].logprob
                        for logprob_dict in sample.logprobs
                    ]  # Extract per-token log probs
                    token_log_probs_list.append(per_token_log_probs)

        # Compute rewards and advantages
        rewards, advantages = self._compute_rewards_and_advantages(completions)

        # Create trajectory data
        trajectory = TrajectoryData(
            policy_version=self.policy_version,
            completions=completions,
            vllm_token_ids=token_ids_list,
            vllm_token_log_probs=token_log_probs_list,
            prompt_token_ids=prompt_token_ids_list,
            rewards=rewards,
            advantages=advantages,
        )

        logger.debug(
            f"{os.getpid()=} Generating finish generate (policy v{self.policy_version})..."
        )
        return trajectory

    @endpoint
    async def update(self, version: int, vllm_compat_state: dict) -> None:
        """Update generate weights.
        Called by the orchestrator (simple_grpo.py).

        Args:
            version: New policy version number
            vllm_compat_state: vLLM-compatible state dict
        """
        # TODO: remove the helper function (_update_vllm_model_weights) once we clean up the weight updates
        self._update_vllm_model_weights(vllm_compat_state)
        self.policy_version = version
        logger.debug(
            f"{os.getpid()=} Generator updating weights to policy v{version}..."
        )

    def __del__(self):
        """Cleanup vLLM engine."""
        if hasattr(self, "_engine"):
            del self._engine
            torch.cuda.empty_cache()
