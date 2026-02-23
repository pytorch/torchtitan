# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import os

from dataclasses import dataclass, field
from typing import List

import torch
from monarch.actor import Actor, endpoint

# TODO: Replace with ``from torchtitan.config.configs import ParallelismConfig``
# once the config branch lands.
from torchtitan.config.job_config import Parallelism as ParallelismConfig
from torchtitan.distributed import utils as dist_utils

from torchtitan.experiments.rl.unified.configs import RLTrainer, VLLMSamplingConfig

# TODO: Replace with ``from torchtitan.config import Configurable``
# once the config branch lands.
from torchtitan.experiments.rl.unified.configurable import Configurable
from torchtitan.experiments.rl.vllm_compat.simple_rl import (
    compute_grpo_advantages,
    compute_grpo_advantages_stable,
    trivial_reward_function,
)
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
    completions: List[str]
    vllm_token_ids: List[List[int]]
    vllm_token_log_probs: List[List[float]]
    prompt_token_ids: List[List[int]]
    rewards: torch.Tensor
    advantages: torch.Tensor


class VLLMEngine(Configurable):
    """
    vLLM engine for fast rollouts with weight updates.

    The engine is created at initialization time from the model path.
    Subsequent weight updates are done in-place via
    ``load_weights_from_state_dict``.

    Constructed via ``config.build(model_path=..., dump_folder=...)``
    which calls ``VLLMEngine(config=..., model_path=..., dump_folder=...)``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """vLLM engine configuration for rollout generation."""

        dtype: str = "bfloat16"
        """Data type for model weights (auto, float16, bfloat16, float32)."""

        gpu_memory_limit: float = 0.5
        """Fraction of GPU memory to use for the vLLM engine (0.0 to 1.0)."""

        enforce_eager: bool = True
        """Disable CUDA graphs in vLLM (use eager execution)."""

        seed: int = 42
        """Random seed for reproducible generation."""

        parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
        """Parallelism configuration for the vLLM engine."""

        sampling: VLLMSamplingConfig = field(default_factory=VLLMSamplingConfig)
        """Default sampling parameters for generation."""

    def __init__(
        self,
        config: Config,
        *,
        model_path: str,
        dump_folder: str,
    ) -> None:
        self.config = config
        self.base_model_path = model_path

        cfg = self.config
        engine_args = EngineArgs(
            # Model configuration
            model=model_path,
            trust_remote_code=True,
            dtype=cfg.dtype,
            # Parallelism configuration
            tensor_parallel_size=cfg.parallelism.tensor_parallel_degree,
            distributed_executor_backend="external_launcher",
            # Memory and performance
            gpu_memory_utilization=cfg.gpu_memory_limit,
            enforce_eager=cfg.enforce_eager,
            # Seed
            seed=cfg.seed,
            # HuggingFace overrides to use TorchTitan model.
            # TODO: make this field configurable and align with model registration
            hf_overrides={"architectures": ["Qwen3TorchTitanForCausalLM"]},
            attention_config=AttentionConfig(
                backend=AttentionBackendEnum.FLASH_ATTN,
            ),
        )

        logger.info("Initializing LLMEngine from EngineArgs...")
        self.engine = LLMEngine.from_engine_args(engine_args)
        logger.info("vLLM rollout engine initialized")

    def _get_model(self):
        """Access the model from the vLLM engine.
        Returns a TorchTitanVLLMModelWrapper instance.
        """
        return self.engine.model_executor.driver_worker.get_model()

    def update_weights(self, state_dict: dict) -> None:
        """
        Update vLLM model weights in-place from a state dict.

        Args:
            state_dict: Model state dict to load
        """
        load_weights = self._get_model().load_weights_from_state_dict(state_dict)
        logger.info(
            f"Updated weights into vLLM engine model. "
            f"Number of parameters: {len(load_weights)}"
        )

    @torch.no_grad()
    def generate(
        self,
        prompt_texts: list[str],
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        n_samples_per_prompt: int = 4,
    ) -> tuple[
        list[str], torch.Tensor, list[list[int]], list[list[float]], list[list[int]]
    ]:
        """
        Generate samples using vLLM LLMEngine.

        Args:
            prompt_texts: List of prompt strings
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            n_samples_per_prompt: Number of samples per prompt

        Returns:
            completions: List of completion strings
            log_probs: [batch] - Sum of log probs for each completion
            token_ids: List of token ID lists for each completion (generated tokens only)
            token_log_probs: List of per-token log prob lists for each completion
            prompt_token_ids: List of prompt token ID lists for each completion
        """
        logger.info(
            f"Starting generation: {len(prompt_texts)} prompts, "
            f"n_samples_per_prompt={n_samples_per_prompt}, "
            f"max_tokens={max_new_tokens}, temp={temperature}"
        )

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
            n=n_samples_per_prompt,
            seed=42,
            logprobs=1,
            prompt_logprobs=1,  # Also get prompt log probs to access prompt token IDs
            output_kind=RequestOutputKind.FINAL_ONLY,  # Only return completed outputs
        )

        # Add one request per prompt; vLLM handles n_samples_per_prompt via n=
        for request_id, prompt in enumerate(prompt_texts):
            self.engine.add_request(str(request_id), prompt, sampling_params)

        # Step through engine until all requests are finished
        all_outputs = []
        while self.engine.has_unfinished_requests():
            request_outputs = self.engine.step()
            all_outputs.extend(request_outputs)

        # Extract completions and log probs
        completions = []
        log_probs_list = []
        token_ids_list = []
        token_log_probs_list = []
        prompt_token_ids_list = []

        for output in all_outputs:
            prompt_token_ids = output.prompt_token_ids

            for sample in output.outputs:
                completions.append(sample.text)

                # Store prompt tokens for this sample
                prompt_token_ids_list.append(prompt_token_ids)

                # Extract token IDs (generated tokens only)
                token_ids = sample.token_ids
                token_ids_list.append(token_ids)

                # Extract per-token log probs
                per_token_log_probs = [
                    list(logprob_dict.values())[0].logprob
                    for logprob_dict in sample.logprobs
                ]
                token_log_probs_list.append(per_token_log_probs)

                # Sum log probs across generated tokens
                total_log_prob = sum(per_token_log_probs)
                log_probs_list.append(total_log_prob)

        log_probs = torch.tensor(log_probs_list, dtype=torch.float32)

        return (
            completions,
            log_probs,
            token_ids_list,
            token_log_probs_list,
            prompt_token_ids_list,
        )

    def __del__(self):
        """Cleanup vLLM engine."""
        if hasattr(self, "engine"):
            del self.engine
            torch.cuda.empty_cache()


class GeneratorState:
    """States for the Generator's state machine."""

    READY_TO_GENERATE = "READY_TO_GENERATE"
    READY_TO_UPDATE = "READY_TO_UPDATE"


class Generator(Actor, Configurable):
    """
    Generates rollouts using vLLM engine.

    Maintains a vLLM engine that is synchronized with the Trainer
    via weight sync. Generates completions for given prompts and
    computes rewards/advantages.

    Constructed via ``config.build(rl_config=..., prompt_texts=..., expected_answers=...)``
    which calls ``Generator(config=..., rl_config=..., prompt_texts=..., expected_answers=...)``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Generator actor configuration."""

        vllm_engine: VLLMEngine.Config = field(default_factory=VLLMEngine.Config)
        """vLLM rollout engine configuration."""

        vllm_attention_backend: str = "FLASH_ATTN"
        """vLLM attention backend to use (e.g., FLASH_ATTN, XFORMERS)."""

    def __init__(
        self,
        config: Config,
        *,
        rl_config: RLTrainer.Config,
        prompt_texts: list[str],
        expected_answers: list[str],
    ):
        self.config = config
        self.rl_config = rl_config

        # Set vLLM environment variables from config before any vLLM initialization
        if rl_config.batch_invariant_mode:
            os.environ["VLLM_BATCH_INVARIANT"] = "1"
            init_batch_invariance(AttentionBackendEnum.FLASH_ATTN)

        os.environ["VLLM_ATTENTION_BACKEND"] = config.vllm_attention_backend

        self.prompt_texts = prompt_texts
        self.expected_answers = expected_answers

        # Extract needed fields from configs
        self.model_path = rl_config.trainer.checkpoint.initial_load_path
        self.max_new_tokens = config.vllm_engine.sampling.max_tokens
        self.temperature = config.vllm_engine.sampling.temperature
        self.group_size = rl_config.policy_optimization.group_size
        self.grpo_beta = rl_config.policy_optimization.beta
        self.use_stable_grpo = rl_config.policy_optimization.use_stable

        # Initialize distributed environment for SPMD generator
        # TODO: Replace with CommConfig once the config branch lands.
        from torchtitan.config.job_config import Comm

        world_size = dist_utils.init_distributed(Comm())

        # Build vLLM engine from its config
        self.vllm_engine = config.vllm_engine.build(
            model_path=self.model_path,
            dump_folder=rl_config.dump_folder,
        )

        # State machine
        self.state = GeneratorState.READY_TO_UPDATE
        self.cond = asyncio.Condition()
        self.policy_version = 0

        # Reward function. TODO: Use a real reward function
        self.reward_fn = trivial_reward_function

        logger.info("Generator initialized with vLLM engine")

    @endpoint
    async def generate(self) -> None:
        """Generate trajectories and compute rewards/advantages."""
        logger.info(
            f"{os.getpid()=} Generating start generate (policy v{self.policy_version})..."
        )
        async with self.cond:
            # Wait until ready to generate (weights have been updated)
            await self.cond.wait_for(
                lambda: self.state == GeneratorState.READY_TO_GENERATE
            )

            # Generate samples using vLLM
            (
                completions,
                vllm_log_probs,
                vllm_token_ids,
                vllm_token_log_probs,
                prompt_token_ids,
            ) = self.vllm_engine.generate(
                self.prompt_texts,
                self.max_new_tokens,
                self.temperature,
                n_samples_per_prompt=self.group_size,
            )

            # Compute rewards
            logger.info(
                f"Computing rewards: {len(completions)} completions, "
                f"{len(self.expected_answers)} expected answers, "
                f"group_size={self.group_size}"
            )
            rewards = self.reward_fn(
                completions, self.expected_answers, self.group_size
            )

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

            # Create trajectory data
            trajectory = TrajectoryData(
                policy_version=self.policy_version,
                completions=completions,
                vllm_token_ids=vllm_token_ids,
                vllm_token_log_probs=vllm_token_log_probs,
                prompt_token_ids=prompt_token_ids,
                rewards=rewards,
                advantages=advantages,
            )

            # Signal ready for update
            self.state = GeneratorState.READY_TO_UPDATE
            self.cond.notify_all()

            logger.info(
                f"{os.getpid()=} Generating finish generate (policy v{self.policy_version})..."
            )
            return trajectory

    @endpoint
    async def update(self, version: int, state_dict: dict) -> None:
        """Update generator weights.

        Args:
            version: New policy version number
            state_dict: Model state dict
        """
        async with self.cond:
            self.vllm_engine.update_weights(state_dict)
            # Update version and state
            self.policy_version = version
            self.state = GeneratorState.READY_TO_GENERATE
            self.cond.notify_all()
            logger.info(
                f"{os.getpid()=} Generator updating weights to policy v{version}..."
            )
