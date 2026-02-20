# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import os

from dataclasses import dataclass
from typing import List

import torch
from monarch.actor import Actor, endpoint
from safetensors.torch import save_file
from torchtitan.config.job_config import Comm
from torchtitan.distributed import utils as dist_utils

# Import unified module - this automatically registers TorchTitan models with vLLM
from torchtitan.experiments.rl import unified  # noqa: F401

from torchtitan.experiments.rl.unified.job_config import JobConfig

from torchtitan.experiments.rl.vllm_compat.simple_rl import (
    compute_grpo_advantages,
    compute_grpo_advantages_stable,
    trivial_reward_function,
)

from vllm.config import AttentionConfig
from vllm import EngineArgs, LLMEngine, SamplingParams
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


class VLLMRolloutEngine:
    """
    vLLM engine for fast rollouts with weight updates.

    Note: vLLM loads from model_config.model path, so we create a temporary
    directory with updated weights and restart the engine. This is faster than
    recreating temp dirs repeatedly and handles config/tokenizer files properly.

    Args:
        job_config: JobConfig dataclass containing all configuration
        model_path: Path to HuggingFace model (for config/tokenizer)
    """

    def __init__(
        self,
        job_config: JobConfig,
        model_path: str,
    ):
        # Store job_config for accessing configuration
        self.job_config = job_config
        self.base_model_path = model_path

        self.temp_model_dir = os.path.abspath(
            os.path.join(job_config.job.dump_folder, "vllm_temp_model")
        )
        os.makedirs(self.temp_model_dir, exist_ok=True)

        import glob

        # Copy config/tokenizer files from base model to temp dir
        import shutil

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
                shutil.copy2(src, self.temp_model_dir)

        # Copy the original model shard files if they exist
        # We'll overwrite these with our single model.safetensors later
        for shard_file in glob.glob(os.path.join(model_path, "model-*.safetensors")):
            dst = os.path.join(self.temp_model_dir, os.path.basename(shard_file))
            shutil.copy2(shard_file, dst)

        # Copy index file if it exists
        index_file = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(index_file):
            shutil.copy2(index_file, self.temp_model_dir)

        self.engine = None
        logger.info("vLLM rollout engine initialized (will load on first use)")

    def update_weights(self, vllm_state: dict) -> None:
        """
        Update vLLM model weights from vLLM-compat state dict.

        This converts weights to vLLM format, saves them, and reloads using
        vLLM's reload_weights() API after updating the model path config.

        Args:
            vllm_state: vLLM model state dict
        """
        # Save to temp model directory
        import os

        checkpoint_path = os.path.join(self.temp_model_dir, "model.safetensors")

        # Update the shard files that vLLM will actually load
        # We need to split our weights to match the original 2-shard structure
        import glob

        shard_files = sorted(
            glob.glob(os.path.join(self.temp_model_dir, "model-*.safetensors"))
        )
        index_file = os.path.join(self.temp_model_dir, "model.safetensors.index.json")

        # TODO: need to replace this with Torchtitan's checkpoint save and load
        # right now we hardcoded to work with 2 safetensor files which we only
        # tested on Qwen3 0.6B model. In the longer term, need to use TorchStore
        # to achieve the weight communication.
        # only generator rank 0 saves the weight
        if torch.distributed.get_rank() == 0:
            logger.info(f"Saving weights to {checkpoint_path}")

            # Ensure weights stay in bfloat16
            vllm_state = {
                k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
                for k, v in vllm_state.items()
            }
            # Fallback: save as single file
            save_file(vllm_state, checkpoint_path)

        # Synchronize all ranks before reloading to ensure rank 0 finished writing
        torch.distributed.barrier()
        logger.info(
            f"[Rank {torch.distributed.get_rank()}] Synchronized after weight save"
        )

        # First time: create the engine using LLMEngine and EngineArgs
        if self.engine is None:
            generation = self.job_config.generation

            engine_args = EngineArgs(
                # Model configuration
                model=self.temp_model_dir,
                trust_remote_code=True,
                dtype=generation.dtype,
                # Parallelism configuration
                tensor_parallel_size=generation.parallelism.tensor_parallel_degree,
                distributed_executor_backend="external_launcher",
                # Memory and performance
                gpu_memory_utilization=generation.gpu_memory_utilization,
                enforce_eager=generation.enforce_eager,
                # Seed
                seed=self.job_config.debug.seed,
                # HuggingFace overrides to use TorchTitan model.
                # TODO: make this field configurable and align with model registration
                hf_overrides={"architectures": ["Qwen3TorchTitanForCausalLM"]},
                attention_config=AttentionConfig(
                    backend=AttentionBackendEnum.FLASH_ATTN,
                ),
            )

            logger.info("Initializing LLMEngine from EngineArgs...")
            self.engine = LLMEngine.from_engine_args(engine_args)
            logger.info("Created new vLLM LLMEngine")
        else:
            # Direct parameter copy into model tensors.
            # This bypasses vLLM's reload_weights() which uses a layerwise
            # reload mechanism that moves params to meta device — incompatible
            # with TorchTitanVLLMModelWrapper.load_weights().
            # Same approach as MSL RL's VllmLoadSnapshotMethod.update_tensor_shard().
            self._direct_weight_update(vllm_state)

    def _direct_weight_update(self, vllm_state: dict) -> None:
        """Update model weights by copying directly into GPU parameters."""
        from torchtitan.experiments.rl.vllm_compat.weights.converter import (
            vllm_to_torchtitan,
        )

        # Convert vLLM/HF format → TorchTitan format
        titan_state = vllm_to_torchtitan(vllm_state)

        # Access model from vLLM engine (same as MSL RL)
        model = self.llm.llm_engine.model_executor.driver_worker.get_model()
        params = dict(model.named_parameters())

        updated = 0
        for name, new_weight in titan_state.items():
            # TorchTitanVLLMModelWrapper stores the model as self.model,
            # so parameters have a "model." prefix
            param_name = f"model.{name}"
            if param_name in params:
                param = params[param_name]
                param.data.copy_(new_weight.to(device=param.device, dtype=param.dtype))
                updated += 1
            else:
                logger.warning(f"Parameter {param_name} not found in vLLM model")

        logger.info(f"Updated {updated}/{len(titan_state)} parameters via direct copy")

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
            seed=42,
            logprobs=1,
            prompt_logprobs=1,  # Also get prompt log probs to access prompt token IDs
            output_kind=RequestOutputKind.FINAL_ONLY,  # Only return completed outputs
        )

        # Add requests to the engine
        # For n_samples_per_prompt > 1, submit each prompt multiple times with different request id
        request_id = 0
        prompt_indices = []  # Track which prompt each request corresponds to
        for prompt_idx, prompt in enumerate(prompt_texts):
            for sample_idx in range(n_samples_per_prompt):
                self.engine.add_request(str(request_id), prompt, sampling_params)
                prompt_indices.append(prompt_idx)
                request_id += 1

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
            # Extract prompt token IDs from the output
            prompt_token_ids = output.prompt_token_ids

            # Each output now has exactly 1 sample (we submitted multiple requests)
            assert (
                len(output.outputs) == 1
            ), f"Expected 1 output, got {len(output.outputs)}"
            sample = output.outputs[0]

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


class Generator(Actor):
    """
    Generates rollouts using vLLM engine.

    Maintains a vLLM engine that is synchronized with the Trainer
    via weight sync. Generates completions for given prompts and
    computes rewards/advantages.

    Args:
        job_config: JobConfig dataclass containing all configuration
        prompt_texts: List of prompt strings
        expected_answers: List of expected answers
    """

    def __init__(
        self,
        job_config: JobConfig,
        prompt_texts: List[
            str
        ],  # TODO: This field need to be removed once dataloader is implemented
        expected_answers: List[str],
    ):
        # Set vLLM environment variables from config before any vLLM initialization
        policy_opt = job_config.policy_optimization
        if policy_opt.vllm_batch_invariant:
            os.environ["VLLM_BATCH_INVARIANT"] = "1"
            init_batch_invariance(AttentionBackendEnum.FLASH_ATTN)

        os.environ["VLLM_ATTENTION_BACKEND"] = policy_opt.vllm_attention_backend

        # Store job_config for accessing configuration
        self.job_config = job_config
        self.prompt_texts = prompt_texts
        self.expected_answers = expected_answers

        # Extract needed fields from job_config
        self.model_path = job_config.checkpoint.initial_load_path
        self.max_new_tokens = job_config.generation.sampling.max_tokens
        self.temperature = job_config.generation.sampling.temperature
        self.group_size = job_config.policy_optimization.grpo_group_size
        self.grpo_beta = job_config.policy_optimization.grpo_beta
        self.use_stable_grpo = job_config.policy_optimization.use_stable_grpo

        # Initialize distributed environment for SPMD generator
        world_size = dist_utils.init_distributed(
            Comm(),
        )
        # Initialize vLLM engine with job_config
        self.vllm_engine = VLLMRolloutEngine(job_config, self.model_path)

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
    async def update(self, version: int, vllm_compat_state: dict) -> None:
        """Update generate weights.

        Args:
            version: New policy version number
            vllm_compat_state: vLLM-compatible state dict
        """
        async with self.cond:
            self.vllm_engine.update_weights(vllm_compat_state)
            # Update version and state
            self.policy_version = version
            self.state = GeneratorState.READY_TO_GENERATE
            self.cond.notify_all()
            logger.info(
                f"{os.getpid()=} Generator updating weights to policy v{version}..."
            )
