# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import os

from typing import List

import torch
from monarch.actor import Actor, endpoint
from torch.distributed._tensor import DTensor
from torchtitan.config.job_config import Comm
from torchtitan.distributed import utils as dist_utils

# Import unified module - this automatically registers TorchTitan models with vLLM
from torchtitan.experiments.rl import unified  # noqa: F401

from torchtitan.experiments.rl.unified.actors.grader import Episodes
from torchtitan.experiments.rl.unified.job_config import JobConfig

from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.model_executor.layers.batch_invariant import init_batch_invariance
from vllm.sampling_params import RequestOutputKind

from vllm.v1.attention.backends.registry import AttentionBackendEnum

logger = logging.getLogger(__name__)


class VLLMGenerator:
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

        # Load TorchTitan plugin at runtime
        from torchtitan.experiments.rl.unified import register

        register(model_flavor=job_config.model.flavor)
        logger.info("Loaded TorchTitan vLLM plugin")

        # Create the vLLM engine at initialization time
        generation_config = self.job_config.generation
        model_path = job_config.checkpoint.initial_load_path

        engine_args = EngineArgs(
            # Model configuration
            model=model_path,
            trust_remote_code=True,
            dtype=generation_config.dtype,
            # Parallelism configuration
            tensor_parallel_size=generation_config.parallelism.tensor_parallel_degree,
            distributed_executor_backend="external_launcher",
            # Memory and performance
            gpu_memory_utilization=generation_config.gpu_memory_utilization,
            enforce_eager=generation_config.enforce_eager,
            # Seed (use debug seed if set, otherwise default to 42)
            seed=job_config.debug.seed if job_config.debug.seed is not None else 42,
            # HuggingFace overrides to use TorchTitan model.
            # TODO: make this field configurable and align with model registration
            hf_overrides={"architectures": ["Qwen3TorchTitanForCausalLM"]},
        )

        logger.info("Initializing LLMEngine from EngineArgs...")
        self.engine = LLMEngine.from_engine_args(engine_args)
        logger.info("vLLM rollout engine initialized")

    def _get_model(self):
        """Access the model from the vLLM engine tensor operations.
        Returns a TorchTitanVLLMModelWrapper instance.
        """
        return self.engine.model_executor.driver_worker.get_model()

    def update_weights(self, state_dict: dict) -> None:
        """
        Update vLLM actor model weights from state dict.

        Args:
            state_dict: DTensor-wrapped state dict matching the model's placements
        """
        load_weights = self._get_model().load_weights_from_state_dict(state_dict)

        logger.info(
            f"Updated weights into vLLM engine actor model. Number of parameters: {len(load_weights)}"
        )

        assert self.engine is not None
        # Use collective_rpc to call reload_weights on all workers
        # This reloads weights from temp_model_dir without recreating the engine
        self.engine.collective_rpc("reload_weights")

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
    returns unscored trajectory data for the Scorer to process.

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
        # When running under Monarch, setup_env_for_distributed already
        # initializes the process group, so skip re-initialization.
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        else:
            world_size = dist_utils.init_distributed(
                Comm(),
            )
        # Initialize vLLM engine with job_config
        self.vllm_engine = VLLMGenerator(job_config, self.model_path)

        # State machine
        self.state = GeneratorState.READY_TO_UPDATE
        self.cond = asyncio.Condition()
        self.policy_version = 0

        logger.info("Generator initialized with vLLM engine")

    @endpoint
    async def generate(self) -> Episodes:
        """Generate trajectories without computing rewards/advantages.

        Returns:
            Episodes for the Scorer to process (rewards=None)
        """
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

            logger.info(f"Generated {len(completions)} completions for scoring")

            # Create trajectory data (rewards initialized to zeros, filled by Grader)
            trajectory = Episodes(
                policy_version=self.policy_version,
                completions=completions,
                vllm_token_ids=vllm_token_ids,
                vllm_token_log_probs=vllm_token_log_probs,
                prompt_token_ids=prompt_token_ids,
                expected_answers=self.expected_answers,
                rewards=torch.zeros(len(completions)),
            )

            # Signal ready for update
            self.state = GeneratorState.READY_TO_UPDATE
            self.cond.notify_all()

            logger.info(
                f"{os.getpid()=} Generating finish generate (policy v{self.policy_version})..."
            )
            return trajectory

    @endpoint
    async def update(self, version: int, all_weights: dict) -> None:
        """Update generator weights.

        Reconstructs DTensors from plain local tensors using the vLLM model's
        device mesh and placements, then loads into the vLLM engine.

        Args:
            version: New policy version number
            all_weights: Dict mapping GPU rank to plain local tensor state dict
        """
        async with self.cond:
            my_rank = torch.distributed.get_rank()
            state_dict = all_weights[my_rank]

            # Re-wrap plain tensors as DTensors matching the vLLM model's placements.
            # The trainer unwraps DTensors to local tensors before sending (to avoid
            # cross-mesh issues). Since trainer and generator share the same TP layout,
            # the local tensor data matches the target placement directly.
            model = self.vllm_engine._get_model()
            model_state_dict = {k: v for k, v in model.model.state_dict().items()}
            for name, tensor in state_dict.items():
                if name in model_state_dict and isinstance(
                    model_state_dict[name], DTensor
                ):
                    if isinstance(tensor, DTensor):
                        continue
                    target_dtensor = model_state_dict[name]
                    device_mesh = target_dtensor.device_mesh
                    state_dict[name] = DTensor.from_local(
                        tensor.to(device_mesh.device_type),
                        device_mesh=device_mesh,
                        placements=list(target_dtensor.placements),
                    )

            self.vllm_engine.update_weights(state_dict)
            # Update version and state
            self.policy_version = version
            self.state = GeneratorState.READY_TO_GENERATE
            self.cond.notify_all()
            logger.info(
                f"{os.getpid()=} Generator updating weights to policy v{version}..."
            )
