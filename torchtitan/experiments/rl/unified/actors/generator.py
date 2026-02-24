# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import os

from dataclasses import dataclass, field

import torch
from monarch.actor import Actor, endpoint
from torch.distributed._tensor import DTensor
from torchtitan.config import CommConfig, Configurable
from torchtitan.config.configs import ParallelismConfig
from torchtitan.distributed import utils as dist_utils

from torchtitan.experiments.rl.unified.actors.grader import Episodes
from torchtitan.experiments.rl.unified.configs import (
    PolicyOptimizationConfig,
    VLLMSamplingConfig,
)

from torchtitan.protocols.model_spec import ModelSpec
from vllm import EngineArgs, LLMEngine, SamplingParams

from vllm.config import AttentionConfig
from vllm.model_executor.layers.batch_invariant import init_batch_invariance
from vllm.sampling_params import RequestOutputKind
from vllm.v1.attention.backends.registry import AttentionBackendEnum

logger = logging.getLogger(__name__)


class VLLMEngine(Configurable):
    """
    vLLM engine for fast rollouts with weight updates.

    The engine is created at initialization time from the model path.
    Subsequent weight updates are done in-place via
    ``load_weights_from_state_dict``.

    Constructed via ``VLLMEngine(config, model_path=..., dump_folder=...)``.
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
        model_spec: ModelSpec,
        model_path: str,
        dump_folder: str,
    ) -> None:
        self.config = config
        self.base_model_path = model_path

        # Register TorchTitan model with vLLM before creating the engine
        from torchtitan.experiments.rl.unified.plugin import register as register_vllm_model

        register_vllm_model(model_spec)

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

        vllm_engine: VLLMEngine.Config = field(default_factory=VLLMEngine.Config)
        """vLLM rollout engine configuration."""

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
        from torchtitan.experiments.rl.unified.plugin import register

        register(model_spec)

        # Set vLLM environment variables from config before any vLLM initialization
        if batch_invariant_mode:
            os.environ["VLLM_BATCH_INVARIANT"] = "1"
            init_batch_invariance(AttentionBackendEnum.FLASH_ATTN)

        os.environ["VLLM_ATTENTION_BACKEND"] = config.vllm_attention_backend

        self.prompt_texts = prompt_texts
        self.expected_answers = expected_answers

        # Extract needed fields from configs
        self.model_path = model_path
        self.max_new_tokens = config.vllm_engine.sampling.max_tokens
        self.temperature = config.vllm_engine.sampling.temperature
        self.group_size = policy_optimization.group_size

        # Initialize distributed environment for SPMD generator
        world_size = dist_utils.init_distributed(CommConfig())

        # Build vLLM engine
        self.vllm_engine = VLLMEngine(
            config.vllm_engine,
            model_spec=model_spec,
            model_path=self.model_path,
            dump_folder=dump_folder,
        )

        # State machine
        self.state = GeneratorState.READY_TO_UPDATE
        self.cond = asyncio.Condition()
        self.policy_version = 0

        logger.info("Generator initialized with vLLM engine")

    @endpoint
    async def generate(self) -> Episodes:
        """Generate completions and return an Episodes object.

        Rewards are initialized to zeros — the Grader fills them in.
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

            # Create episode with zero rewards (Grader will fill them in)
            episode = Episodes(
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
            return episode

    @endpoint
    async def update(self, version: int, state_dict: dict) -> None:
        """Update generator weights.

        Args:
            version: New policy version number
            state_dict: Per-rank state dicts keyed by GPU index,
                e.g. {0: state_dict_gpu0, 1: state_dict_gpu1}
        """
        async with self.cond:
            # Extract this rank's state dict from the per-rank dict
            rank = int(os.environ.get("LOCAL_RANK", 0))
            local_state_dict = state_dict[rank]

            # Convert plain local tensors (TP shards from trainer) to DTensors
            # matching the vLLM model's sharding layout. The trainer exports
            # weights via to_local() which strips DTensor metadata.
            model_state_dict = dict(
                self.vllm_engine._get_model().model.state_dict()
            )
            for name, tensor in local_state_dict.items():
                if name in model_state_dict and isinstance(
                    model_state_dict[name], DTensor
                ):
                    if isinstance(tensor, DTensor):
                        continue
                    target_dtensor = model_state_dict[name]
                    local_state_dict[name] = DTensor.from_local(
                        tensor.to(target_dtensor.device_mesh.device_type),
                        device_mesh=target_dtensor.device_mesh,
                        placements=target_dtensor.placements,
                    )

            self.vllm_engine.update_weights(local_state_dict)
            # Update version and state
            self.policy_version = version
            self.state = GeneratorState.READY_TO_GENERATE
            self.cond.notify_all()
            logger.info(
                f"{os.getpid()=} Generator updating weights to policy v{version}..."
            )
