# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

from dataclasses import dataclass, field

import torch
from monarch.actor import Actor, endpoint
from torch.distributed.tensor import DTensor
from torchtitan.config import Configurable
from torchtitan.config.configs import ParallelismConfig

from torchtitan.experiments.rl.unified.actors.grader import Episode
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


class Generator(Actor, Configurable):
    """
    Generates rollouts using vLLM engine.

    Maintains a vLLM engine that is synchronized with the Trainer
    via weight sync. Generates completions for given prompts and
    computes rewards/advantages.

    Args:
        config: Generator-specific configuration.
        model_path: Path to the HF model checkpoint.
        batch_invariant_mode: Enable batch-invariant mode for deterministic ops.
        policy_optimization: GRPO hyperparameters.
        prompt_texts: List of prompt strings.
        expected_answers: List of expected answer strings.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Generator actor configuration."""

        parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
        """Parallelism configuration for the vLLM engine."""

        sampling: VLLMSamplingConfig = field(default_factory=VLLMSamplingConfig)
        """Default sampling parameters for generation."""

        vllm_attention_backend: str = "FLASH_ATTN"
        """vLLM attention backend to use (e.g., FLASH_ATTN, XFORMERS)."""
        
        vllm_model_dtype: str = "bfloat16"
        """Data type for model weights, passed directly to vLLM (auto, float16, bfloat16, float32)."""

        vllm_gpu_memory_limit: float = 0.5
        """Fraction of GPU memory to use for the vLLM engine (0.0 to 1.0)."""

        vllm_enforce_eager: bool = True
        """Disable CUDA graphs in vLLM (use eager execution)."""

        vllm_seed: int | None = None
        """Random seed for vLLM engine and sampling. None for non-deterministic."""

    def __init__(
        self,
        config: Config,
        *,
        model_spec: ModelSpec,
        model_path: str,
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

        # Build vLLM engine
        engine_kwargs = dict(
            model=model_path,
            trust_remote_code=True,
            dtype=config.vllm_model_dtype,
            tensor_parallel_size=config.parallelism.tensor_parallel_degree,
            distributed_executor_backend="external_launcher",
            gpu_memory_utilization=config.vllm_gpu_memory_limit,
            enforce_eager=config.vllm_enforce_eager,
            hf_overrides={"architectures": [VLLM_MODEL_NAME]},
            attention_config=AttentionConfig(
                backend=AttentionBackendEnum.FLASH_ATTN,
            ),
        )
        if config.vllm_seed is not None:
            engine_kwargs["seed"] = config.vllm_seed
        engine_args = EngineArgs(**engine_kwargs)

        logger.info("Initializing LLMEngine from EngineArgs...")
        self._engine = LLMEngine.from_engine_args(engine_args)
        logger.info("vLLM rollout engine initialized")

        self.policy_version = 0

        logger.info("Generator initialized with vLLM engine")

    def _get_model(self):
        """Access the model from the vLLM engine.
        Returns a TorchTitanVLLMModelWrapper instance.
        """
        return self._engine.model_executor.driver_worker.get_model()

    @endpoint
    async def generate(self) -> Episode:
        """Generate episodes and return Episode with zero rewards.
        Called by the orchestrator (simple_grpo.py). The Grader fills in rewards.
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
                seed=self.config.vllm_seed,
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
                    prompt_token_ids_list.append(prompt_token_ids)
                    token_ids_list.append(sample.token_ids)
                    per_token_log_probs = [
                        list(logprob_dict.values())[0].logprob
                        for logprob_dict in sample.logprobs
                    ]
                    token_log_probs_list.append(per_token_log_probs)

        # Create episode with zero rewards (Grader will fill them in)
        episode = Episode(
            policy_version=self.policy_version,
            completions=completions,
            vllm_token_ids=token_ids_list,
            vllm_token_log_probs=token_log_probs_list,
            prompt_token_ids=prompt_token_ids_list,
            expected_answers=self.expected_answers,
            rewards=torch.zeros(len(completions)),
        )

        logger.info(
            f"{os.getpid()=} Generating finish generate (policy v{self.policy_version})..."
        )
        return episode

    @endpoint
    async def update(self, version: int, state_dict: dict) -> None:
        """Update generator weights.
        Called by the orchestrator (simple_grpo.py).

        Args:
            version: New policy version number
            state_dict: Per-rank state dicts keyed by GPU index,
                e.g. {0: state_dict_gpu0, 1: state_dict_gpu1}
        """
        # Extract this rank's state dict from the per-rank dict
        rank = int(os.environ.get("LOCAL_RANK", 0))
        local_state_dict = state_dict[rank]

        # Convert plain local tensors (TP shards from trainer) to DTensors
        # matching the vLLM model's sharding layout. The trainer exports
        # weights via to_local() which strips DTensor metadata.
        model_state_dict = dict(
            self._get_model().model.state_dict()
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

        load_weights = self._get_model().load_weights_from_state_dict(local_state_dict)
        self.policy_version = version
        logger.debug(
            f"Updated weights into vLLM engine model. "
            f"Number of parameters: {len(load_weights)}. "
            f"{os.getpid()=} Generator updating weights to policy v{version}..."
        )

    def __del__(self):
        """Cleanup vLLM engine."""
        if hasattr(self, "_engine"):
            del self._engine
            torch.cuda.empty_cache()
