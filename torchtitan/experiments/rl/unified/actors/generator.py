# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass, field
from typing import Literal

import torch
from monarch.actor import Actor, endpoint
from torch.distributed.tensor import distribute_tensor, DTensor
from torchtitan.config import Configurable
from torchtitan.config.configs import ParallelismConfig
from torchtitan.experiments.rl.unified.plugin import (
    register_model_to_vllm_model_registry,
    VLLM_MODEL_NAME,
)
from torchtitan.experiments.rl.unified.types import Episode
from torchtitan.protocols.model_spec import ModelSpec
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.config import AttentionConfig, CompilationConfig
from vllm.model_executor.layers.batch_invariant import init_batch_invariance
from vllm.sampling_params import RequestOutputKind
from vllm.v1.attention.backends.registry import AttentionBackendEnum

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class GeneratorCompileConfig:
    """Compilation and CUDA graph settings for the vLLM generator."""

    backend: Literal["none", "eager", "inductor"] = "eager"
    """torch.compile backend for vLLM
    When set to a value other than "none", enables compilation with the specified backend.
    See https://docs.vllm.ai/en/stable/api/vllm/config/#vllm.config.CompilationConfig.backend
    NOTE: "eager" means compile with dynamo backend (like torch.compile(backend="eager"))
    NOTE: inductor will offer the best performance, but will impact numerics - use eager for
    bitwise identical results."""

    cudagraph_mode: Literal[
        "none", "piecewise", "full", "full_and_piecewise"
    ] = "piecewise"
    """CUDA graph capture mode for vLLM.
    Piecewise capture supports dynamic sizes and splits cudagraphs around non capturable
      ops like attention
    Full capture captures one graph at the expense of less dynamism and requires full
      capturability
    full_and_piecewise does both and selects which to use based on dynamism
    NOTE: Piecewise graph capture requires torch.compile for graph capture and splitting
    See https://docs.vllm.ai/en/latest/design/cuda_graphs/#cudagraphmodes for more details."""

    def __post_init__(self) -> None:
        if self.backend == "none" and self.cudagraph_mode in (
            "piecewise",
            "full_and_piecewise",
        ):
            raise ValueError(
                f"cudagraph_mode='{self.cudagraph_mode}' requires piecewise graph "
                "capture which depends on torch.compile. Set backend "
                "to 'eager' or 'inductor'."
            )

    @property
    def is_eager(self) -> bool:
        """Inferred from backend and cudagraph_mode."""
        return self.backend == "none" and self.cudagraph_mode == "none"

    def get_vllm_compilation_config(self) -> CompilationConfig | None:
        """Build a vLLM ``CompilationConfig``, or return ``None`` when both
        compilation and CUDA graphs are disabled.
        """
        if self.is_eager:
            return None

        kwargs: dict = dict(cudagraph_mode=self.cudagraph_mode)
        if self.backend == "none":
            # Disable torch.compile but keep CUDA graphs (e.g. full mode).
            # mode=0 (CompilationMode.NONE) prevents vLLM from inferring
            # VLLM_COMPILE based on the default optimization level.
            kwargs["mode"] = 0
        else:
            kwargs["backend"] = self.backend

        return CompilationConfig(**kwargs)


@dataclass(kw_only=True, slots=True)
class SamplingConfig:
    """Sampling parameters passed to vLLM's SamplingParams."""

    temperature: float = 0.8
    """Sampling temperature. 0.0 = greedy, higher = more random."""

    top_p: float = 0.95
    """Nucleus sampling threshold."""

    max_tokens: int = 100
    """Maximum number of tokens to generate per completion."""


class VLLMGenerator(Actor, Configurable):
    """
    Generates rollouts using vLLM engine.

    Maintains a vLLM engine that is synchronized with the Trainer
    via weight sync. Generates completions for given prompts and
    computes rewards/advantages.

    Args:
        config: Generator-specific configuration.
        model_path: Path to the HF model checkpoint.
        batch_invariant_mode: Enable batch-invariant mode for deterministic ops.
        prompt_texts: List of prompt strings.
        TODO: refine `prompt_texts` according to input type (eg, a list of token sequences, or conversion)
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Generator actor configuration.
        TODO: Expose a EngineConfig field to passing config to vLLM Engine"""

        parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
        """Parallelism configuration for the vLLM engine."""

        sampling: SamplingConfig = field(default_factory=SamplingConfig)
        """Default sampling parameters for generation."""

        attention_backend: str = "FLASH_ATTN"
        """vLLM attention backend to use (e.g., FLASH_ATTN).
        Now we only support / explored FlashAttention"""

        model_dtype: str = "bfloat16"
        """Data type for model weights, passed directly to vLLM (auto, float16, bfloat16, float32)."""

        gpu_memory_limit: float = 0.9
        """Fraction of GPU memory to use for the vLLM engine (0.0 to 1.0)."""

        compile: GeneratorCompileConfig = field(default_factory=GeneratorCompileConfig)
        """Compilation and CUDA graph settings for the vLLM engine."""

        num_samples_per_prompt: int = 8
        """Number of completions to generate per prompt."""

        seed: int | None = None
        """Random seed for vLLM engine and sampling. None for non-deterministic."""

        def __post_init__(self):
            assert self.parallelism.data_parallel_shard_degree in (1, -1), (
                f"Generator does not support data parallel sharding, "
                f"got dp_shard={self.parallelism.data_parallel_shard_degree}"
            )
            assert self.parallelism.data_parallel_replicate_degree == 1, (
                f"Generator does not support data parallel replication, "
                f"got dp_replicate={self.parallelism.data_parallel_replicate_degree}"
            )

    def __init__(
        self,
        config: Config,
        *,
        model_spec: ModelSpec,
        model_path: str,
        batch_invariant_mode: bool,
    ):
        self.config = config
        self.model_spec = model_spec

        # Register TorchTitan model with vLLM before any engine creation
        register_model_to_vllm_model_registry(model_spec)

        # Set vLLM environment variables from config before any vLLM initialization
        os.environ["VLLM_ATTENTION_BACKEND"] = config.attention_backend

        if batch_invariant_mode:
            os.environ["VLLM_BATCH_INVARIANT"] = "1"
            init_batch_invariance(AttentionBackendEnum[config.attention_backend])

        # Extract needed fields from configs
        self.model_path = model_path
        self.max_new_tokens = config.sampling.max_tokens
        self.temperature = config.sampling.temperature
        self.top_p = config.sampling.top_p
        self.num_samples_per_prompt = config.num_samples_per_prompt

        # Build vLLM engine
        engine_kwargs = dict(
            model=model_path,
            trust_remote_code=True,
            dtype=config.model_dtype,
            tensor_parallel_size=config.parallelism.tensor_parallel_degree,
            # Monarch already spawned TP workers via proc mesh. "external_launcher"
            # tells vLLM to run one worker per process (no subprocess spawning)
            distributed_executor_backend="external_launcher",
            gpu_memory_utilization=config.gpu_memory_limit,
            enforce_eager=config.compile.is_eager,
            hf_overrides={"architectures": [VLLM_MODEL_NAME]},
            attention_config=AttentionConfig(
                backend=AttentionBackendEnum[config.attention_backend],
            ),
            disable_log_stats=True,
        )
        vllm_compilation_config = config.compile.get_vllm_compilation_config()
        if vllm_compilation_config is not None:
            engine_kwargs["compilation_config"] = vllm_compilation_config
        if config.seed is not None:
            engine_kwargs["seed"] = config.seed
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
    async def generate(
        self,
        prompt_texts: list[str],
        expected_answers: list[str],
    ) -> list[Episode]:
        """Generate completions and return a flat list of Episodes.

        Each prompt produces ``num_samples_per_prompt`` Episodes. Episodes
        from the same prompt share a ``group_id`` so the controller can
        compute group-level advantages later.

        Args:
            prompt_texts: List of prompt strings for which to generate completions.
            expected_answers: List of expected answers, one per prompt.
                They are copied into each Episode so downstream graders can use them
                for reward computation and generator doesn't use this field.
        """
        logger.debug(
            f"{os.getpid()=} Generating start generate (policy v{self.policy_version})..."
        )

        with torch.no_grad():
            # Generate samples using vLLM
            sampling_params = SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_new_tokens,
                n=self.num_samples_per_prompt,
                seed=self.config.seed,
                logprobs=1,
                prompt_logprobs=1,  # Also get prompt log probs to access prompt token IDs
                output_kind=RequestOutputKind.FINAL_ONLY,  # Only return completed outputs
            )

            for request_id, prompt in enumerate(prompt_texts):
                self._engine.add_request(str(request_id), prompt, sampling_params)

            # Step through engine until all requests are finished
            all_outputs = []
            while self._engine.has_unfinished_requests():
                request_outputs = self._engine.step()
                all_outputs.extend(request_outputs)

            # Sort outputs by request_id to guarantee prompt ordering,
            # since vLLM may return completed requests out of order.
            all_outputs.sort(key=lambda o: int(o.request_id))

            # Build flat list of Episodes; assign a group_id per prompt.
            # TODO: Assigning group_id here is GRPO-specific and should be
            # decoupled from the generator in the future.
            episodes: list[Episode] = []
            for idx, output in enumerate(all_outputs):
                prompt_token_ids = output.prompt_token_ids
                gid = f"{os.getpid()}_{self.policy_version}_{idx}"

                for sample in output.outputs:
                    per_token_log_probs = [
                        list(logprob_dict.values())[0].logprob
                        for logprob_dict in sample.logprobs
                    ]
                    episodes.append(
                        Episode(
                            policy_version=self.policy_version,
                            prompt_token_ids=prompt_token_ids,
                            text=sample.text,
                            token_ids=sample.token_ids,
                            token_log_probs=per_token_log_probs,
                            expected_answer=expected_answers[idx]
                            if expected_answers
                            else "",
                            group_id=gid,
                        )
                    )

        logger.debug(
            f"{os.getpid()=} Generating finish generate (policy v{self.policy_version})..."
        )
        return episodes

    @endpoint
    async def update(self, version: int, state_dict: dict) -> None:
        """Update generator weights.
        Called by the orchestrator.

        Args:
            version: New policy version number
            state_dict: Full (unsharded) state dict with plain tensors.
        """
        # Reshard full tensors to match this generator's DTensor layout
        model_state_dict = dict(self._get_model().model.state_dict())
        for name, tensor in state_dict.items():
            if name in model_state_dict and isinstance(model_state_dict[name], DTensor):
                if isinstance(tensor, DTensor):
                    continue
                target_dtensor = model_state_dict[name]
                state_dict[name] = distribute_tensor(
                    tensor.to(target_dtensor.device_mesh.device_type),
                    device_mesh=target_dtensor.device_mesh,
                    placements=target_dtensor.placements,
                )

        load_weights = self._get_model().load_weights_from_state_dict(state_dict)
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
