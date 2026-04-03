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
import torchstore as ts
from monarch.actor import Actor, endpoint
from torchtitan.config import Configurable
from torchtitan.config.configs import ParallelismConfig
from torchtitan.experiments.rl.models.vllm_wrapper import (
    register_model_to_vllm_model_registry,
    VLLM_MODEL_NAME,
)
from torchtitan.experiments.rl.types import Completion
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools.utils import has_cuda_capability
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
    """Generates completions using a vLLM engine.

    Maintains a vLLM engine synchronized with the Trainer via weight sync.
    Given prompts, returns Completion objects with token IDs and logprobs.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Generator actor configuration."""

        model_spec: ModelSpec | None = None
        """Model specification. Set programmatically by the orchestrator."""

        hf_assets_path: str = ""
        """Path to HF model checkpoint. Set programmatically by the orchestrator."""

        batch_invariant_mode: bool = True
        """Enable batch-invariant mode for deterministic NCCL ops."""

        parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
        """Parallelism configuration for the vLLM engine."""

        sampling: SamplingConfig = field(default_factory=SamplingConfig)
        """Default sampling parameters for generation."""

        model_dtype: str = "bfloat16"
        """Data type for model weights (auto, float16, bfloat16, float32)."""

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

    def __init__(self, config: Config):
        self.config = config

        assert (
            config.model_spec is not None
        ), "model_spec must be set before creating VLLMGenerator"
        model_spec = config.model_spec

        # The generator uses the RL-specific TP-only parallelize plan for
        # vLLM compatibility. Override on a copy so the shared model_spec
        # (used by the trainer with core Titan's parallelize) is untouched.
        from dataclasses import replace as dataclass_replace

        from torchtitan.experiments.rl.models.parallelize import (
            parallelize_qwen3 as rl_parallelize_qwen3,
        )

        generator_model_spec = dataclass_replace(
            model_spec, parallelize_fn=rl_parallelize_qwen3
        )
        self.model_spec = generator_model_spec

        # Register TorchTitan model with vLLM before any engine creation
        register_model_to_vllm_model_registry(generator_model_spec)

        # Set vLLM environment variables from config before any vLLM initialization
        os.environ["VLLM_ATTENTION_BACKEND"] = "CUSTOM"

        if config.batch_invariant_mode:
            os.environ["VLLM_BATCH_INVARIANT"] = "1"
            init_batch_invariance(AttentionBackendEnum.CUSTOM)

        # Build vLLM engine
        engine_kwargs = dict(
            model=config.hf_assets_path,
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
                backend=AttentionBackendEnum.CUSTOM,
            ),
            disable_log_stats=True,
        )
        # FA2 requires block_size to be a multiple of 256
        if not has_cuda_capability(9, 0):
            engine_kwargs["block_size"] = 256
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
    async def generate(self, prompts: list[str]) -> list[Completion]:
        """Generate completions for the given prompts.

        Each prompt produces ``num_samples_per_prompt`` completions. Completions from the
        same prompt share a ``group_id`` so the controller can compute
        group-level advantages later.

        Args:
            prompts: List of prompt strings to generate completions for.

        Returns:
            Flat list of Completions (len = len(prompts) * num_samples_per_prompt).
        """
        logger.debug(
            f"{os.getpid()=} generate start (policy v{self.policy_version})..."
        )

        with torch.no_grad():
            sampling_params = SamplingParams(
                temperature=self.config.sampling.temperature,
                top_p=self.config.sampling.top_p,
                max_tokens=self.config.sampling.max_tokens,
                n=self.config.num_samples_per_prompt,
                seed=self.config.seed,
                logprobs=1,
                prompt_logprobs=1,
                output_kind=RequestOutputKind.FINAL_ONLY,
            )

            for request_id, prompt in enumerate(prompts):
                self._engine.add_request(str(request_id), prompt, sampling_params)

            all_outputs = []
            while self._engine.has_unfinished_requests():
                request_outputs = self._engine.step()
                all_outputs.extend(request_outputs)

            # Sort by request_id to guarantee prompt ordering,
            # since vLLM may return completed requests out of order.
            all_outputs.sort(key=lambda o: int(o.request_id))

            completions: list[Completion] = []
            for idx, output in enumerate(all_outputs):
                gid = f"{os.getpid()}_{self.policy_version}_{idx}"

                for sample in output.outputs:
                    per_token_logprobs = [
                        next(iter(logprob_dict.values())).logprob
                        for logprob_dict in sample.logprobs
                    ]
                    completions.append(
                        Completion(
                            prompt_tokens=output.prompt_token_ids,
                            response_tokens=sample.token_ids,
                            logprobs=per_token_logprobs,
                            group_id=gid,
                            text=sample.text,
                            policy_version=self.policy_version,
                        )
                    )

        logger.debug(
            f"{os.getpid()=} generate finish (policy v{self.policy_version})..."
        )
        return completions

    @endpoint
    async def pull_weights(self, version: int) -> None:
        """Pull latest weights from TorchStore.

        When ``direct_rdma=True``, weights are read directly from the
        trainer's GPU memory via one-sided RDMA, bypassing StorageVolumes.
        When ``False``, data is fetched through StorageVolumes (which may
        themselves use RDMA as their transport internally).

        Args:
            version: New policy version number.
        """
        from monarch.rdma import is_rdma_available

        model_sd = self._get_model().model.state_dict()
        await ts.get_state_dict(
            "model_state_dict",
            user_state_dict=model_sd,
            strict=False,
            direct_rdma=is_rdma_available(),
        )
        self.policy_version = version
        logger.debug(
            f"{os.getpid()=} Generator pulled model state dict for policy v{version}"
        )

    def __del__(self):
        """Cleanup vLLM engine."""
        if hasattr(self, "_engine"):
            del self._engine
            torch.cuda.empty_cache()
