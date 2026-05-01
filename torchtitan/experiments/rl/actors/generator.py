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
from torchtitan.config.configs import DebugConfig, ParallelismConfig
from torchtitan.distributed.utils import set_batch_invariance
from torchtitan.experiments.rl.models.vllm_registry import (
    register_model_to_vllm_model_registry,
    VLLM_MODEL_NAME,
)
from torchtitan.experiments.rl.types import Completion
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools.utils import has_cuda_capability
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.config import AttentionConfig, CompilationConfig
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

    n: int = 8
    """Number of completions to generate per prompt (vLLM SamplingParams.n)."""

    temperature: float = 0.8
    """Sampling temperature. 0.0 = greedy, higher = more random."""

    top_p: float = 0.95
    """Nucleus sampling threshold."""

    max_tokens: int = 100
    """Maximum number of tokens to generate per completion."""


class VLLMGenerator(Actor, Configurable):
    """
    Generates rollouts using vLLM engine.

    Maintains a vLLM engine synchronized with the Trainer via weight
    sync. ``generate()`` produces a flat list of Completions; reward
    and advantage computation live in the controller.

    Args:
        config: Generator-specific configuration.
        model_spec: TorchTitan model specification.
        model_path: Path to the HF model checkpoint.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Generator actor configuration.
        TODO: Expose a EngineConfig field to passing config to vLLM Engine"""

        parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)
        """Parallelism configuration for the vLLM engine."""

        sampling: SamplingConfig = field(default_factory=SamplingConfig)
        """Default sampling parameters for generation."""

        model_dtype: str = "bfloat16"
        """Data type for model weights, passed directly to vLLM (auto, float16, bfloat16, float32)."""

        gpu_memory_limit: float = 0.9
        """Fraction of GPU memory to use for the vLLM engine (0.0 to 1.0)."""

        compile: GeneratorCompileConfig = field(default_factory=GeneratorCompileConfig)
        """Compilation and CUDA graph settings for the vLLM engine."""

        debug: DebugConfig = field(default_factory=DebugConfig)
        """Debug and determinism settings."""

        def __post_init__(self):
            # Generator only supports TP. vLLM handles its own parallelism
            # and we only apply TP via the core parallelize function.
            p = self.parallelism
            if p.data_parallel_replicate_degree != 1:
                raise ValueError(
                    f"Generator does not support data parallel replication, "
                    f"got dp_replicate={p.data_parallel_replicate_degree}"
                )
            if p.pipeline_parallel_degree > 1:
                raise ValueError(
                    f"Generator does not support pipeline parallelism, "
                    f"got pp={p.pipeline_parallel_degree}"
                )
            if p.context_parallel_degree > 1:
                raise ValueError(
                    f"Generator does not support context parallelism, "
                    f"got cp={p.context_parallel_degree}"
                )
            if p.expert_parallel_degree > 1:
                raise ValueError(
                    f"Generator does not support expert parallelism, "
                    f"got ep={p.expert_parallel_degree}"
                )

    def __init__(
        self,
        config: Config,
        *,
        model_spec: ModelSpec,
        model_path: str,
    ):
        self.config = config
        self.model_spec = model_spec

        # Register TorchTitan model with vLLM before any engine creation
        register_model_to_vllm_model_registry(model_spec)

        from torchtitan.models.common.attention import FlexAttention

        inner_attn = model_spec.model.layers[0].attention.inner_attention
        self._use_flex = isinstance(inner_attn, FlexAttention.Config)

        if self._use_flex:
            os.environ["VLLM_ATTENTION_BACKEND"] = "FLEX_ATTENTION"
        else:
            os.environ["VLLM_ATTENTION_BACKEND"] = "CUSTOM"

        set_batch_invariance(config.debug.batch_invariant)

        self._set_determinism(config.debug)

        self.model_path = model_path

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
                backend=AttentionBackendEnum.FLEX_ATTENTION
                if self._use_flex
                else AttentionBackendEnum.CUSTOM,
            ),
            disable_log_stats=True,
        )
        if not has_cuda_capability(9, 0) and not self._use_flex:
            # FA2 requires block_size to be a multiple of 256
            engine_kwargs["block_size"] = 256
        vllm_compilation_config = config.compile.get_vllm_compilation_config()
        if vllm_compilation_config is not None:
            engine_kwargs["compilation_config"] = vllm_compilation_config
        if config.debug.seed is not None:
            engine_kwargs["seed"] = config.debug.seed
        engine_args = EngineArgs(**engine_kwargs)

        logger.info("Initializing LLMEngine from EngineArgs...")
        self._engine = LLMEngine.from_engine_args(engine_args)
        logger.info("vLLM rollout engine initialized")

        self.policy_version = 0

        logger.info("Generator initialized with vLLM engine")

    @staticmethod
    def _set_determinism(debug: DebugConfig) -> None:
        """Apply deterministic flags for the generator.

        The generator doesn't use torchtitan's ParallelDims, so we apply
        the deterministic flags directly instead of using set_determinism().
        """
        if debug.deterministic:
            torch.use_deterministic_algorithms(
                True, warn_only=debug.deterministic_warn_only
            )
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        if debug.seed is not None:
            torch.manual_seed(debug.seed)

    def _get_model(self):
        """Access the model from the vLLM engine.
        Returns a TorchTitanVLLMModelWrapper instance.
        """
        return self._engine.model_executor.driver_worker.get_model()

    @endpoint
    async def generate(
        self,
        prompts: list[str],
        *,
        sampling_config: SamplingConfig | None = None,
    ) -> list[Completion]:
        """Generate completions for the given prompts.

        Returns a flat list of length ``len(prompts) * sampling.n``
        ordered by ``prompt_idx``, with ``sampling.n`` consecutive
        completions per prompt.

        Args:
            prompts: List of prompt strings.
            sampling_config: Optional per-call override for the generator's
                default SamplingConfig. ``seed`` always comes from
                ``config.debug.seed`` (not part of SamplingConfig).
        """
        _sampling_config = (
            sampling_config if sampling_config is not None else self.config.sampling
        )

        logger.debug(
            f"{os.getpid()=} Generating start generate (policy v{self.policy_version})..."
        )

        with torch.no_grad():
            sampling_params = SamplingParams(
                temperature=_sampling_config.temperature,
                top_p=_sampling_config.top_p,
                max_tokens=_sampling_config.max_tokens,
                n=_sampling_config.n,
                seed=self.config.debug.seed,
                logprobs=1,
                output_kind=RequestOutputKind.FINAL_ONLY,
            )

            for i, prompt in enumerate(prompts):
                self._engine.add_request(str(i), prompt, sampling_params)

            all_outputs = []
            while self._engine.has_unfinished_requests():
                all_outputs.extend(self._engine.step())

            # vLLM may return requests out of order; sort by the integer
            # request_id we assigned so prompt_idx lines up with the input.
            all_outputs.sort(key=lambda o: int(o.request_id))

            completions: list[Completion] = []
            for output in all_outputs:
                prompt_idx = int(output.request_id)
                prompt_token_ids = output.prompt_token_ids
                for sample in output.outputs:
                    per_token_logprobs = [
                        list(logprob_dict.values())[0].logprob
                        for logprob_dict in sample.logprobs
                    ]
                    completions.append(
                        Completion(
                            policy_version=self.policy_version,
                            prompt_idx=prompt_idx,
                            prompt_token_ids=prompt_token_ids,
                            text=sample.text,
                            token_ids=sample.token_ids,
                            token_logprobs=per_token_logprobs,
                        )
                    )

        logger.debug(
            f"{os.getpid()=} Generating finish generate (policy v{self.policy_version})..."
        )
        return completions

    @endpoint
    async def pull_model_state_dict(self, version: int) -> None:
        """Pull latest weights from TorchStore.

        When ``direct_rdma=True``, weights are read directly from the
        trainer's GPU memory via one-sided RDMA, bypassing StorageVolumes.
        When ``False``, data is fetched through StorageVolumes (which may
        themselves use RDMA as their transport internally).

        See ``push_model_state_dict`` for more details on the distinction.

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
        # Invalidate the KV prefix cache so stale values computed with the
        # old weights are never reused for new generations.
        self._engine.reset_prefix_cache()
        logger.debug(
            f"{os.getpid()=} Generator pulled model state dict for policy v{version}"
        )

    def __del__(self):
        """Cleanup vLLM engine."""
        if hasattr(self, "_engine"):
            del self._engine
            torch.cuda.empty_cache()
