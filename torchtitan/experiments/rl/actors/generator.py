# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import os
from dataclasses import dataclass, field

import torch
import torchstore as ts
from monarch.actor import Actor, current_rank, endpoint
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.config import (
    CompileConfig,
    Configurable,
    DebugConfig,
    ParallelismConfig,
)
from torchtitan.distributed.utils import set_batch_invariance
from torchtitan.experiments.rl.models.vllm_registry import (
    registry_to_vllm,
    TORCHTITAN_CONFIG_FORMAT,
)
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.types import Completion
from torchtitan.models.common.attention import FlexAttention, VarlenAttention
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.tools.logging import init_logger
from torchtitan.tools.utils import has_cuda_capability
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.config import AttentionConfig, CompilationConfig
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.attention.backends.registry import AttentionBackendEnum

logger = logging.getLogger(__name__)


def _prepare_generation_request_metrics(
    output: RequestOutput, *, prefix: str
) -> list[m.Metric]:
    """Prepare vLLM metrics from a RequestOutput.

    For `[num_prompts]` submitted prompts, vLLM returns `[num_prompts]`
    per parent `RequestOutput`s (one per `add_request` call), each carrying
    a single `RequestStateStats` on `.metrics`.

    Caveat under `SamplingParams.n > 1`: vLLM stores one `RequestStateStats`
    per child request; the parent output exposes the **last-finishing**
    child's timeline. `arrival_time` is shared across siblings, but
    [`queued_ts`, `scheduled_ts`, `first_token_ts`, `last_token_ts`,
    `num_generation_tokens`] describe one specific child — not an aggregate,
    not the first sibling's. The other `n-1` siblings' stats are dropped by
    vLLM at ``output_processor._finish_request``.
    """

    # TODO: Per-request fields here come from RequestOutput.metrics
    # (RequestStateStats). Engine-aggregate stats, such as KV-cache usage,
    # prefix-cache hit rate, preemptions, and batch occupancy, live in
    # SchedulerStats / IterationStats and require registering a
    # vllm.v1.metrics.loggers.StatLoggerBase via
    # LLMEngine.from_engine_args(..., stat_loggers=[...]).

    metric_values: dict[str, float] = {}
    if output.num_cached_tokens is not None:
        metric_values[f"{prefix}/num_cached_tokens"] = output.num_cached_tokens

    stats = output.metrics
    if stats is not None:
        metric_values[f"{prefix}/queue_time_ms"] = (
            stats.scheduled_ts - stats.queued_ts
        ) * 1000

        if stats.num_generation_tokens > 0:
            metric_values[f"{prefix}/time_to_first_token_ms"] = (
                stats.first_token_latency * 1000
            )
            metric_values[f"{prefix}/prefill_time_ms"] = (
                stats.first_token_ts - stats.scheduled_ts
            ) * 1000

        if stats.num_generation_tokens > 1:
            first_to_last_token_ms = (stats.last_token_ts - stats.first_token_ts) * 1000
            metric_values[f"{prefix}/decode_time_ms"] = first_to_last_token_ms
            metric_values[
                f"{prefix}/inter_token_latency_ms"
            ] = first_to_last_token_ms / (stats.num_generation_tokens - 1)

    # Emit each value with both Mean and Max aggregators.
    return [
        metric
        for key, value in metric_values.items()
        for metric in (m.Metric(key, m.Mean(value)), m.Metric(key, m.Max(value)))
    ]


@dataclass(kw_only=True, slots=True)
class VLLMCudagraphConfig:
    """CUDA graph capture settings for the vLLM inference engine.

    torch.compile is configured separately via ``CompileConfig`` at the
    ``RLTrainer`` level, shared by both trainer and generator.  Only CUDA
    graph capture, which is vLLM-specific, is controlled here.

    When enabled, vLLM captures the forward pass as a single CUDA graph
    ("full" mode).  "piecewise" modes are intentionally excluded: they
    require vLLM's whole-model torch.compile to split the graph around
    non-capturable ops, which conflicts with per-layer compile.
    """

    enable: bool = True
    """Whether to enable CUDA graph capture (vLLM "full" mode)."""

    # TODO: Validate CUDA graph capture with MoE / Expert Parallelism.
    # MoE routing produces dynamic shapes that may conflict with full
    # CUDA graph capture despite being torch.compile-compatible
    # post https://github.com/pytorch/torchtitan/pull/3142

    # TODO: Explore applying CUDA graph capture on the torchtitan trainer
    # side as well (not just the vLLM generator).
    # https://github.com/pytorch/torchtitan/issues/3175

    def get_vllm_compilation_config(
        self, *, max_num_seqs: int
    ) -> CompilationConfig | None:
        """Build a vLLM ``CompilationConfig``, or return ``None`` when
        CUDA graphs are disabled.

        ``max_num_seqs`` determines CUDA graph capture sizes: powers of
        2 from 1 up to ``max_num_seqs``, plus ``max_num_seqs`` itself
        if it isn't already a power of 2.
        """
        if not self.enable:
            return None
        if max_num_seqs <= 0:
            raise ValueError(f"max_num_seqs must be positive, got {max_num_seqs}")
        sizes = [1 << i for i in range(int(math.log2(max_num_seqs)) + 1)]
        if max_num_seqs not in sizes:
            sizes.append(max_num_seqs)
        return CompilationConfig(
            cudagraph_mode="full",
            mode=0,
            cudagraph_capture_sizes=sorted(sizes),
        )


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

    Maintains a vLLM engine synchronized with the Trainer via weight
    sync. ``generate()`` produces a flat list of Completions; reward
    and advantage computation live in the controller.

    Args:
        config: Generator-specific configuration.
        model_spec: TorchTitan model specification.
        model_path: Path to the HF model checkpoint.
        compile_config: Per-layer torch.compile config shared with the
            trainer so both sides compile identically.
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

        cudagraph: VLLMCudagraphConfig = field(default_factory=VLLMCudagraphConfig)
        """CUDA graph capture settings for the vLLM engine."""

        checkpoint: CheckpointManager.Config = field(
            default_factory=CheckpointManager.Config
        )
        """Controls whether the vLLM wrapper loads initial HF weights.
        In the RL loop this should stay disabled (default ``enable=False``)
        because weights arrive from TorchStore. For standalone inference,
        set ``enable=True`` and ``initial_load_in_hf=True``."""

        debug: DebugConfig = field(default_factory=DebugConfig)
        """Debug and determinism settings."""

        def __post_init__(self):
            # VLLMGenerator supports TP plus MoE EP. vLLM handles its own
            # process groups, and the wrapper applies the model parallelisms.
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
            if p.enable_sequence_parallel:
                raise ValueError(
                    "Generator does not support sequence parallelism: "
                    "spmd_types erasure mode requires sequence length to be "
                    "evenly divisible by TP, which doesn't hold for inference "
                    "(uneven batches). Set enable_sequence_parallel=False."
                )
            if not p.disable_loss_parallel:
                raise ValueError(
                    "Generator requires disable_loss_parallel=True, "
                    f"got disable_loss_parallel={p.disable_loss_parallel}"
                )

    def __init__(
        self,
        config: Config,
        *,
        model_spec: ModelSpec,
        model_path: str,
        compile_config: CompileConfig,
        max_num_seqs: int,
        output_dir: str,
        stop_token_ids: list[int],
    ):
        init_logger()
        sl.init_structured_logger(
            source="rl_generator",
            output_dir=output_dir,
            rank=current_rank().rank,
            enable=config.debug.enable_structured_logging,
        )
        sl.log_trace_instant("structured_logger_started")

        self.config = config
        self.model_spec = model_spec

        # max_num_seqs controls vLLM's maximum batch dimension: it sets
        # the upper bound for concurrent sequences, determines KV-cache
        # block allocation (and therefore GPU memory usage), and bounds
        # the CUDA graph capture sizes.  Always computed by the caller
        # (RLTrainer) as num_groups_per_rollout_batch * group_size.
        self._max_num_seqs = max_num_seqs

        # Renderer role-boundary stop tokens (e.g. Qwen3 `<|im_end|>`), injected by the
        # controller;
        self._stop_token_ids = stop_token_ids

        # Register TorchTitan model + parser with vLLM
        registry_to_vllm(
            model_spec,
            parallelism=config.parallelism,
            compile_config=compile_config,
            checkpoint_config=config.checkpoint,
        )

        # Set vLLM environment variables from config before any vLLM initialization
        inner_attn = model_spec.model.layers[0].attention.inner_attention
        assert isinstance(
            inner_attn,
            (VarlenAttention.Config, FlexAttention.Config),
        ), "Only varlen and flex attention backends are allowed."

        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
        set_batch_invariance(config.debug.batch_invariant)

        self._set_determinism(config.debug)

        self.model_path = model_path

        # Build vLLM engine
        enable_ep = config.parallelism.expert_parallel_degree > 1
        engine_kwargs = dict(
            # ``model`` is the path to the HF checkpoint directory. The
            # config is sourced from torchtitan's ModelSpec via
            # ``config_format=TORCHTITAN_CONFIG_FORMAT`` (no config.json
            # read), but vLLM still uses this path to locate the
            # tokenizer assets and the safetensors weight shards.
            model=model_path,
            trust_remote_code=True,
            # Use the torchtitan custom config parser (registered by
            # registry_to_vllm above). It builds PretrainedConfig from
            # ModelSpec instead of reading config.json from disk.
            config_format=TORCHTITAN_CONFIG_FORMAT,
            dtype=config.model_dtype,
            tensor_parallel_size=config.parallelism.tensor_parallel_degree,
            # NOTE: Monarch launches the generator workers and sets the torch
            # elastic distributed env; with external_launcher, vLLM uses that
            # world to build its process groups. vLLM does not take an
            # explicit EP degree: when this boolean is set, it converts all
            # DP * TP ranks into the expert-parallel group for MoE layers.
            enable_expert_parallel=enable_ep,
            # Monarch already spawned TP workers via proc mesh. "external_launcher"
            # tells vLLM to run one worker per process (no subprocess spawning)
            distributed_executor_backend="external_launcher",
            gpu_memory_utilization=config.gpu_memory_limit,
            enforce_eager=not config.cudagraph.enable,
            attention_config=AttentionConfig(
                backend=(
                    AttentionBackendEnum.FLEX_ATTENTION
                    if isinstance(inner_attn, FlexAttention.Config)
                    else AttentionBackendEnum.CUSTOM
                ),
            ),
            # Enables RequestOutput.metrics, so generator metrics can be returned
            disable_log_stats=False,
        )
        engine_kwargs["max_model_len"] = model_spec.model.max_seq_len
        engine_kwargs["max_num_seqs"] = self._max_num_seqs
        # FA2 requires block_size to be a multiple of 256
        if not has_cuda_capability(9, 0):
            engine_kwargs["block_size"] = 256
        vllm_compilation_config = config.cudagraph.get_vllm_compilation_config(
            max_num_seqs=self._max_num_seqs,
        )
        if vllm_compilation_config is not None:
            engine_kwargs["compilation_config"] = vllm_compilation_config
        if config.debug.seed is not None:
            engine_kwargs["seed"] = config.debug.seed
        engine_args = EngineArgs(**engine_kwargs)

        with sl.log_trace_span("vllm_init"):
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
        Returns a VLLMModelWrapper instance.
        """
        return self._engine.model_executor.driver_worker.get_model()

    @endpoint
    async def sync_log_step(self, step: int, relative_step: int | None = None) -> None:
        """Sync the structured-logger step counter from the controller."""
        sl.set_step(step, relative_step=relative_step)

    @endpoint
    @sl.log_trace_span("generate")
    async def generate(
        self,
        tokenized_prompts: list[list[int]],
        *,
        request_ids: list[str],
        sampling_config: SamplingConfig | None = None,
        metrics_prefix: str = "generator",
    ) -> tuple[list[Completion], list[m.Metric]]:
        """Generate completions and generator metrics for tokenized prompts.

        Takes ``tokenized_prompts`` as ``[num_prompts][prompt_tokens]``.
        Returns completions in the same order as ``request_ids`` plus generator
        metrics.

        Args:
            tokenized_prompts: Tokenized prompts shaped ``[num_prompts][prompt_tokens]``.
            request_ids: One id per prompt echoed on each ``Completion.request_id``.
            sampling_config: Optional per-call override for the generator's
                default SamplingConfig. ``seed`` always comes from
                ``config.debug.seed`` (not part of SamplingConfig).
            metrics_prefix: Namespace prepended to every returned metric key
                (default ``"generator"``). Callers that need to keep streams
                separate, e.g. ``"validation/generator"``, can override it.
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
                n=1,  # group_size pre-expands prompts; the RL loop always samples n=1
                stop_token_ids=self._stop_token_ids or None,
                seed=self.config.debug.seed,
                logprobs=1,
                output_kind=RequestOutputKind.FINAL_ONLY,
            )

            if len(request_ids) != len(tokenized_prompts):
                raise ValueError(
                    f"got {len(request_ids)} request_ids for {len(tokenized_prompts)} prompts"
                )
            if len(set(request_ids)) != len(request_ids):
                raise ValueError(f"request_ids must be unique; got {request_ids}")

            # render_cmpl is vLLM's input-pipeline entry.
            # The tokenize step is a no-op for already-tokenized prompts. The
            # lower-level alternative is vllm.inputs.tokens_input; we use the
            # high-level API to stay resilient to vLLM internal changes.
            engine_inputs = self._engine.renderer.render_cmpl(
                [{"prompt_token_ids": ids} for ids in tokenized_prompts]
            )
            for engine_input, request_id in zip(
                engine_inputs, request_ids, strict=True
            ):
                self._engine.add_request(
                    request_id=request_id,
                    prompt=engine_input,
                    params=sampling_params,
                )

            all_outputs = []
            with sl.log_trace_span("engine_steps"):
                while self._engine.has_unfinished_requests():
                    all_outputs.extend(self._engine.step())

            # Return completions in caller input order; the controller maps positionally.
            request_order = {request_id: i for i, request_id in enumerate(request_ids)}
            all_outputs.sort(key=lambda output: request_order[output.request_id])

            completions: list[Completion] = []
            generation_metrics: list[m.Metric] = []
            output_token_counts: list[int] = []
            for output in all_outputs:
                generation_metrics.extend(
                    _prepare_generation_request_metrics(output, prefix=metrics_prefix)
                )
                for sample in output.outputs:
                    per_token_logprobs = [
                        list(logprob_dict.values())[0].logprob
                        for logprob_dict in sample.logprobs
                    ]
                    output_token_counts.append(len(sample.token_ids))
                    completions.append(
                        Completion(
                            policy_version=self.policy_version,
                            request_id=output.request_id,
                            token_ids=sample.token_ids,
                            token_logprobs=per_token_logprobs,
                            finish_reason=sample.finish_reason,
                        )
                    )
            generation_metrics.append(
                m.Metric(
                    f"{metrics_prefix}/output_tokens",
                    m.Sum.from_list(output_token_counts),
                )
            )

        logger.debug(
            f"{os.getpid()=} Generating finish generate (policy v{self.policy_version})..."
        )

        # TODO: consider passing metrics as an arg to Completion and Trajectory,
        # e.g. Completion(metrics=generation_metrics), so when we log the rollouts
        # to a json or database,  we can associate each rollout with its metrics
        # I am keeping like this for now for simplicity.
        return completions, generation_metrics

    @endpoint
    @sl.log_trace_span("pull_model_state_dict")
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

    @endpoint
    async def close(self) -> None:
        """Release the vLLM engine.

        With ``external_launcher``, vLLM reuses the process group and actor
        lifetime that Monarch owns. Calling vLLM's internal
        ``engine_core.shutdown()`` can block while Monarch is also trying to
        stop the same actor mesh, so this endpoint only closes renderer-local
        resources and leaves process teardown to ``ProcMesh.stop()``.
        """
        if self._engine is not None:
            renderer = getattr(self._engine, "renderer", None)
            if renderer is not None:
                logger.info("Shutting down vLLM renderer")
                renderer.shutdown()
            self._engine = None
