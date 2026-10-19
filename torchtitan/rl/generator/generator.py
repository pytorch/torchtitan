# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The RL generator service: rollout generation with an inference engine."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
import torchstore as ts
from torchstore import RankRole

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.config import Configurable, DebugConfig
from torchtitan.distributed.batch_invariant import set_batch_invariance
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.dist_moe import DistMoeRuntime
from torchtitan.observability import structured_logger as sl
from torchtitan.observability.logging import init_logger
from torchtitan.rl.generator.dispatcher import RequestDispatcher
from torchtitan.rl.generator.engine import GenerationEngine
from torchtitan.rl.generator.engine_loop import EngineLoop, EngineLoopThread
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.types import EngineRequest, SamplingConfig
from torchtitan.rl.rollout.types import Completion
from torchtitan.rl.routing.intra_generator import IntraGeneratorRouter

logger = logging.getLogger(__name__)


def _default_engine_config() -> GenerationEngine.Config:
    # Imported here so the generator depends on vLLM only when it is the configured engine.
    from torchtitan.rl.generator.vllm.engine import VLLMEngine

    return VLLMEngine.Config()


class Generator(Configurable):
    """Generates completions with an inference engine, behind the endpoints the controller and
    router call.

    The controller fires independent calls (`generate`, `load_model_state_dict`, `close`), and the
    router calls `prefetch_model_state_dict` before `load_model_state_dict` to prefetch weights into
    CPU. Each rank runs one SPMD engine loop on a dedicated engine loop thread; rank 0 takes the
    calls and drives the other ranks through it.

    The generator is split by concern:
    - this class: the endpoints, setup, and the model state dict prefetch;
    - `EngineLoop` (`engine_loop.py`): the loop that turns queued calls into engine steps and
      model state dict loads on every rank;
    - `RequestDispatcher` (`dispatcher.py`): routes requests across data-parallel replicas and
      resolves each call's reply when its request finishes;
    - `GenerationEngine` (`engine.py`): the inference engine interface, implemented for vLLM by
      `VLLMEngine` (`vllm/engine.py`) and selected by `Config.engine`.

    `prefetch_model_state_dict` runs on the caller's event loop and writes the staging buffers the
    engine loop reads on a load: the router awaits it before `load_model_state_dict`, and the
    controller awaits that load before the trainer's next push, so the two never overlap.

    Args:
        config: Generator-specific configuration, including its model.
        model_path: Path to the HF model checkpoint.
        max_num_seqs: The engine's upper bound on concurrently scheduled sequences (it may admit
            fewer if KV is tight).
        output_dir: Structured-logger output directory.
        open_result_channel: Opens the asynchronous channel used to fan in
            completions from nonzero data-parallel replicas to rank 0. It must
            return the sending port and receiving endpoint. The Monarch actor
            adapter supplies ``Channel.open``; standalone runtimes can supply
            an equivalent transport. It is unused when data parallelism is
            disabled.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Generator configuration."""

        model: Decoder.Config
        """Generator model config, built for ``parallelism.to_training()``.

        The RL controller requires the same model class as the trainer's model;
        keeping the two aligned otherwise is the recipe's responsibility.
        """

        parallelism: InferenceParallelismConfig = field(
            default_factory=InferenceParallelismConfig
        )
        """Parallelism configuration for the inference engine."""

        engine: GenerationEngine.Config = field(default_factory=_default_engine_config)
        """The inference engine and its engine-specific settings, selected by config type,
        e.g. ``VLLMEngine.Config()`` (default)."""

        intra_generator_router: IntraGeneratorRouter.Config = field(
            default_factory=IntraGeneratorRouter.Config
        )
        """In-mesh DP routing config: how rank 0 partitions requests across the
        engine's data-parallel ranks (no effect when data_parallel_degree == 1,
        where there is a single DP rank)."""

        sampling: SamplingConfig = field(default_factory=SamplingConfig)
        """Default sampling parameters for generation."""

        dist_moe_runtime: DistMoeRuntime.Config | None = None
        """Dist-MoE runtime policy, with ``inference=True``, required when the
        model config carries Dist-MoE routed experts. Separate from the trainer's
        ``Trainer.Config.dist_moe_runtime``: the generator plans scratch-only execution
        over its own, smaller, expert-parallel mesh."""

        model_dtype: str = "bfloat16"
        """Data type for model weights, passed to the engine (auto, float16, bfloat16, float32)."""

        gpu_memory_limit: float = 0.9
        """Fraction of GPU memory to use for the engine (0.0 to 1.0)."""

        max_num_batched_tokens: int | None = None
        """The engine's chunked-prefill chunk size: max tokens scheduled per engine step
        (prefill + decode, summed over the batch). ``None`` (default) leaves the
        engine's own default in place."""

        checkpointer: CheckpointManager.Config | None = None
        """Optional initial-weight loader for the engine's model.

        In the RL loop this stays ``None`` because weights arrive from
        TorchStore. Standalone inference supplies a config that loads the
        initial Hugging Face weights.
        """

        debug: DebugConfig = field(default_factory=DebugConfig)
        """Debug and determinism settings."""

        max_engine_steps_between_decisions: int = 16
        """Controls how many `engine.step()` calls the `engine_loop` performs before processing a new decision.
        Every generation call is queued for execution by the `engine_loop`. A higher value enables buffering
        of more requests to avoid a prefill between every engine decode step, which is inefficient."""

        reset_kv_cache_on_model_state_dict_load: bool = False
        """Reset cached and running-request KV after each model state dict load.

        The default preserves in-flight requests and their KV: a rollout group keeps the
        cache salt pinned when this generator first admitted it, so its rollouts and
        later turns reuse its KV across model state dict loads, while new groups use the current
        version.
        Enable this to clear prefix-cache entries and preempt running requests; the engine
        then recomputes their KV under the new weights when they resume."""

        def __post_init__(self):
            # The generator runs full expert parallelism: the engine forms the EP
            # group from all DP*TP ranks, so expert_parallel_degree must equal
            # data_parallel_degree * tensor_parallel_degree (or 1 to disable EP).
            p = self.parallelism
            full_ep = p.data_parallel_degree * p.tensor_parallel_degree
            if p.data_parallel_degree > 1 and p.expert_parallel_degree == 1:
                raise ValueError(
                    "generator data_parallel_degree may be greater than 1 only "
                    "to supply ranks for expert parallelism. For independent "
                    "generator replicas, set data_parallel_degree=1 and increase "
                    "Controller.Config.num_generators instead."
                )
            if p.expert_parallel_degree not in (1, full_ep):
                raise ValueError(
                    f"expert_parallel_degree ({p.expert_parallel_degree}) must be 1 "
                    f"(no expert parallelism) or equal data_parallel_degree * "
                    f"tensor_parallel_degree ({full_ep}) in the generator."
                )

            if (
                self.debug.batch_invariant
                and not self.reset_kv_cache_on_model_state_dict_load
            ):
                raise ValueError(
                    "batch_invariant requires reset_kv_cache_on_model_state_dict_load=True so "
                    "cached KV cannot cross a policy update"
                )

    def __init__(
        self,
        config: Config,
        *,
        model_path: str,
        max_num_seqs: int,
        output_dir: str,
        rank: int | None = None,
        generator_name: str = "generator",
        open_result_channel: Callable[[], tuple[Any, Any]] | None = None,
    ):
        init_logger()
        # TODO: Quiet torchstore's per-op transport-resolve INFO spam (very noisy in CI).
        logging.getLogger("torchstore.transport").setLevel(logging.WARNING)
        sl.init_structured_logger(
            source="rl_generator",
            output_dir=output_dir,
            rank=rank if rank is not None else int(os.environ.get("RANK", "0")),
            enable=config.debug.enable_structured_logging,
        )
        sl.log_trace_instant("structured_logger_started")

        self.config = config

        self._rank = rank if rank is not None else int(os.environ.get("RANK", "0"))
        dp_degree = config.parallelism.data_parallel_degree
        tp_degree = config.parallelism.tensor_parallel_degree
        # TODO: revisit if PP/CP are added.
        dp_rank = self._rank // tp_degree
        tp_rank = self._rank % tp_degree

        set_batch_invariance(config.debug.batch_invariant)
        self._set_determinism(config.debug)

        with sl.log_trace_span("engine_init"):
            # The engine is built and used only on the engine loop thread (see `EngineLoop`).
            self._engine_loop_thread = EngineLoopThread(name="engine-loop")
            engine: GenerationEngine = self._engine_loop_thread.call(
                lambda: config.engine.build(
                    model=config.model,
                    parallelism=config.parallelism,
                    model_path=model_path,
                    max_num_seqs=max_num_seqs,
                    model_dtype=config.model_dtype,
                    gpu_memory_limit=config.gpu_memory_limit,
                    max_num_batched_tokens=config.max_num_batched_tokens,
                    checkpointer=config.checkpointer,
                    dist_moe_runtime=config.dist_moe_runtime,
                    debug=config.debug,
                    reset_kv_cache_on_model_state_dict_load=(
                        config.reset_kv_cache_on_model_state_dict_load
                    ),
                    rank=self._rank,
                    generator_name=generator_name,
                    output_dir=output_dir,
                )
            )
            # The prefetch buffers below and each prefetch's TorchStore read pin memory through
            # the CUDA runtime, on the calling thread's device. Both run on this thread, which
            # would otherwise use device 0, rank 0's GPU.
            torch.cuda.set_device(
                self._engine_loop_thread.call(torch.cuda.current_device)
            )
            logger.info("Generator engine initialized")

        # The default PG was initialized during engine build. Confirm the configured
        # rank matches the torch-distributed global rank so the two views cannot
        # silently diverge.
        torch_distributed_rank = dist.get_rank()
        if self._rank != torch_distributed_rank:
            raise RuntimeError(
                f"rank mismatch: configured rank ({self._rank}) != "
                f"torch dist.get_rank() ({torch_distributed_rank})"
            )
        # Confirm the DP layout we computed above matches what the engine derived
        # independently during its build, so the two views can't silently diverge.
        if engine.data_parallel_size != dp_degree:
            raise RuntimeError(
                f"DP layout mismatch on rank {self._rank}: our dp_size "
                f"({dp_degree}) != engine data_parallel_size "
                f"({engine.data_parallel_size})"
            )
        if engine.data_parallel_rank != dp_rank:
            raise RuntimeError(
                f"DP layout mismatch on rank {self._rank}: our dp_rank "
                f"({dp_rank}) != engine data_parallel_rank "
                f"({engine.data_parallel_rank})"
            )

        self._prefetched_model_state_dict = engine.prefetched_model_state_dict

        broadcast_group = dist.new_group(backend="gloo")  # for LoopDecisions
        # The engine loop owns the engine from here on and releases it on exit, so this
        # generator keeps no reference to it.
        self._engine_loop = EngineLoop(
            engine,
            event_loop=self._engine_loop_thread.event_loop,
            request_dispatcher=RequestDispatcher(
                rank=self._rank,
                dp_rank=dp_rank,
                tp_rank=tp_rank,
                dp_degree=dp_degree,
                broadcast_group=broadcast_group,
                intra_generator_router=config.intra_generator_router,
                open_result_channel=open_result_channel,
            ),
            rank=self._rank,
            broadcast_group=broadcast_group,
            max_engine_steps_between_decisions=config.max_engine_steps_between_decisions,
            reset_kv_cache_on_model_state_dict_load=config.reset_kv_cache_on_model_state_dict_load,
        )

        logger.info("Generator initialized")

    @staticmethod
    def _set_determinism(debug: DebugConfig) -> None:
        """Apply deterministic flags for the generator.

        The generator doesn't use torchtitan's ParallelismContext, so we apply
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

    @property
    def policy_version(self) -> int:
        """The policy version of the installed model state dict."""
        return self._engine_loop.policy_version

    async def sync_log_step(self, step: int, relative_step: int | None = None) -> None:
        """Sync the structured-logger step counter from the controller."""
        sl.set_step(step, relative_step=relative_step)
        self._engine_loop.log_engine_stats(step)

    async def start_engine_loop(self) -> None:
        """Start the background engine loop on every rank (one-time, idempotent)."""
        self._engine_loop.start()

    async def wait_engine_loop(self) -> None:
        """Wait until this rank's engine loop exits, raising whatever crashed it.

        The controller watches this so a crashed loop fails the run, instead of leaving a closed
        generator that fails every call routed to it.
        """
        await self._engine_loop.wait()

    @sl.log_trace_span("generate")
    async def generate(
        self,
        prompt_token_ids: list[int],
        *,
        request_id: str,
        group_id: int,
        routing_session_id: str,
        sampling_config: SamplingConfig | None = None,
        metrics_prefix: str = "generator",
    ) -> Completion:
        """Generates one completion for one prompt.

        Can be accepted by rank 0 only (rank 0 owns the queue + replies and
        drives the followers through the engine loop). Returns the `Completion`,
        which carries its own per-generation metrics (`Completion.metrics`) that
        the controller attaches to the rollout turn. If the engine rejects the request
        when the engine loop admits it (e.g. a prompt that leaves no room for an
        output token), only this call raises the engine's error (with vLLM,
        `VLLMValidationError` or `ValueError`).

        Args:
            prompt_token_ids: One tokenized prompt `[token_ids]`.
            request_id: Unique id for this request, echoed on the `Completion`.
            group_id: Rollout group id. Requests of one group share a prefix cache
                salt; `release_groups` drops it. Also sets the vLLM priority; see
                `EngineRequest.group_id`.
            routing_session_id: Stable session key for in-mesh DP routing.
            sampling_config: Optional per-call override for the generator's
                default SamplingConfig.
            metrics_prefix: Namespace prepended to every metric key on the returned
                `Completion` (default ``"generator"``). Callers that need to keep streams
                separate, e.g. ``"validation/generator"``, can override it.

        Example:

            completion = await generator.slice(hosts=0, gpus=0).generate.call_one(
                [1, 2, 3],
                request_id="step=3/group=0/sample=0/turn=0",
                group_id=0,
                routing_session_id="group=0/rollout=0",
            )
        """
        sampling = (
            sampling_config if sampling_config is not None else self.config.sampling
        )
        assert (
            sampling.stop_token_ids is not None
        ), f"{request_id}: stop_token_ids must be set from the renderer"
        return await self._engine_loop.generate(
            EngineRequest(
                request_id=request_id,
                prompt_token_ids=prompt_token_ids,
                sampling=sampling,
                group_id=group_id,
                routing_session_id=routing_session_id,
                arrival_time=time.time(),
            ),
            metrics_prefix=metrics_prefix,
        )

    async def release_groups(self, group_ids: list[int]) -> None:
        """Drop the pinned cache salts and the held KV of finished rollout groups.

        Args:
            group_ids: Groups with no more generation calls.
        """
        self._engine_loop.release_groups(group_ids)

    async def release_sessions(self, session_ids: list[str]) -> None:
        """Release the held KV of sessions whose rollout ended (rank 0; a no-op unless the
        engine holds session KV).

        Args:
            session_ids: Routing session ids with no more generation calls.
        """
        self._engine_loop.release_sessions(session_ids)

    async def initialize_torchstore_client(self, requester_index: int) -> None:
        """Initialize this process as a TorchStore routing requester.

        Args:
            requester_index: Index used to namespace this generator mesh.
        """
        await ts.client(role=RankRole.REQUESTER, group=requester_index)

    @sl.log_trace_span("load_model_state_dict")
    async def load_model_state_dict(self, version: int, *, drain: bool = False) -> None:
        """Queues a model state dict load for `version` and blocks until the engine loop has finished loading.
        Loads queued together are applied once, at the highest version.

        The network transfer has already completed and this load applies the
        prefetched weights to the GPU.

        Args:
            version: Policy version to load
            drain: False applies the load between step bursts while in-flight requests keep decoding.
                True first finishes every admitted request; `generate` calls that arrive meanwhile wait
                and are admitted after the load, so they run on the new weights.
        """
        await self._engine_loop.load_model_state_dict(version, drain=drain)

    @sl.log_trace_span("prefetch_model_state_dict")
    async def prefetch_model_state_dict(self) -> None:
        """Fetch weights into pinned CPU memory without interrupting generation."""
        # Async RL uses a StorageVolume snapshot so generators do not read
        # live trainer GPU tensors while optimizer steps may be mutating them.
        await ts.get_state_dict(
            "model_state_dict",
            user_state_dict=self._prefetched_model_state_dict,
            strict=False,
            direct_rdma=False,
        )

    async def close(self) -> None:
        """Stop the engine loop, which releases the dispatcher and the engine on exit."""
        await self._engine_loop.close()
