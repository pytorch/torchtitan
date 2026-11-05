# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""vLLM inference engine for the RL generator: every vLLM call the generator makes."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from hashlib import sha256
from typing import Any

import torch
import torch.distributed as dist
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.config import AttentionConfig, KVTransferConfig
from vllm.distributed import get_world_group
from vllm.exceptions import VLLMClientError
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.config import DebugConfig, PrecisionConfig
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.dist_moe import DistMoeRuntime
from torchtitan.rl.generator.engine import GenerationEngine, ModelStateDictLoadMode
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.types import (
    EngineRequest,
    FinishedRequest,
    RequestMetricsInputs,
    SamplingConfig,
)
from torchtitan.rl.generator.vllm.batch_invariance import (
    force_logprobs_fn_for_batch_invariance,
)
from torchtitan.rl.generator.vllm.config import VLLMCudaGraphConfig
from torchtitan.rl.generator.vllm.registry import (
    register_to_vllm,
    TORCHTITAN_CONFIG_FORMAT,
    TORCHTITAN_WORKER_CLS,
    vllm_attention_backend,
)
from torchtitan.rl.generator.vllm.session_kv_holder import VLLMSessionKVHolder
from torchtitan.rl.generator.vllm.stat_logger import (
    StatLoggerContext,
    VllmOtelStatLogger,
)
from torchtitan.tools.utils import has_cuda_capability

logger = logging.getLogger(__name__)


def add_extra_vllm_engine_args(
    engine_kwargs: dict[str, Any], extra_engine_args: dict[str, Any]
) -> None:
    """Add ``VLLMEngine.Config.extra_engine_args`` to ``engine_kwargs`` in place.

    Raises ``ValueError`` for keys TorchTitan already set in ``engine_kwargs`` or
    derives from a config field, so they cannot silently diverge from it.
    """
    # EngineArgs backed by dedicated config fields, including optional arguments
    # that may be absent from engine_kwargs.
    config_field_by_engine_arg = {
        "max_num_batched_tokens": "Generator.Config.max_num_batched_tokens",
        "compilation_config": "VLLMEngine.Config.cuda_graph",
        "kv_transfer_config": "VLLMEngine.Config.kv_transfer_config",
        "seed": "Generator.Config.debug.seed",
    }
    owned_keys = sorted(
        extra_engine_args.keys()
        & (engine_kwargs.keys() | config_field_by_engine_arg.keys())
    )
    if owned_keys:
        details = ", ".join(
            (
                f"{key} (use {config_field_by_engine_arg[key]})"
                if key in config_field_by_engine_arg
                else key
            )
            for key in owned_keys
        )
        raise ValueError(
            f"extra_engine_args cannot set EngineArgs that TorchTitan controls: {details}"
        )
    engine_kwargs.update(extra_engine_args)


class VLLMEngine(GenerationEngine):
    """The generator's vLLM ``LLMEngine``, behind the calls the engine loop makes.

    See ``GenerationEngine`` for the build arguments and the threading contract;
    ``LLMEngine`` is not thread-safe.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GenerationEngine.Config):
        cuda_graph: VLLMCudaGraphConfig = field(default_factory=VLLMCudaGraphConfig)
        """CUDA graph capture settings for the vLLM engine."""

        kv_transfer_config: KVTransferConfig | None = None
        """Native vLLM KV offload configuration. None: no connector. See
        ``SIMPLE_CPU_OFFLOAD.md`` (local) and ``MOONCAKE_STORE.md`` (distributed)."""

        extra_engine_args: dict[str, Any] = field(default_factory=dict)
        """Additional keyword arguments passed to vLLM's ``EngineArgs``, for engine
        options without a dedicated field (e.g. ``{"watermark": 0.03}``). vLLM
        validates them. Keys that TorchTitan sets itself, or derives from a config
        field (e.g. ``max_num_batched_tokens``), raise a ``ValueError``; see
        ``add_extra_vllm_engine_args``."""

        stat_logger: VllmOtelStatLogger.Config | None = None
        """Optional logger instantiated on TP rank 0 to export vLLM metrics."""

        session_kv_holder: VLLMSessionKVHolder.Config | None = None
        """Keep each multi-turn session's cached prefix on GPU between its turns (see
        `VLLMSessionKVHolder`). With KV offload configured, also send CPU cache
        release hints when sessions end. None disables both session behaviors."""

        @property
        def supports_hot_swap(self) -> bool:
            return self.kv_transfer_config is None

        @property
        def holds_session_kv(self) -> bool:
            return self.session_kv_holder is not None

        def __post_init__(self):
            if self.kv_transfer_config is None:
                return
            # TODO(acisseJZhong): support prefill/decode disaggregation connectors
            # if needed, and MultiConnector to combine them with KV offload.
            if self.kv_transfer_config.kv_connector not in (
                "MooncakeStoreConnector",
                "SimpleCPUOffloadConnector",
            ):
                raise ValueError(
                    "Titan RL supports MooncakeStoreConnector and SimpleCPUOffloadConnector; "
                    "MultiConnector and other KV connectors are not supported."
                )
            if self.kv_transfer_config.kv_role != "kv_both":
                raise ValueError(
                    "KV offload requires kv_role='kv_both': "
                    "connectors both save and load KV, so vLLM must enable "
                    "producer finalization and consumer block-lifetime protection."
                )
            if self.kv_transfer_config.kv_connector_module_path not in (
                None,
                (
                    "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector"
                    if self.kv_transfer_config.kv_connector == "MooncakeStoreConnector"
                    else "vllm.distributed.kv_transfer.kv_connector.v1.simple_cpu_offload_connector"
                ),
            ):
                raise ValueError(
                    "KV offload requires vLLM's built-in connector in "
                    "user config. Titan installs its lifecycle adapter automatically; "
                    "user-provided custom kv_connector_module_path values are not supported."
                )

    def __init__(
        self,
        config: Config,
        *,
        model: Decoder.Config,
        parallelism: InferenceParallelismConfig,
        model_path: str,
        max_num_seqs: int,
        precision: PrecisionConfig,
        gpu_memory_limit: float,
        max_num_batched_tokens: int | None,
        checkpointer: CheckpointManager.Config | None,
        dist_moe_runtime: DistMoeRuntime.Config | None,
        debug: DebugConfig,
        reset_kv_cache_on_model_state_dict_load: bool,
        model_state_dict_load_mode: ModelStateDictLoadMode,
        rank: int,
        generator_name: str,
        output_dir: str,
    ):
        self._uses_kv_offload = config.kv_transfer_config is not None
        self._coordinate_kv_offload = self._uses_kv_offload and (
            parallelism.tensor_parallel_degree * parallelism.data_parallel_degree > 1
        )
        if config.kv_transfer_config is not None:
            if config.kv_transfer_config.kv_connector == "MooncakeStoreConnector":
                cache_prefix = config.kv_transfer_config.kv_connector_extra_config.get(
                    "cache_prefix"
                )
                if not isinstance(cache_prefix, str) or not cache_prefix.strip():
                    raise ValueError("Mooncake Store requires a non-empty cache_prefix")
            if reset_kv_cache_on_model_state_dict_load:
                raise ValueError(
                    "KV offload requires reset_kv_cache_on_model_state_dict_load=False: "
                    "asynchronous saves can still hold GPU blocks after requests finish. "
                    "Policy-version salts isolate KV without resetting the cache."
                )
        if (
            config.session_kv_holder is not None
            and reset_kv_cache_on_model_state_dict_load
        ):
            raise ValueError(
                "session_kv_holder needs reset_kv_cache_on_model_state_dict_load=False: vLLM "
                "cannot reset the prefix cache while blocks are held"
            )
        self.config = config
        # Initial random weights must never cache KV as trainer version 0.
        self._loaded_policy_version: int | None = None
        self._reset_kv_cache_on_model_state_dict_load = (
            reset_kv_cache_on_model_state_dict_load
        )
        self._setup(
            model,
            parallelism=parallelism,
            precision=precision,
            checkpointer=checkpointer,
            dist_moe_runtime=dist_moe_runtime,
            batch_invariant=debug.batch_invariant,
            model_state_dict_load_mode=model_state_dict_load_mode,
        )
        engine_args = self._build_engine_args(
            model,
            parallelism=parallelism,
            model_path=model_path,
            max_num_seqs=max_num_seqs,
            param_dtype=precision.param_dtype,
            gpu_memory_limit=gpu_memory_limit,
            max_num_batched_tokens=max_num_batched_tokens,
            seed=debug.seed,
            generator_name=generator_name,
        )
        tp_degree = parallelism.tensor_parallel_degree
        stat_loggers = None
        if rank % tp_degree == 0:
            stat_loggers = self._build_stat_loggers(
                StatLoggerContext(
                    rank=rank,
                    tp_rank=0,
                    dp_rank=rank // tp_degree,
                    generator_name=generator_name,
                    output_dir=output_dir,
                )
            )
        logger.info("Initializing LLMEngine from EngineArgs...")
        self._engine = LLMEngine.from_engine_args(
            engine_args, stat_loggers=stat_loggers
        )
        # With GPU holding enabled, each TP rank holds the same blocks in its own
        # scheduler. Only the engine loop thread mutates session state; TP ranks
        # apply updates in the same order.
        self._session_kv_holder: VLLMSessionKVHolder | None = None
        if config.session_kv_holder is not None:
            scheduler = self._engine.engine_core.engine_core.scheduler
            self._session_kv_holder = config.session_kv_holder.build(
                scheduler=scheduler,
                offload_connector=scheduler.connector
                if self._uses_kv_offload
                else None,
            )

    @staticmethod
    def _setup(
        model: Decoder.Config,
        *,
        parallelism: InferenceParallelismConfig,
        precision: PrecisionConfig,
        checkpointer: CheckpointManager.Config | None,
        dist_moe_runtime: DistMoeRuntime.Config | None,
        batch_invariant: bool,
        model_state_dict_load_mode: ModelStateDictLoadMode,
    ) -> None:
        """Process-wide vLLM setup before the engine build: register the TorchTitan model
        and config parser with vLLM, and select vLLM's model runner."""
        register_to_vllm(
            model,
            parallelism=parallelism,
            precision=precision,
            checkpointer_config=checkpointer,
            dist_moe_runtime=dist_moe_runtime,
            model_state_dict_load_mode=model_state_dict_load_mode,
        )
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        if batch_invariant:
            # The vLLM v2 logprob Triton kernel bypasses the batch-invariant aten
            # overrides; route it through trainer's function to match the trainer exactly.
            force_logprobs_fn_for_batch_invariance()

    def _build_engine_args(
        self,
        model: Decoder.Config,
        *,
        parallelism: InferenceParallelismConfig,
        model_path: str,
        max_num_seqs: int,
        param_dtype: str,
        gpu_memory_limit: float,
        max_num_batched_tokens: int | None,
        seed: int | None,
        generator_name: str,
    ) -> EngineArgs:
        """Build the vLLM ``EngineArgs`` (see ``GenerationEngine`` for the arguments)."""
        enable_ep = parallelism.expert_parallel_degree > 1
        engine_kwargs = dict(
            # ``model`` is the path to the HF checkpoint directory. The
            # config is sourced from TorchTitan's model config via
            # ``config_format=TORCHTITAN_CONFIG_FORMAT`` (no config.json
            # read), but vLLM still uses this path to locate the
            # tokenizer assets and the safetensors weight shards.
            model=model_path,
            trust_remote_code=True,
            # Use the torchtitan custom config parser (registered by
            # register_to_vllm in _setup). It builds PretrainedConfig from
            # model config instead of reading config.json from disk.
            config_format=TORCHTITAN_CONFIG_FORMAT,
            dtype=param_dtype,
            tensor_parallel_size=parallelism.tensor_parallel_degree,
            data_parallel_size=parallelism.data_parallel_degree,
            # NOTE: Monarch launches the generator workers and sets the torch
            # elastic distributed env; with external_launcher, vLLM uses that
            # world to build its process groups. vLLM does not take an
            # explicit EP degree: when this boolean is set, it converts all
            # DP * TP ranks into the expert-parallel group for MoE layers.
            enable_expert_parallel=enable_ep,
            worker_cls=TORCHTITAN_WORKER_CLS,
            # Monarch already spawned TP workers via proc mesh. "external_launcher"
            # tells vLLM to run one worker per process (no subprocess spawning)
            distributed_executor_backend="external_launcher",
            gpu_memory_utilization=gpu_memory_limit,
            enforce_eager=self.config.cuda_graph.mode == "NONE",
            kv_transfer_config=self.config.kv_transfer_config,
            attention_config=AttentionConfig(backend=vllm_attention_backend(model)),
            # Enables RequestOutput.metrics, so generator metrics can be returned
            disable_log_stats=False,
            enable_cumem_allocator=False,
            # Token-in-token-out: prompts and outputs are token ids, so vLLM
            # needs no tokenizer. This also drops the tokenizer's eos_token_id
            # as a stop; the generation config's eos ids are dropped by
            # ignore_eos in _build_sampling_params.
            skip_tokenizer_init=True,
        )
        engine_kwargs["max_model_len"] = model.max_context_length
        # Return logprobs of the distribution vLLM samples from (after temperature). vLLM's default
        # returns the raw model's logprobs, before temperature.
        engine_kwargs["logprobs_mode"] = "processed_logprobs"
        engine_kwargs["max_num_seqs"] = max_num_seqs
        if max_num_batched_tokens is not None:
            engine_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
        # Smaller `priority` runs first and is preempted last; we pass `EngineRequest.group_id`
        # (-1 for all validation), so older groups have higher priority (see its docstring).
        # TODO: validation (priority -1) runs before training. Revisit once validation overlaps training.
        engine_kwargs["scheduling_policy"] = "priority"
        # Which sliding-window / Mamba (GDN) state blocks are hashed into the prefix
        # cache; full-attention groups hash every block regardless.
        # - 0 (vLLM default since v0.29): only the replay boundary (the last block
        #   boundary before the prompt's final token) and shared-prefix junctions.
        #   Other prompt blocks and every block crossed during decode are freed
        #   unhashed, so a multi-turn rollout's next turn, whose prompt contains the
        #   previous completion, cannot reuse anything past the previous prompt.
        # - None: every full block boundary, as before v0.29. Blocks are still freed
        #   when the request moves past them; hashed ones stay reusable until evicted.
        engine_kwargs["prefix_cache_retention_interval"] = None
        if self.config.kv_transfer_config is not None:
            extra = self.config.kv_transfer_config.kv_connector_extra_config
            if self.config.kv_transfer_config.kv_connector == "MooncakeStoreConnector":
                extra = dict(extra)
                # This option names an IPC socket, not a network port. Independent
                # generators on the same host must not bind/unlink each other's socket.
                extra.setdefault(
                    "lookup_rpc_port",
                    sha256(
                        f"{extra['cache_prefix']}:{generator_name}".encode()
                    ).hexdigest()[:16],
                )
            engine_kwargs["kv_transfer_config"] = replace(
                self.config.kv_transfer_config,
                kv_connector_extra_config=extra,
                kv_connector_module_path=(
                    "torchtitan.rl.generator.vllm.mooncake"
                    if self.config.kv_transfer_config.kv_connector
                    == "MooncakeStoreConnector"
                    else "torchtitan.rl.generator.vllm.simple_cpu_offload"
                ),
            )
            if self._coordinate_kv_offload:
                engine_kwargs[
                    "scheduler_cls"
                ] = "torchtitan.rl.generator.vllm.cpu_kv_offload.create_kv_offload_scheduler"
        # FA2 requires block_size to be a multiple of 256
        if not has_cuda_capability(9, 0):
            engine_kwargs["block_size"] = 256
        vllm_compilation_config = self.config.cuda_graph.get_vllm_compilation_config(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            expert_sequence_parallel_size=parallelism.expert_sequence_parallel_size,
            enable_sequence_parallel=parallelism.enable_sequence_parallel,
        )
        if vllm_compilation_config is not None:
            engine_kwargs["compilation_config"] = vllm_compilation_config
        if seed is not None:
            engine_kwargs["seed"] = seed
        add_extra_vllm_engine_args(engine_kwargs, self.config.extra_engine_args)
        return EngineArgs(**engine_kwargs)

    def _build_stat_loggers(self, context: StatLoggerContext) -> list | None:
        """Build the vLLM stat logger factories, or None without ``config.stat_logger``."""
        stat_logger_config = self.config.stat_logger
        if stat_logger_config is None:
            logger.info(
                "VllmOtelStatLogger inactive because stat_logger=None. To record vLLM "
                "metrics, set VLLMEngine.Config.stat_logger to VllmOtelStatLogger.Config() "
                "and set OTEL_METRICS_EXPORTER=jsonl or otlp"
            )
            return None

        def build_stat_logger(vllm_config, engine_index):
            return stat_logger_config.build(
                vllm_config=vllm_config,
                engine_index=engine_index,
                context=context,
            )

        return [build_stat_logger]

    @property
    def data_parallel_size(self) -> int:
        """The data-parallel size vLLM derived for this engine."""
        return self._engine.vllm_config.parallel_config.data_parallel_size

    @property
    def data_parallel_rank(self) -> int:
        """The data-parallel rank vLLM derived for this engine."""
        return self._engine.vllm_config.parallel_config.data_parallel_rank

    @property
    def holds_session_kv(self) -> bool:
        return self._session_kv_holder is not None

    @property
    def prefetched_model_state_dict(self) -> dict[str, torch.Tensor] | None:
        """The pinned CPU buffers a model state dict is prefetched into, and loaded from by
        a model state dict load. Safe to fill from any thread while no load runs. None with
        ``model_state_dict_load_mode="gpu_fetch"``."""
        return self._get_model()._prefetched_model_state_dict

    def add_requests(
        self, requests: list[EngineRequest]
    ) -> list[tuple[str, Exception]]:
        """Admit ``requests`` in order and return the ``(request_id, error)`` of each request
        vLLM rejected."""
        rejected_requests: list[tuple[str, Exception]] = []
        for request in requests:
            prompt = {"prompt_token_ids": request.prompt_token_ids}
            if self._uses_kv_offload:
                if self._loaded_policy_version is None:
                    rejected_requests.append(
                        (
                            request.request_id,
                            ValueError("Load trainer weights before using KV offload"),
                        )
                    )
                    continue
                # Offload isolates exact weights, including across replicas.
                # A group can outlive a weight update, so its minimum is not enough.
                prompt["cache_salt"] = str(self._loaded_policy_version)
            elif not self._reset_kv_cache_on_model_state_dict_load:
                # Preserve the existing stale-KV semantics within a group.
                prompt["cache_salt"] = str(request.min_policy_version)
            # vLLM validates a request (prompt length, sampling params, token ids) as it
            # admits it: render_cmpl and InputProcessor.process_inputs raise
            # VLLMClientError (e.g. VLLMValidationError, not a ValueError subclass) or
            # ValueError before LLMEngine.add_request registers the request with the
            # output processor and engine core, so a rejected request leaves no engine
            # state (recheck on a vLLM upgrade). The checks are deterministic, so every TP
            # rank of this replica rejects the same requests.
            try:
                # render_cmpl is vLLM's input pipeline (tokenize is a no-op for tokenized
                # prompts); the high-level entry stays resilient to vLLM internals vs
                # vllm.inputs.tokens_input.
                [engine_input] = self._engine.renderer.render_cmpl([prompt])
                sampling_params = _build_sampling_params(request.sampling)
                if self._coordinate_kv_offload:
                    # shared_request_id is a copy of Titan's request.request_id,
                    # shared by all TP ranks. vLLM may randomize its internal ID;
                    # this metadata preserves the original ID so the scheduler
                    # can match KV completions across ranks.
                    sampling_params.extra_args = {
                        "kv_transfer_params": {"shared_request_id": request.request_id}
                    }
                # Track vLLM's internal ID, which may include its random suffix.
                internal_request_id = self._engine.add_request(
                    request_id=request.request_id,
                    prompt=engine_input,
                    params=sampling_params,
                    arrival_time=request.arrival_time,
                    priority=max(request.group_id, -1),
                )
                if self._session_kv_holder is not None:
                    self._session_kv_holder.track(
                        internal_request_id,
                        request.routing_session_id,
                        request.group_id,
                    )
            except (ValueError, VLLMClientError) as exc:
                rejected_requests.append((request.request_id, exc))
        return rejected_requests

    def has_unfinished_requests(self) -> bool:
        """Whether requests or their remaining KV work need more engine steps."""
        if not self._uses_kv_offload:
            return self._engine.has_unfinished_requests()

        # All ranks enter this check before stepping and use the result directly.
        has_work = self._has_unfinished_local_requests()
        if self._coordinate_kv_offload:
            work = torch.tensor(has_work, dtype=torch.int32)
            dist.all_reduce(
                work, op=dist.ReduceOp.MAX, group=get_world_group().cpu_group
            )
            has_work = bool(work.item())
        return has_work

    def _has_unfinished_local_requests(self) -> bool:
        # LLMEngine tracks user outputs only; the in-process scheduler also tracks
        # finish-time saves and the completion steps that release their GPU blocks.
        # TODO(acisseJZhong): expose scheduler work through the public LLMEngine interface:
        # https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/llm_engine.py
        core = self._engine.engine_core.engine_core
        return (
            self._engine.output_processor.has_unfinished_requests()
            or core.scheduler.has_requests()
            or bool(core.batch_queue)
            or (
                self._session_kv_holder is not None
                and self._session_kv_holder.has_pending_work
            )
        )

    def step(self) -> list[FinishedRequest]:
        """Run one engine step, a TP collective, and return the requests it finished."""
        if self._coordinate_kv_offload and self._engine.dp_group is not None:
            core = self._engine.engine_core.engine_core
            # Idle DP replicas must still join the EP forward. Include KV-only
            # steps, which upstream executes as dummy forwards under DP.
            self._engine.has_unfinished_requests_dp(core.scheduler.has_requests())
            if self._engine.should_execute_dummy_batch:
                self._engine.should_execute_dummy_batch = False
                self._engine.engine_core.execute_dummy_batch()
            # Continue into step even after a dummy forward: this DP replica may
            # still have queued async outputs and KV completions to drain.
        with torch.no_grad():
            request_outputs = self._engine.step()
        if self._session_kv_holder is not None:
            self._session_kv_holder.after_step()
        return [_finished_request(request_output) for request_output in request_outputs]

    def release_sessions(self, session_ids: list[str], group_ids: list[int]) -> None:
        if not session_ids and not group_ids:
            return
        if self._session_kv_holder is not None:
            self._session_kv_holder.release(session_ids, group_ids)

    def log_stats(self, step: int) -> None:
        if self._session_kv_holder is not None:
            logger.info(
                "Session KV at step %d: %d sessions held, %d evicted so far",
                step,
                self._session_kv_holder.num_sessions,
                self._session_kv_holder.num_evicted,
            )

    def load_model_state_dict(
        self,
        fetch: Callable[[dict[str, torch.Tensor]], Iterable[str]],
        *,
        policy_version: int,
    ) -> None:
        """Fetch the model state dict into the model's weights one FSDP group at a time, and
        reset the KV cache when configured."""
        self._get_model().load_model_state_dict(fetch)
        self._loaded_policy_version = policy_version
        if self._reset_kv_cache_on_model_state_dict_load:
            # Always reset running requests too: the only reason to reset is a strict
            # recompute under the new weights. Keeping running requests' KV while hiding
            # old KV from new requests is already what the default (no reset) does via the salt.
            self._engine.reset_prefix_cache(
                reset_running_requests=True,
            )

    def shutdown(self) -> None:
        """Close renderer-local resources.

        With `external_launcher`, vLLM reuses the process group and actor lifetime that
        Monarch owns. Calling vLLM's internal `engine_core.shutdown()` can block while
        Monarch is also trying to stop the same actor mesh, so process teardown is left to
        `ProcMesh.stop()`.
        """
        renderer = getattr(self._engine, "renderer", None)
        if renderer is not None:
            logger.info("Shutting down vLLM renderer")
            renderer.shutdown()
        self._session_kv_holder = None

    def _get_model(self):
        """The ``VLLMModelWrapper`` the engine runs."""
        return self._engine.model_executor.driver_worker.get_model()


def _build_sampling_params(sampling: SamplingConfig) -> SamplingParams:
    """Translate a `SamplingConfig` into vLLM `SamplingParams` (n=1).

    ``seed`` and ``stop_token_ids`` are carried on the ``SamplingConfig``
    (the controller fills ``stop_token_ids`` and the rollouter offsets
    ``seed`` per sample), so each sample in a group is a distinct ``n=1``
    request that stays diverse and bitwise-reproducible.

    The engine loads no tokenizer, so its ``eos_token_id`` is not a stop,
    but vLLM still adds the generation config's ``eos_token_id`` (checkpoint
    ``generation_config.json`` or the ``vllm_registry`` HF config).
    ``ignore_eos`` turns that off, so the renderer's ``stop_token_ids``
    (which include EOS) are the only stops.
    """
    return SamplingParams(
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        max_tokens=sampling.max_tokens,
        n=1,  # always expects a single sample per request. Caller can call N times.
        stop_token_ids=sampling.stop_token_ids,
        # Drops the generation config's eos ids, which vLLM merges into
        # stop_token_ids even with skip_tokenizer_init.
        ignore_eos=True,
        seed=sampling.seed,
        logprobs=0,  # return only the sampled token's logprob (for the GRPO ratio)
        # Token ids in, token ids and logprob floats out: stops are token ids and nothing reads
        # text, so skip vLLM's per-token detokenization and per-token logprob dicts.
        detokenize=False,
        flat_logprobs=True,
        # Return each request's result once, when it is fully done, instead of streaming partial
        # outputs as tokens arrive.
        # TODO(async-rl): use RequestOutputKind.CUMULATIVE for exact per-token
        #   (start_token, version) boundaries; today we keep only the per-turn min/max.
        output_kind=RequestOutputKind.FINAL_ONLY,
    )


def _finished_request(request_output: RequestOutput) -> FinishedRequest:
    """Convert a finished ``RequestOutput`` into a ``FinishedRequest``."""
    # We enforce n=1 in sampling params -> exactly one CompletionOutput per finished request
    # Here we just sanity check it (a single engine.step may still finish several requests).
    if len(request_output.outputs) != 1:
        raise ValueError(
            f"expected n=1 (one sample per request), got "
            f"{len(request_output.outputs)} for {request_output.request_id}"
        )

    # flat_logprobs=True: vLLM returns logprobs as plain lists instead of one dict per token.
    # logprobs=0 keeps only the sampled token, so `.logprobs` has exactly one float per generated token.
    completion_output = request_output.outputs[0]
    return FinishedRequest(
        request_id=request_output.request_id,
        token_ids=list(completion_output.token_ids),
        token_logprobs=list(completion_output.logprobs.logprobs),
        finish_reason=completion_output.finish_reason,
        metrics_inputs=_extract_request_metrics_inputs(request_output),
    )


def _extract_request_metrics_inputs(
    request_output: RequestOutput,
) -> RequestMetricsInputs:
    """Pull the raw metric inputs off a finished ``RequestOutput``.

    For an `add_request` call, vLLM returns a RequestOutput carrying a single
    `RequestStateStats`.
    """
    stats = request_output.metrics
    if stats is None:
        return RequestMetricsInputs(
            num_cached_tokens=request_output.num_cached_tokens, has_stats=False
        )
    return RequestMetricsInputs(
        num_cached_tokens=request_output.num_cached_tokens,
        has_stats=True,
        queued_ts=stats.queued_ts,
        scheduled_ts=stats.scheduled_ts,
        first_token_ts=stats.first_token_ts,
        last_token_ts=stats.last_token_ts,
        first_token_latency=stats.first_token_latency,
        num_generation_tokens=stats.num_generation_tokens,
        num_preemptions=stats.num_preemptions,
    )
