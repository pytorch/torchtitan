# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""vLLM inference engine for the RL generator: every vLLM call the generator makes."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.config import AttentionConfig
from vllm.exceptions import VLLMClientError
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.config import DebugConfig
from torchtitan.models.common.attention import FlexInnerAttention, VarlenInnerAttention
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.dist_moe import DistMoeRuntime
from torchtitan.rl.generator.engine import GenerationEngine
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.types import (
    EngineRequest,
    FinishedRequest,
    RequestMetricsInputs,
    SamplingConfig,
    vllm_priority,
)
from torchtitan.rl.generator.vllm.batch_invariance import (
    force_logprobs_fn_for_batch_invariance,
)
from torchtitan.rl.generator.vllm.config import VLLMCudaGraphConfig
from torchtitan.rl.generator.vllm.registry import (
    register_to_vllm,
    TORCHTITAN_CONFIG_FORMAT,
    TORCHTITAN_WORKER_CLS,
)
from torchtitan.rl.generator.vllm.session_kv_holder import VLLMSessionKVHolder
from torchtitan.rl.generator.vllm.stat_logger import (
    StatLoggerContext,
    VllmOtelStatLogger,
)
from torchtitan.rl.observability import metrics as m
from torchtitan.tools.utils import has_cuda_capability

logger = logging.getLogger(__name__)


def add_extra_vllm_engine_args(
    engine_kwargs: dict[str, Any], extra_engine_args: dict[str, Any]
) -> None:
    """Add ``VLLMEngine.Config.extra_engine_args`` to ``engine_kwargs`` in place.

    Raises ``ValueError`` for keys TorchTitan already set in ``engine_kwargs`` or
    derives from a config field, so they cannot silently diverge from it.
    """
    # EngineArgs that TorchTitan sets only when the mapped config field is set (or
    # derives from it), so they may be absent from engine_kwargs.
    config_field_by_engine_arg = {
        "max_num_batched_tokens": "Generator.Config.max_num_batched_tokens",
        "max_logprobs": "Generator.Config.sampling.num_topk_logprobs",
        "compilation_config": "VLLMEngine.Config.cuda_graph",
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
        `VLLMSessionKVHolder`). None: off."""

    def __init__(
        self,
        config: Config,
        *,
        model: Decoder.Config,
        parallelism: InferenceParallelismConfig,
        model_path: str,
        max_num_seqs: int,
        model_dtype: str,
        gpu_memory_limit: float,
        max_num_batched_tokens: int | None,
        num_topk_logprobs: int,
        checkpointer: CheckpointManager.Config | None,
        dist_moe_runtime: DistMoeRuntime.Config | None,
        debug: DebugConfig,
        reset_kv_cache_on_model_state_dict_load: bool,
        rank: int,
        generator_name: str,
        output_dir: str,
    ):
        if (
            config.session_kv_holder is not None
            and reset_kv_cache_on_model_state_dict_load
        ):
            raise ValueError(
                "session_kv_holder needs reset_kv_cache_on_model_state_dict_load=False: vLLM "
                "cannot reset the prefix cache while blocks are held"
            )
        self.config = config
        self._reset_kv_cache_on_model_state_dict_load = (
            reset_kv_cache_on_model_state_dict_load
        )
        self._setup(
            model,
            parallelism=parallelism,
            checkpointer=checkpointer,
            dist_moe_runtime=dist_moe_runtime,
            batch_invariant=debug.batch_invariant,
            return_routed_experts=config.return_routed_experts,
        )
        engine_args = self._build_engine_args(
            model,
            parallelism=parallelism,
            model_path=model_path,
            max_num_seqs=max_num_seqs,
            model_dtype=model_dtype,
            gpu_memory_limit=gpu_memory_limit,
            max_num_batched_tokens=max_num_batched_tokens,
            num_topk_logprobs=num_topk_logprobs,
            seed=debug.seed,
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
        # Every rank holds the same blocks in its own scheduler; only the engine loop thread
        # touches them.
        self._session_kv_holder: VLLMSessionKVHolder | None = None
        if config.session_kv_holder is not None:
            self._session_kv_holder = config.session_kv_holder.build(
                scheduler=self._engine.engine_core.engine_core.scheduler
            )
        # The holder's running totals at the last `session_kv_metrics`.
        self._session_kv_totals: dict[str, int] = {}

    @staticmethod
    def _setup(
        model: Decoder.Config,
        *,
        parallelism: InferenceParallelismConfig,
        checkpointer: CheckpointManager.Config | None,
        dist_moe_runtime: DistMoeRuntime.Config | None,
        batch_invariant: bool,
        return_routed_experts: bool,
    ) -> None:
        """Process-wide vLLM setup before the engine build: register the TorchTitan model
        and config parser with vLLM, and select vLLM's model runner."""
        register_to_vllm(
            model,
            parallelism=parallelism,
            checkpointer_config=checkpointer,
            dist_moe_runtime=dist_moe_runtime,
            return_routed_experts=return_routed_experts,
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
        model_dtype: str,
        gpu_memory_limit: float,
        max_num_batched_tokens: int | None,
        num_topk_logprobs: int,
        seed: int | None,
    ) -> EngineArgs:
        """Build the vLLM ``EngineArgs`` (see ``GenerationEngine`` for the arguments)."""
        attention = model.first_full_attention
        assert attention is not None, "No full attention layer found in the model spec."
        inner_attention_config = attention.inner_attention
        assert isinstance(
            inner_attention_config,
            (VarlenInnerAttention.Config, FlexInnerAttention.Config),
        ), "Only varlen and flex attention backends are allowed."

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
            dtype=model_dtype,
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
            attention_config=AttentionConfig(
                backend=(
                    AttentionBackendEnum.FLEX_ATTENTION
                    if isinstance(inner_attention_config, FlexInnerAttention.Config)
                    else AttentionBackendEnum.CUSTOM
                ),
            ),
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
        if num_topk_logprobs > 0:
            engine_kwargs["max_logprobs"] = num_topk_logprobs
        engine_kwargs["max_num_seqs"] = max_num_seqs
        if max_num_batched_tokens is not None:
            engine_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
        if self.config.return_routed_experts:
            # TODO: vLLM after 0b7f11a1ee (2026-09-21) captures routed experts only under Model
            # Runner V2 (forced off in `_setup`); the router hook works unchanged there.
            engine_kwargs["enable_return_routed_experts"] = True
        # Smaller `priority` runs first and is preempted last; `vllm_priority` sets it from the
        # group id (see `EngineRequest.group_id`).
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
        # FA2 requires block_size to be a multiple of 256
        if not has_cuda_capability(9, 0):
            engine_kwargs["block_size"] = 256
        vllm_compilation_config = self.config.cuda_graph.get_vllm_compilation_config(
            max_num_seqs=max_num_seqs,
            max_model_len=model.max_context_length,
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
    def prefetched_model_state_dict(self) -> dict[str, torch.Tensor]:
        """The pinned CPU buffers a model state dict is prefetched into, and loaded from by
        ``load_prefetched_model_state_dict``. Safe to fill from any thread while no load runs."""
        return self._get_model()._prefetched_model_state_dict

    def add_requests(
        self, requests: list[EngineRequest]
    ) -> list[tuple[str, Exception]]:
        """Admit ``requests`` in order and return the ``(request_id, error)`` of each request
        vLLM rejected."""
        rejected_requests: list[tuple[str, Exception]] = []
        for request in requests:
            prompt = {"prompt_token_ids": request.prompt_token_ids}
            if not self._reset_kv_cache_on_model_state_dict_load:
                # Salt by the pinned version so a request only reuses KV
                # computed under that version.
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
                # vLLM appends a random suffix to the id; the scheduler knows only that one.
                internal_request_id = self._engine.add_request(
                    request_id=request.request_id,
                    prompt=engine_input,
                    params=_build_sampling_params(request.sampling),
                    arrival_time=request.arrival_time,
                    priority=vllm_priority(request.group_id),
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

    def abort_requests(self, request_ids: list[str]) -> list[str]:
        """Abort the requests among `request_ids` this replica is running and free their KV;
        return their ids."""
        # vLLM maps each external id to its internal ones and drops the entry when the request
        # finishes, so the map says which ids still run here (vLLM internals; recheck on a vLLM
        # upgrade). The sync engine emits no output for an aborted request.
        external_req_ids = self._engine.output_processor.external_req_ids
        running = [
            request_id for request_id in request_ids if request_id in external_req_ids
        ]
        # One id per call: vLLM frees one call's ids in set order, which differs per process. TP ranks
        # that free blocks in different orders cache different prefixes and schedule different batches.
        for request_id in running:
            self._engine.abort_request([request_id])
        return running

    def has_unfinished_requests(self) -> bool:
        """Whether any admitted request is still running."""
        return self._engine.has_unfinished_requests()

    def step(self) -> list[FinishedRequest]:
        """Run one engine step, a TP collective, and return the requests it finished."""
        with torch.no_grad():
            request_outputs = self._engine.step()
        if self._session_kv_holder is not None:
            self._session_kv_holder.after_step()
        return [_finished_request(request_output) for request_output in request_outputs]

    def release_sessions(self, session_ids: list[str], group_ids: list[int]) -> None:
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

    def session_kv_metrics(self) -> list[m.Metric]:
        """This replica's session-KV metrics: levels now, counts since the last call.

        Example:

            call 1: holder.num_evicted 3, num_sessions 80  ->  evicted_sessions 3, held_sessions 80
            call 2: holder.num_evicted 5, num_sessions 78  ->  evicted_sessions 2, held_sessions 78

        Levels get Mean and Max over replicas (the typical and the fullest one), counts a Sum.
        """
        holder = self._session_kv_holder
        if holder is None:
            return []
        scheduler = self._engine.engine_core.engine_core.scheduler
        levels = {
            "gpu_used_fraction": scheduler.kv_cache_manager.usage,
            "held_sessions": holder.num_sessions,
        }
        totals = {
            "gpu_hit_prefix_tokens": holder.gpu_hit_prefix_tokens,
            "recomputed_prefix_tokens": holder.recomputed_prefix_tokens,
            "evicted_sessions": holder.num_evicted,
            "stall_evicted_sessions": holder.num_stall_evicted,
        }
        counts = {
            key: total - self._session_kv_totals.get(key, 0)
            for key, total in totals.items()
        }
        self._session_kv_totals = totals
        return [
            *(
                m.Metric(f"generator/session_kv/{key}", metric_type(value))
                for key, value in levels.items()
                for metric_type in (m.Mean, m.Max)
            ),
            *(
                m.Metric(f"generator/session_kv/{key}", m.Sum(value))
                for key, value in counts.items()
            ),
        ]

    def load_prefetched_model_state_dict(self) -> None:
        """Copy the prefetched model state dict from pinned CPU memory into the model, and
        reset the KV cache when configured."""
        model = self._get_model()
        model.load_state_dict(model._prefetched_model_state_dict)
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
        routed_experts_prompt_start=sampling.routed_experts_prompt_start,
        # The sampled token's logprob (for the GRPO ratio), plus the top-k if requested.
        logprobs=sampling.num_topk_logprobs,
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
    # logprobs=0 (num_topk_logprobs=0) keeps only the sampled token: one float per generated token.
    completion_output = request_output.outputs[0]
    flat_logprobs = completion_output.logprobs
    token_logprobs = list(flat_logprobs.logprobs)
    topk_token_ids = topk_logprobs = None
    num_tokens = len(completion_output.token_ids)
    if len(token_logprobs) > num_tokens:
        # logprobs=k > 0: each token holds k + 1 entries, [sampled, top-1, ..., top-k].
        # k is per request (validation asks for 0), so read it from the layout.
        row_width = len(token_logprobs) // num_tokens
        topk_token_ids = np.asarray(flat_logprobs.token_ids, dtype=np.int32).reshape(
            num_tokens, row_width
        )[:, 1:]
        topk_logprobs = np.asarray(token_logprobs, dtype=np.float32).reshape(
            num_tokens, row_width
        )[:, 1:]
        token_logprobs = token_logprobs[::row_width]
    return FinishedRequest(
        request_id=request_output.request_id,
        token_ids=list(completion_output.token_ids),
        token_logprobs=token_logprobs,
        finish_reason=completion_output.finish_reason,
        metrics_inputs=_extract_request_metrics_inputs(request_output),
        routed_expert_ids=completion_output.routed_experts,
        topk_token_ids=topk_token_ids,
        topk_logprobs=topk_logprobs,
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
