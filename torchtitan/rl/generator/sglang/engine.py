# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SGLang inference engine for the RL generator: every SGLang call the generator makes."""

from __future__ import annotations

import copy
import dataclasses
import gc
import logging
import os
import time
from array import array
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.distributed as dist
from sglang.srt.managers.io_struct import (
    AbortReq,
    BatchTokenIDOutput,
    ContinueGenerationReqInput,
    PauseGenerationReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.req_types import FINISH_ABORT
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.scheduler import (
    allocate_distinct_stream,
    is_war_barrier_enabled,
    Scheduler,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import publish, SpawnRanks
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.config import DebugConfig, PrecisionConfig
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.dist_moe import DistMoeRuntime
from torchtitan.rl.generator.engine import GenerationEngine, ModelStateDictLoadMode
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.sglang.attention import TORCHTITAN_VARLEN_ATTENTION_BACKEND
from torchtitan.rl.generator.sglang.batch_invariance import (
    force_trainer_logprobs_for_batch_invariance,
)
from torchtitan.rl.generator.sglang.model_wrapper import replace_sglang_layer_configs
from torchtitan.rl.generator.sglang.registry import (
    CONTEXT_LENGTH_MARGIN,
    register_to_sglang,
    TORCHTITAN_CONFIG_PARSER,
)
from torchtitan.rl.generator.types import (
    EngineRequest,
    FinishedRequest,
    RequestMetricsInputs,
    SamplingConfig,
)

logger = logging.getLogger(__name__)

# The ServerArgs SGLangEngine sets itself, which extra_server_args cannot set.
_TORCHTITAN_SERVER_ARGS = frozenset(
    {
        "model_path",
        "model_config_parser",
        "skip_tokenizer_init",
        "dtype",
        "tp_size",
        "mem_fraction_static",
        "max_running_requests",
        "context_length",
        "attention_backend",
        "enable_priority_scheduling",
        "schedule_low_priority_values_first",
        "enable_session_radix_cache",
        "disable_cuda_graph",
        "disable_prefill_cuda_graph",
        "cuda_graph_backend_prefill",
        "skip_server_warmup",
        "log_level",
        "chunked_prefill_size",
        "random_seed",
        "enable_deterministic_inference",
        "disable_custom_all_reduce",
    }
)


class SGLangEngine(GenerationEngine):
    """The generator's SGLang scheduler, run in-process behind the calls the engine loop makes.

    Each rank builds its own SGLang ``Scheduler`` (no tokenizer, detokenizer or
    scheduler subprocesses) and steps it once per ``step()``; SGLang's TP schedulers make
    identical decisions, as the engine loop's SPMD contract requires. ``step()`` runs one
    iteration of SGLang's overlap event loop, so the CPU schedules the next batch while
    the GPU runs the current one, and returns the requests whose results that iteration
    processed.

    See ``GenerationEngine`` for the build arguments and the threading contract.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GenerationEngine.Config):
        cuda_graph: Literal[
            "none", "decode", "decode_and_prefill"
        ] = "decode_and_prefill"
        """CUDA graphs SGLang captures:

        - ``"none"``: everything runs eagerly.
        - ``"decode"``: decode batches; prefill runs eagerly. Starts faster.
        - ``"decode_and_prefill"`` (default): also prefill batches, as SGLang's
          breakable CUDA graphs, which run attention eagerly between graph segments.
          Without them, prefill is bound by the eager model's CPU overhead.
        """

        extra_server_args: dict[str, Any] = field(default_factory=dict)
        """Additional keyword arguments passed to SGLang's ``ServerArgs``, for options
        without a dedicated field (e.g. SGLang's hierarchical cache). Keys that
        TorchTitan sets itself raise a ``ValueError``."""

        def __post_init__(self) -> None:
            owned_keys = sorted(self.extra_server_args.keys() & _TORCHTITAN_SERVER_ARGS)
            if owned_keys:
                raise ValueError(
                    "extra_server_args cannot set ServerArgs that TorchTitan controls: "
                    f"{owned_keys}"
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
        del generator_name, output_dir
        if parallelism.data_parallel_degree != 1:
            # TODO(https://github.com/sgl-project/sglang/issues/48476): SGLang builds
            # one torch world per data-parallel replica; run DP replicas in one world by
            # building SGLang's TP groups over it.
            raise ValueError(
                "SGLangEngine supports data_parallel_degree=1 only; run more "
                "generators to scale out"
            )
        if parallelism.expert_parallel_degree != 1 or dist_moe_runtime is not None:
            raise ValueError("SGLangEngine does not support MoE expert parallelism yet")
        if parallelism.enable_sequence_parallel:
            raise ValueError("SGLangEngine does not support sequence parallelism")
        # Raises for layers SGLang cannot run, before any process group or SGLang state.
        model = replace_sglang_layer_configs(copy.deepcopy(model))
        self.config = config
        self._reset_kv_cache_on_model_state_dict_load = (
            reset_kv_cache_on_model_state_dict_load
        )
        # SGLang honors per-request seeds only in its deterministic mode, which
        # _build_server_args turns on with DebugConfig.seed.
        self._honors_request_seeds = debug.seed is not None
        self._max_context_length = model.max_context_length
        if not dist.is_initialized():
            # As vLLM's external_launcher: the actor framework launched this process
            # with the torch elastic env (RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR,
            # MASTER_PORT); SGLang then builds its groups over this world.
            torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
            dist.init_process_group(init_method="env://")
        if dist.get_world_size() != parallelism.tensor_parallel_degree:
            raise ValueError(
                f"SGLangEngine needs one process per TP rank, got world size "
                f"{dist.get_world_size()} for tensor_parallel_degree "
                f"{parallelism.tensor_parallel_degree}"
            )

        if debug.seed is not None:
            # TODO(https://github.com/sgl-project/sglang/issues/48474): SGLang's
            # deterministic mode (needed for seeded sampling, see _build_server_args)
            # also overrides aten matmuls and reductions with SGLang's batch-invariant
            # kernels, process-wide. Batch invariance is DebugConfig.batch_invariant's,
            # with the trainer's kernels; skip SGLang's.
            ModelRunner.maybe_enable_batch_invariant_mode = lambda self: None
        if debug.batch_invariant:
            force_trainer_logprobs_for_batch_invariance()
        register_to_sglang(
            model,
            parallelism=parallelism,
            precision=precision,
            checkpointer_config=checkpointer,
            model_state_dict_load_mode=model_state_dict_load_mode,
        )
        server_args = self._build_server_args(
            model,
            parallelism=parallelism,
            model_path=model_path,
            max_num_seqs=max_num_seqs,
            param_dtype=precision.param_dtype,
            gpu_memory_limit=gpu_memory_limit,
            max_num_batched_tokens=max_num_batched_tokens,
            debug=debug,
        )
        publish(
            server_args,
            role="scheduler",
            ranks=SpawnRanks(
                world_rank=rank,
                dp_rank=None,
                gpu_id=torch.accelerator.current_device_index(),
            ),
        )
        logger.info("Building the SGLang scheduler...")
        self._scheduler = Scheduler(server_args, PortArgs.init_new(server_args))
        # TODO(https://github.com/sgl-project/sglang/issues/48474): SGLang has no option
        # to drop the checkpoint generation config's EOS ids it adds to every request's
        # stops; the renderer's stop tokens (EOS included) are the only stops.
        self._scheduler.model_config.hf_eos_token_id = set()
        # TODO(https://github.com/sgl-project/sglang/issues/48473): SGLang has no
        # in-process output sink; stand in for the sockets to its detokenizer and
        # tokenizer processes, so results and aborts come here.
        self._outputs = _OutputCollector()
        self._scheduler.output_streamer.send_to_detokenizer = self._outputs
        self._scheduler.ipc_channels = dataclasses.replace(
            self._scheduler.ipc_channels, send_to_tokenizer=self._outputs
        )
        # Requests that finished outside `step()` (e.g. retracted during a model state
        # dict load), returned by the next `step()`.
        self._finished: list[FinishedRequest] = []
        # The sessions of each rollout group with a request admitted, for
        # `release_sessions`.
        self._group_sessions: dict[int, set[str]] = {}
        self._init_overlap_event_loop()
        # Exclude the ~1.5M objects SGLang's modules and the build created from garbage
        # collection, as SGLang's freeze_gc does: each full collection (e.g. the engine
        # loop's after a model state dict load) otherwise traverses them for ~1s.
        gc.collect()
        gc.freeze()

    def _build_server_args(
        self,
        model: Decoder.Config,
        *,
        parallelism: InferenceParallelismConfig,
        model_path: str,
        max_num_seqs: int,
        param_dtype: str,
        gpu_memory_limit: float,
        max_num_batched_tokens: int | None,
        debug: DebugConfig,
    ) -> ServerArgs:
        """Build SGLang's ``ServerArgs`` (see ``GenerationEngine`` for the arguments)."""
        server_kwargs: dict[str, Any] = dict(
            # Locates the checkpoint's generation config; the model config comes from
            # the TorchTitan config parser and the weights from the wrapper.
            model_path=model_path,
            model_config_parser=TORCHTITAN_CONFIG_PARSER,
            # Token-in-token-out: prompts and outputs are token ids.
            skip_tokenizer_init=True,
            dtype=param_dtype,
            tp_size=parallelism.tensor_parallel_degree,
            # SGLang sizes weights and the KV pool to this fraction and leaves activations
            # little room on top: at 0.8, Qwen3-0.6B (TP 2) peaks at 81% of GPU memory,
            # as with vLLM's gpu_memory_utilization=0.8.
            mem_fraction_static=gpu_memory_limit,
            max_running_requests=max_num_seqs,
            context_length=model.max_context_length + CONTEXT_LENGTH_MARGIN,
            attention_backend=TORCHTITAN_VARLEN_ATTENTION_BACKEND,
            # Smaller `priority` runs first and is preempted last; EngineRequest.group_id
            # (-1 for all validation) puts older groups first (see its docstring).
            # SGLang breaks ties by when a request entered its waiting queue, and
            # every TP rank enqueues in the same order, so all ranks schedule alike.
            enable_priority_scheduling=True,
            schedule_low_priority_values_first=True,
            # Each finished request references its session's cached prefix until the
            # session is released, so its next turn finds it: eviction takes
            # unreferenced KV first, and referenced KV only when nothing else is left.
            enable_session_radix_cache=True,
            disable_cuda_graph=self.config.cuda_graph == "none",
            # The wrapper loads its weights when SGLang builds it.
            skip_server_warmup=True,
            log_level="warning",
        )
        if self.config.cuda_graph == "decode_and_prefill":
            server_kwargs["cuda_graph_backend_prefill"] = "breakable"
        else:
            server_kwargs["disable_prefill_cuda_graph"] = True
        if max_num_batched_tokens is not None:
            server_kwargs["chunked_prefill_size"] = max_num_batched_tokens
        if debug.seed is not None:
            server_kwargs["random_seed"] = debug.seed
            # SGLang samples with each request's seed only in deterministic mode,
            # which samples with its PyTorch backend.
            server_kwargs["enable_deterministic_inference"] = True
        if debug.batch_invariant:
            # SGLang's custom all-reduce picks its algorithm by message size.
            server_kwargs["disable_custom_all_reduce"] = True
        assert server_kwargs.keys() <= _TORCHTITAN_SERVER_ARGS, (
            server_kwargs.keys() - _TORCHTITAN_SERVER_ARGS
        )
        server_kwargs.update(self.config.extra_server_args)
        return ServerArgs(**server_kwargs)

    def _init_overlap_event_loop(self) -> None:
        """Set up what SGLang's ``run_event_loop`` sets before its overlap loop."""
        # TODO(https://github.com/sgl-project/sglang/issues/48472): mirrors the pinned
        # SGLang version's run_event_loop, private fields included; SGLang has no public
        # API to run its loop one iteration at a time.
        scheduler = self._scheduler
        if not scheduler.enable_overlap:
            raise ValueError("SGLangEngine needs SGLang's overlap scheduler")
        scheduler.enable_skip_finishing_decode = True
        scheduler.result_queue = deque()
        # The schedule stream must differ from the forward stream to overlap them.
        if (
            scheduler.schedule_stream.cuda_stream
            == scheduler.forward_stream.cuda_stream
        ):
            scheduler.schedule_stream = allocate_distinct_stream(
                scheduler.device_module, (scheduler.forward_stream,)
            )
        scheduler._war_barrier_enabled = is_war_barrier_enabled()

    @property
    def data_parallel_size(self) -> int:
        return 1

    @property
    def data_parallel_rank(self) -> int:
        return 0

    @property
    def prefetched_model_state_dict(self) -> dict[str, torch.Tensor] | None:
        return self._get_model().model_state_dict_loader.prefetched_model_state_dict

    def add_requests(
        self, requests: list[EngineRequest]
    ) -> list[tuple[str, Exception]]:
        """Admit ``requests`` in order and return the ``(request_id, error)`` of each request
        SGLang rejected.

        A request with invalid sampling parameters is rejected before SGLang sees it.
        SGLang validates the rest as it admits a request and aborts an invalid one
        instead of raising, so that rejection is the abort, before SGLang schedules it.
        The checks are deterministic, so every TP rank rejects the same requests.
        """
        sglang_requests = []
        rejected: list[tuple[str, Exception]] = []
        for request in requests:
            try:
                sampling_params = self._build_sampling_params(
                    request.sampling, prompt_length=len(request.prompt_token_ids)
                )
            except ValueError as error:
                rejected.append((request.request_id, error))
                continue
            # Without a reset on model state dict load, salt by the pinned version so
            # a request only reuses KV computed under that version.
            cache_salt = (
                None
                if self._reset_kv_cache_on_model_state_dict_load
                else str(request.min_policy_version)
            )
            sglang_request = TokenizedGenerateReqInput(
                rid=request.request_id,
                input_text=None,
                input_ids=array("q", request.prompt_token_ids),
                input_embeds=None,
                mm_inputs=None,
                token_type_ids=None,
                sampling_params=sampling_params,
                return_logprob=True,
                # Only the sampled tokens' logprobs, none of the prompt's.
                logprob_start_len=-1,
                top_logprobs_num=0,
                token_ids_logprob=None,
                stream=False,
                priority=max(request.group_id, -1),
                cache_salt=cache_salt,
                session_id=request.routing_session_id,
            )
            sglang_requests.append(sglang_request)
            self._group_sessions.setdefault(request.group_id, set()).add(
                request.routing_session_id
            )
        with self._scheduler.device_module.StreamContext(
            self._scheduler.schedule_stream
        ):
            self._scheduler.process_input_requests(sglang_requests)
        admitted = {request.rid for request in sglang_requests}
        # SGLang queues a request that fails validation (e.g. a prompt too long) as
        # aborted, to report at its next schedule; take it back out.
        invalid = self._scheduler.waiting_queue.take(
            lambda req: req.rid in admitted and isinstance(req.to_finish, FINISH_ABORT)
        )
        rejected.extend((req.rid, ValueError(req.to_finish.message)) for req in invalid)
        arrival_times = {
            request.request_id: request.arrival_time for request in requests
        }
        queued = set()
        for req in self._scheduler.waiting_queue:
            if req.rid in admitted:
                self._outputs.admit(req, arrival_times[req.rid])
                queued.add(req.rid)
        for finished, abort_message in self._outputs.take():
            if finished.request_id in admitted:
                assert (
                    abort_message is not None
                ), f"{finished.request_id} finished while being admitted"
                rejected.append((finished.request_id, ValueError(abort_message)))
            else:
                self._finished.append(finished)
        # The collector reports a request's metrics from what admit() recorded.
        unqueued = admitted - queued - {request_id for request_id, _ in rejected}
        assert not unqueued, f"not in SGLang's waiting queue: {sorted(unqueued)}"
        return rejected

    def _build_sampling_params(
        self, sampling: SamplingConfig, *, prompt_length: int
    ) -> SamplingParams:
        """Translate a ``SamplingConfig`` into SGLang ``SamplingParams`` (n=1).

        As in vLLM, prompt and output together take at most the model's
        ``max_context_length`` tokens; SGLang's own cap is above it (see
        ``CONTEXT_LENGTH_MARGIN``). SGLang rejects a prompt of ``max_context_length``
        tokens or more.
        """
        if sampling.seed is not None and not self._honors_request_seeds:
            raise ValueError(
                "SGLangEngine samples with per-request seeds only when the generator's "
                "DebugConfig.seed is set"
            )
        params = SamplingParams(
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_new_tokens=min(
                sampling.max_tokens, max(self._max_context_length - prompt_length, 0)
            ),
            stop_token_ids=sampling.stop_token_ids,
            sampling_seed=sampling.seed,
            n=1,
        )
        params.normalize(None)
        params.verify(self._scheduler.model_config.vocab_size)
        return params

    def has_unfinished_requests(self) -> bool:
        """Whether any admitted request is still running, or finished and not yet
        returned by ``step()``."""
        return bool(self._finished) or not self._scheduler.is_fully_idle()

    def step(self) -> list[FinishedRequest]:
        """Run one iteration of SGLang's overlap event loop, a TP collective, and return
        the requests whose results it processed."""
        scheduler = self._scheduler
        with scheduler.device_module.StreamContext(scheduler.schedule_stream):
            _overlap_event_loop_iteration(scheduler)
        finished, self._finished = self._finished, []
        finished.extend(request for request, _ in self._outputs.take())
        return finished

    def load_model_state_dict(
        self, fetch: Callable[[dict[str, torch.Tensor]], Iterable[str]]
    ) -> None:
        """Fetch the model state dict into the model's weights one FSDP group at a time,
        and reset the KV cache when configured."""
        # TODO: the results of the batch in flight at a load are processed after it, so
        # a request that finishes in that batch is reported at the new policy version
        # though its last token came from the old weights.
        scheduler = self._scheduler
        # On the schedule stream, as SGLang handles these requests: the scheduler's
        # GPU bookkeeping (e.g. the KV pool's free pages) lives there.
        with scheduler.device_module.StreamContext(scheduler.schedule_stream):
            if self._reset_kv_cache_on_model_state_dict_load:
                # Retract running requests to the waiting queue, so they recompute
                # their KV under the new weights; this processes the in-flight
                # batch's results.
                scheduler.pause_generation(PauseGenerationReqInput(mode="retract"))
            # The overlap loop may have a forward in flight, which reads the weights.
            scheduler.schedule_stream.wait_stream(scheduler.forward_stream)
            self._get_model().model_state_dict_loader.load(fetch)
            scheduler.forward_stream.wait_stream(scheduler.schedule_stream)
            if self._reset_kv_cache_on_model_state_dict_load:
                self._finished.extend(request for request, _ in self._outputs.take())
                if not scheduler.flush_cache(empty_cache=False):
                    raise RuntimeError("SGLang did not flush its KV cache")
                scheduler.continue_generation(
                    ContinueGenerationReqInput(torch_empty_cache=False)
                )

    @property
    def holds_session_kv(self) -> bool:
        return True

    def release_sessions(self, session_ids: list[str], group_ids: list[int]) -> None:
        released = set(session_ids)
        for group_id in group_ids:
            released |= self._group_sessions.pop(group_id, set())
        # Sorted, not set order: string hashes differ per process, and every TP rank
        # must release in the same order.
        for session_id in sorted(released):
            self._scheduler.tree_cache.release_radix_session(session_id)

    def shutdown(self) -> None:
        """Nothing to release: the scheduler runs in this process, which the actor
        framework tears down."""

    def _get_model(self):
        """The ``SGLangModelWrapper`` the scheduler runs."""
        return self._scheduler.tp_worker.model_runner.model


class _OutputCollector:
    """Stands in for the scheduler's sockets to the tokenizer and detokenizer processes.

    SGLang sends a request's tokens and logprobs in increments: every
    ``SGLANG_FORCE_STREAM_INTERVAL`` tokens even without streaming, and the rest when it
    finishes. The collector concatenates them and keeps each finished request with its
    abort message (None unless SGLang aborted it), and its metrics inputs from the
    timestamps of SGLang's request object.
    """

    def __init__(self) -> None:
        # Tokens and logprobs received so far, per unfinished request.
        self._partial: dict[str, tuple[list[int], list[float]]] = {}
        self._finished: list[tuple[FinishedRequest, str | None]] = []
        # SGLang's request object, the arrival time and the first wait queue entry
        # time of each admitted request. SGLang restamps its wait queue entry time
        # when it retracts the request.
        self._admitted: dict[str, tuple[Req, float, float]] = {}

    def admit(self, req: Req, arrival_time: float) -> None:
        self._admitted[req.rid] = (
            req,
            arrival_time,
            req.time_stats.wait_queue_entry_time,
        )

    def send_output(self, output: Any, recv_obj: Any = None) -> None:
        if isinstance(output, BatchTokenIDOutput):
            self._receive_tokens(output)
        elif isinstance(output, AbortReq):
            token_ids, token_logprobs = self._partial.pop(output.rid, ([], []))
            self._admitted.pop(output.rid, None)
            reason = output.finished_reason or {}
            self._finished.append(
                (
                    FinishedRequest(
                        request_id=output.rid,
                        token_ids=token_ids,
                        token_logprobs=token_logprobs,
                        finish_reason="abort",
                        metrics_inputs=RequestMetricsInputs(
                            num_cached_tokens=None, has_stats=False
                        ),
                    ),
                    reason.get("message", "aborted by SGLang"),
                )
            )

    def take(self) -> list[tuple[FinishedRequest, str | None]]:
        finished, self._finished = self._finished, []
        return finished

    def _receive_tokens(self, output: BatchTokenIDOutput) -> None:
        offsets = output.output_ids_offsets
        for i, (request_id, reason) in enumerate(
            zip(output.rids, output.finished_reasons, strict=True)
        ):
            token_ids, token_logprobs = self._partial.setdefault(request_id, ([], []))
            token_ids.extend(output.output_ids_flat[offsets[i] : offsets[i + 1]])
            token_logprobs.extend(output.output_token_logprobs_val[i])
            if reason is None:
                continue
            del self._partial[request_id]
            if len(token_logprobs) != len(token_ids):
                raise ValueError(
                    f"{request_id}: {len(token_ids)} tokens but "
                    f"{len(token_logprobs)} logprobs"
                )
            stats = output.request_stats[i] if output.request_stats else None
            num_cached_tokens = None if stats is None else stats.cached_tokens
            if reason["type"] == "abort":
                # SGLang may abort a request while admitting it, before admit().
                self._admitted.pop(request_id, None)
                metrics_inputs = RequestMetricsInputs(
                    num_cached_tokens=num_cached_tokens, has_stats=False
                )
            else:
                req, arrival_time, queued_ts = self._admitted.pop(request_id)
                metrics_inputs = _metrics_inputs(
                    req,
                    arrival_time=arrival_time,
                    queued_ts=queued_ts,
                    num_generation_tokens=len(token_ids),
                    num_cached_tokens=num_cached_tokens,
                )
            finished = FinishedRequest(
                request_id=request_id,
                token_ids=token_ids,
                token_logprobs=token_logprobs,
                finish_reason=reason["type"],
                metrics_inputs=metrics_inputs,
            )
            abort_message = (
                reason.get("message", "aborted by SGLang")
                if reason["type"] == "abort"
                else None
            )
            self._finished.append((finished, abort_message))


def _metrics_inputs(
    req: Req,
    *,
    arrival_time: float,
    queued_ts: float,
    num_generation_tokens: int,
    num_cached_tokens: int | None,
) -> RequestMetricsInputs:
    """A finished request's metrics inputs, from the timestamps SGLang's scheduler
    stamps on its request object (which it exports only with its own metrics on)."""
    # TODO(https://github.com/sgl-project/sglang/issues/48482): reads SGLang's
    # per-request time stats directly; SGLang exports them
    # (RequestOutputStats.time_stats) only with enable_metrics, which also starts its
    # Prometheus collectors.
    time_stats = req.time_stats
    # SGLang stamps time.perf_counter(); arrival_time is rank 0's time.time().
    perf_counter_to_wall_clock = time.time() - time.perf_counter()
    return RequestMetricsInputs(
        num_cached_tokens=num_cached_tokens,
        has_stats=True,
        queued_ts=queued_ts,
        scheduled_ts=time_stats.forward_entry_time,
        first_token_ts=time_stats.prefill_finished_time,
        last_token_ts=time_stats.completion_time,
        first_token_latency=time_stats.prefill_finished_time
        + perf_counter_to_wall_clock
        - arrival_time,
        num_generation_tokens=num_generation_tokens,
        num_preemptions=req.retraction_count,
    )


def _overlap_event_loop_iteration(scheduler: Scheduler) -> None:
    """One iteration of ``Scheduler.event_loop_overlap``, without receiving requests.

    The engine loop admits requests through ``add_requests`` instead.
    """
    # TODO(https://github.com/sgl-project/sglang/issues/48472): SGLang has no public
    # single-iteration API; this mirrors the loop body of the pinned SGLang version and
    # must be kept in sync with it.
    scheduler.increment_scheduler_iterations()

    def pop_and_process() -> None:
        batch, result = scheduler.result_queue.popleft()
        scheduler.process_batch_result(batch, result)

    process_last_batch_early = scheduler.should_process_last_batch_early(
        scheduler.last_batch
    )
    if process_last_batch_early:
        pop_and_process()

    plan = scheduler.get_next_batch_to_run(
        running_batch=scheduler.running_batch, last_batch=scheduler.last_batch
    )
    scheduler.running_batch = plan.running_batch
    batch = plan.batch_to_run
    scheduler.cur_batch_for_debug = batch

    if batch:
        batch_result = scheduler.run_batch(batch)
        scheduler._apply_war_barrier()
        scheduler.result_queue.append((batch.copy(), batch_result))
    else:
        batch_result = None
        scheduler._sched_idled = True

    if scheduler.last_batch:
        if not process_last_batch_early:
            pop_and_process()
    elif batch is None:
        scheduler.on_idle()

    scheduler.launch_batch_sample_if_needed(batch_result, batch)
    scheduler.last_batch = batch
