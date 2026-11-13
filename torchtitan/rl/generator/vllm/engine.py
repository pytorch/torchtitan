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
from dataclasses import dataclass, field
from typing import Any

import torch
from vllm import EngineArgs, LLMEngine, SamplingParams
from vllm.config import AttentionConfig
from vllm.exceptions import VLLMClientError
from vllm.inputs import EngineInput
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.utils.cache import LRUCache

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.config import DebugConfig, PrecisionConfig
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.dist_moe import DistMoeRuntime
from torchtitan.rl.generator.engine import GenerationEngine, ModelStateDictLoadMode
from torchtitan.rl.generator.model import validate_ep_token_dispatchers
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.types import (
    EngineRequest,
    FinishedRequest,
    RequestMetricsInputs,
    SamplingConfig,
)
from torchtitan.rl.generator.vllm.batch_invariance import (
    force_trainer_logprobs_for_batch_invariance,
)
from torchtitan.rl.generator.vllm.config import VLLMCudaGraphConfig
from torchtitan.rl.generator.vllm.multimodal_wrapper import (
    image_prompt_to_engine_input,
    keep_prefix_cached_image_data,
)
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

# Byte budget of `VLLMEngine`'s pixel values cache: ~400 Geometry3K images (2.4 MB each).
_PIXEL_VALUES_LRU_CACHE_BYTES = 1 << 30


def add_extra_vllm_engine_args(
    engine_kwargs: dict[str, Any], extra_engine_args: dict[str, Any]
) -> None:
    """Add ``VLLMEngine.Config.extra_engine_args`` to ``engine_kwargs`` in place.

    Raises ``ValueError`` for keys TorchTitan already set in ``engine_kwargs`` or
    derives from a config field, so they cannot silently diverge from it.
    """
    # Keys TorchTitan derives from a config field: the error names that field. Keys set
    # only when their field is set must be listed, since engine_kwargs may lack them.
    config_field_by_engine_arg = {
        "max_num_batched_tokens": "Generator.Config.max_num_batched_tokens",
        "compilation_config": "VLLMEngine.Config.cuda_graph",
        "seed": "Generator.Config.debug.seed",
        "watermark": "VLLMEngine.Config.watermark",
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
    ``LLMEngine`` is not thread-safe. With an ``vision_processor`` it also serves image
    prompts (M-RoPE models such as Qwen3.5; see ``vllm/multimodal_wrapper.py``).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GenerationEngine.Config):
        cuda_graph: VLLMCudaGraphConfig = field(default_factory=VLLMCudaGraphConfig)
        """CUDA graph capture settings for the vLLM engine."""

        watermark: float = 0.1
        """vLLM's ``EngineArgs.watermark``: the fraction of KV cache blocks that must stay
        free after admitting a waiting or preempted request. vLLM's default is 0.0.

        0.1 admits only while KV usage stays <= 90%, so running requests have 10% of the
        blocks to grow into before vLLM preempts one. Raise it when
        ``generator/num_preemptions`` is high.
        Lower it when KV usage sits near ``1 - watermark`` while requests wait.

        Example, 10,000 blocks, 500 free, a waiting request needs 100 (400 free after):

            watermark=0.0    admitted
            watermark=0.03   admitted (400 >= 300)
            watermark=0.1    waits (400 < 1,000)
            no request running: admitted at any watermark
        """

        extra_engine_args: dict[str, Any] = field(default_factory=dict)
        """Additional keyword arguments passed to vLLM's ``EngineArgs``, for engine
        options without a dedicated field (e.g. ``{"long_prefill_token_threshold": 4096}``).
        vLLM validates them. Keys that TorchTitan sets itself, or derives from a config
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
        precision: PrecisionConfig,
        gpu_memory_limit: float,
        max_num_batched_tokens: int | None,
        checkpointer: CheckpointManager.Config | None,
        dist_moe_runtime: DistMoeRuntime.Config | None,
        debug: DebugConfig,
        reset_kv_cache_on_model_state_dict_load: bool,
        model_state_dict_load_mode: ModelStateDictLoadMode,
        vision_processor: VisionProcessor.Config | None,
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
        validate_ep_token_dispatchers(
            model,
            parallelism=parallelism,
            cuda_graph=config.cuda_graph.mode != "NONE",
            deterministic=debug.deterministic,
        )
        self.config = config
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
            vision_processor=vision_processor,
        )
        self._vision_processor: VisionProcessor | None = (
            vision_processor.build() if vision_processor is not None else None
        )
        # Pixel values by image hash: a group's samples and later turns resend the same
        # images, and preprocessing one takes ~6.5 ms on the engine loop thread.
        self._pixel_values_lru_cache: LRUCache[str, torch.Tensor] = LRUCache(
            _PIXEL_VALUES_LRU_CACHE_BYTES,
            getsizeof=lambda pixel_values: pixel_values.nbytes,
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
            serves_images=vision_processor is not None,
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
        if vision_processor is not None:
            keep_prefix_cached_image_data()
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
        vision_processor: VisionProcessor.Config | None,
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
            vision_processor=vision_processor,
        )
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        if batch_invariant:
            # The vLLM v2 logprob Triton kernel bypasses the batch-invariant aten
            # overrides; route it through trainer's function to match the trainer exactly.
            force_trainer_logprobs_for_batch_invariance()

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
        serves_images: bool,
    ) -> EngineArgs:
        """Build the vLLM ``EngineArgs`` (see ``GenerationEngine`` for the arguments;
        ``serves_images``: the engine has a vision processor)."""
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
        engine_kwargs["watermark"] = self.config.watermark
        # Smaller `priority` runs first and is preempted last; we pass `EngineRequest.group_id`
        # (-1 for all validation), so older groups have higher priority (see its docstring).
        # TODO: validation requests (priority -1) are admitted before training requests, also while a
        # validation runs beside training. Measure the step-time cost of a validation; consider a priority
        # between training groups.
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
        if serves_images:
            # Image prompts bypass vLLM's processor, so its cache would only hold copies.
            engine_kwargs["mm_processor_cache_gb"] = 0
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
        return self._get_model().model_state_dict_loader.prefetched_model_state_dict

    def add_requests(
        self, requests: list[EngineRequest]
    ) -> list[tuple[str, Exception]]:
        """Admit ``requests`` in order and return the ``(request_id, error)`` of each request
        vLLM rejected."""
        rejected_requests: list[tuple[str, Exception]] = []
        for request in requests:
            # Rejected, not raised: the request's `generate` call fails and the engine keeps
            # serving. Both checks run before LLMEngine.add_request registers the request,
            # so a rejection leaves no engine state:
            #   - `_preprocess_images`: no vision processor, or an image file does not decode;
            #   - vLLM's validation (prompt length, sampling params, token ids): render_cmpl
            #     and InputProcessor.process_inputs raise VLLMClientError (not a ValueError
            #     subclass) or ValueError (recheck on a vLLM upgrade).
            # Both are deterministic and rank 0 broadcasts the same requests to every TP
            # rank, so all ranks reject the same ones; otherwise their collectives hang.
            try:
                # vLLM appends a random suffix to the id; the scheduler knows only that one.
                internal_request_id = self._engine.add_request(
                    request_id=request.request_id,
                    prompt=self._prompt_to_engine_input(request),
                    params=_build_sampling_params(request.sampling),
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

    def _prompt_to_engine_input(self, request: EngineRequest) -> EngineInput:
        """vLLM's engine input for ``request``: its token ids, plus the pixel values of an
        image prompt."""
        # Salt by the pinned version so a request only reuses KV computed under that version.
        cache_salt = (
            None
            if self._reset_kv_cache_on_model_state_dict_load
            else str(request.min_policy_version)
        )
        if request.prompt.images:
            return image_prompt_to_engine_input(
                request.prompt, self._preprocess_images(request), cache_salt=cache_salt
            )
        prompt = {"prompt_token_ids": request.prompt.token_ids}
        if cache_salt is not None:
            prompt["cache_salt"] = cache_salt
        # render_cmpl is vLLM's input pipeline (tokenize is a no-op for tokenized prompts);
        # the high-level entry stays resilient to vLLM internals vs vllm.inputs.tokens_input.
        [engine_input] = self._engine.renderer.render_cmpl([prompt])
        return engine_input

    def _preprocess_images(self, request: EngineRequest) -> list[torch.Tensor]:
        """Each of ``request``'s images as ``(num_patches, patch_dim)`` pixel values
        (`VisionProcessor.resize_and_normalize`, then `patchify`), cached by image hash.

        Raises:
            ValueError: No vision processor, or an image file does not decode;
                `add_requests` rejects the request.
        """
        if self._vision_processor is None:
            raise ValueError(
                f"{request.request_id} has images; set Controller.Config.vision_processor "
                "to serve image prompts."
            )
        pixel_values = []
        for image in request.prompt.images:
            image_pixel_values = self._pixel_values_lru_cache.get(image.hash_value)
            if image_pixel_values is None:
                try:
                    pixels, _ = self._vision_processor.resize_and_normalize(image.data)
                except RuntimeError as error:
                    # torchvision raises RuntimeError for a truncated or non-image file, which
                    # gets past rendering (it reads only the header): fail just its rollout.
                    raise ValueError(
                        f"{request.request_id}: image file {image.source_path} does not "
                        f"decode: {error}"
                    ) from error
                image_pixel_values, _ = self._vision_processor.patchify(pixels)
                self._pixel_values_lru_cache.put(image.hash_value, image_pixel_values)
            pixel_values.append(image_pixel_values)
        return pixel_values

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

    def load_model_state_dict(
        self, fetch: Callable[[dict[str, torch.Tensor]], Iterable[str]]
    ) -> None:
        """Fetch the model state dict into the model's weights one FSDP group at a time, and
        reset the KV cache (and, for image prompts, vLLM's encoder cache) when configured."""
        self._get_model().model_state_dict_loader.load(fetch)
        if self._reset_kv_cache_on_model_state_dict_load:
            # Always reset running requests too: the only reason to reset is a strict
            # recompute under the new weights. Keeping running requests' KV while hiding
            # old KV from new requests is already what the default (no reset) does via the salt.
            self._engine.reset_prefix_cache(
                reset_running_requests=True,
            )
            if self._vision_processor is not None:
                # Unsalted image hashes carry no policy version: drop the old vision weights'
                # encoder outputs too. vLLM's warning here about queued requests is about its
                # processor caches, which image prompts turn off.
                self._engine.reset_encoder_cache()

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
