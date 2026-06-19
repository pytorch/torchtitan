# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TLDR:
_data_input_loop -> _rollout_loop -> _episode_batcher_loop -> packed_training_batch_queue -> _trainer_loop
           v               v                    v                         ^
           |               |                    |                         |
           |               ^                    ^                         |
           +----------------------AsyncRolloutBuffer----------------------+


Detailed diagram:

_data_input_loop                                      _rollout_loop[N] (rollout workers)
+--------------------------------------------------+  +--------------------------------------------------+
| buffer.wait_for_rollout_group_slot()             |  | work = buffer.claim_next_rollout_group()        |
| sample = rollouter.get_training_sample()         |  | group = rollouter.run_group_rollouts(work.sample)|
| work = RolloutGroupWork(group_id, sample)        |  | buffer.record_rollout_group_result(group)       |
| buffer.add_rollout_group_work(work)              |  +-----------------------+--------------------------+
+-----------------------+--------------------------+                          ^ |
                        |                                                     | |
                        | adds work entry                                     | | updates same entry
                        v                                                     | v
AsyncRolloutBuffer
+-----------------------+-----------------------------------------------------+-+--------------------------+
| capacity = ceil(max_offpolicy_steps * num_rollout_groups_per_train_step)                               |
|                                                                                                        |
| caller                  buffer call                                      RolloutGroupWork Status        |
| _data_input_loop        add_rollout_group_work(RolloutGroupWork)         WAITING                        |
| _rollout_loop[N]        claim_next_rollout_group()                       WAITING -> GENERATING          |
| _rollout_loop[N]        record_rollout_group_result(RolloutGroup)        GENERATING -> GENERATED        |
| _episode_batcher_loop   RolloutGroup = take_generated_rollout_group()    GENERATED -> removed           |
+--------------------------------------------------------------------------------+----------------------+
                                                                                 |
                                                                                 | group = buffer.take_generated_rollout_group()
                                                                                 v
_episode_batcher_loop
+--------------------------------------------------------------------------------+
| output = episode_builder.build_episodes(rollout_group=group)                   |
| maybe_training_batches = episode_batcher.add_episode_group(output)             |
| packed_training_batch_queue.put(PackedTrainingBatch)                           |
+-----------------------+------------------------------+-------------------------+
                        |                              |
                        | add output                   | put batch
                        v                              v
EpisodeBatcher                                     packed_training_batch_queue
+----------------------------------------------+   +----------------------------------------------+
| accumulated built episode groups             |   | size 1; holds PackedTrainingBatch | None     |
| stale-filter at flush; pack if target met    |   +---------------------+------------------------+
+----------------------------------------------+                         |
                                                                         | packed = packed_training_batch_queue.get()
                                                                         v
_trainer_loop
+--------------------------------------------------------------------------------+
| train packed batch -> optim step -> push weights -> pull weights               |
+--------------------------------------------------------------------------------+

Backpressure:
_data_input_loop
  waits_for: AsyncRolloutBuffer slot
  unblocked_by: _episode_batcher_loop -> buffer.take_generated_rollout_group()

_rollout_loop[N]
  waits_for: claimable RolloutGroupWork
  unblocked_by: _data_input_loop -> buffer.add_rollout_group_work()

_episode_batcher_loop
  waits_for: takeable RolloutGroup
  unblocked_by: _rollout_loop[N] -> buffer.record_rollout_group_result()

_episode_batcher_loop
  waits_for: packed_training_batch_queue slot
  unblocked_by: _trainer_loop -> packed_training_batch_queue.get()

_trainer_loop
  waits_for: PackedTrainingBatch
  unblocked_by: _episode_batcher_loop -> packed_training_batch_queue.put()
"""

import asyncio
import contextlib
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

# must run before torch import
# TODO: this should be defined earlier, in the .sh script
# since __init__ can import torch before this is set.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: F401  # force torch import after PYTORCH_CUDA_ALLOC_CONF is set
import torchstore as ts
from monarch.actor import ProcMesh
from monarch.spmd import setup_torch_elastic_env_async

from torchtitan.config import CompileConfig, Configurable
from torchtitan.experiments.rl.actors.generator import SamplingConfig, VLLMGenerator
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.batcher import EpisodeBatcher, PackedTrainingBatch
from torchtitan.experiments.rl.controller.async_rollout_buffer import (
    AsyncRolloutBuffer,
    RolloutGroupWork,
)
from torchtitan.experiments.rl.controller.episode_builder import EpisodeBuilder
from torchtitan.experiments.rl.generator_router import GeneratorRouter, RoutingContext
from torchtitan.experiments.rl.losses import GRPOLoss
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.rollout import prepare_rollout_metrics, RolloutGroup
from torchtitan.experiments.rl.rollout.rollouter import Rollouter
from torchtitan.experiments.rl.rollout.types import GenerateFn
from torchtitan.experiments.rl.rollout_recorder import RolloutSampleRecorder
from torchtitan.experiments.rl.types import Completion
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols.model_spec import ModelSpec

logger = logging.getLogger(__name__)


def _rollout_turn_metrics(groups: list[RolloutGroup]) -> list[m.Metric]:
    """Flatten every turn's per-generation metrics (latencies, output tokens) across the groups.

    Shared by the rollout loop (`_rollout_loop`, one group) and validation
    (`_collect_validation_rollouts`, a round of groups).
    """
    return [
        metric
        for group in groups
        for rollout in group.rollouts
        for rollout_turn in rollout.turns
        for metric in rollout_turn.metrics
    ]


def combine_microbatch_metrics(
    microbatch_metrics: list[dict[str, float]],
) -> dict[str, float]:
    """Combine per-microbatch loss metrics over the grad-accumulation: mean/frac keys are pre-normalized
    by num_global_valid_tokens so summing them reconstructs the global value; max keys take the max.

    Matches `_mean`/`_frac` as well as `/mean`/`/frac` so the GRPO loss keys (`loss/ratio_mean`,
    `loss/ratio_clipped_frac`, `loss/generator_logprob_nan_frac`) are summed too — they are all
    pre-normalized, so without this they would report ~1/grad_accum of the truth.

    Example:

        combine_microbatch_metrics([{"loss/ratio_clipped_frac": 0.1, "x/max": 2.0},
                                    {"loss/ratio_clipped_frac": 0.2, "x/max": 5.0}])
        # -> {"loss/ratio_clipped_frac": 0.3, "x/max": 5.0}
    """
    combined: dict[str, float] = {}
    for microbatch in microbatch_metrics:
        for key, value in microbatch.items():
            if key not in combined:
                combined[key] = value
            elif key.endswith("/max"):
                combined[key] = max(combined[key], value)
            elif key.endswith(("/mean", "_mean", "/frac", "_frac")):
                combined[key] += value
    return combined


class StepMetricsRecorder:
    """Collect one train step's timing metrics.

    Example:
        step_metrics = StepMetricsRecorder()
        with step_metrics.record("timing/step/train"):
            ...
        metrics = step_metrics.metrics()   # [Metric("timing/step/train", Mean(...)), ...]
    """

    def __init__(self) -> None:
        self.durations: dict[str, float] = {}

    @contextlib.contextmanager
    def record(self, key: str):
        if key in self.durations:
            raise ValueError(f"duplicate timing key {key!r} in one step")
        start = time.perf_counter()
        try:
            yield
        finally:
            self.durations[key] = time.perf_counter() - start

    def metrics(self) -> list[m.Metric]:
        return [m.Metric(key, m.Mean(value)) for key, value in self.durations.items()]


def _perf_ratio_metrics(
    *, num_global_valid_tokens: int, durations: dict[str, float]
) -> list[m.Metric]:
    """Trainer-side goodput panel derived from one step's timings.

    The gap between goodput (`/ step total`) and active (`/ train`) IS the trainer-idle + sync bubble.

    Example:
        # step total=10, wait=8, train=1, weight_sync=0.5, num_global_valid_tokens=100
        # -> perf/trainer_idle_ratio=0.8, perf/weight_sync_overhead_ratio=0.05,
        #    perf/goodput_tokens_per_second=10, perf/active_tokens_per_second=100
    """
    step_s = sum(durations.values())
    wait_s = durations.get("timing/step/wait_for_training_batch", 0.0)
    train_s = durations.get("timing/step/train", 0.0)
    sync_s = durations.get("timing/step/weight_sync/push", 0.0) + durations.get(
        "timing/step/weight_sync/pull", 0.0
    )
    ratios = {
        "perf/trainer_idle_ratio": wait_s / step_s if step_s else 0.0,
        "perf/weight_sync_overhead_ratio": sync_s / step_s if step_s else 0.0,
        "perf/goodput_tokens_per_second": (
            num_global_valid_tokens / step_s if step_s else 0.0
        ),
        "perf/active_tokens_per_second": (
            num_global_valid_tokens / train_s if train_s else 0.0
        ),
    }
    return [m.Metric(key, m.NoReduce(value)) for key, value in ratios.items()]


@dataclass(kw_only=True, slots=True)
class ValidationConfig:
    """Held-out validation cadence + size."""

    num_samples: int = 20
    """Held-out prompts scored greedily (temp=0, n=1) per validation pass. 0 skips validation."""

    every_n_steps: int = 0
    """Steps between mid-training validation passes. 0 = pre/post only (periodic is a TODO)."""


@dataclass(kw_only=True, slots=True)
class AsyncControlConfig(Configurable.Config):
    """Public async-loop knobs; tensor shapes and filters live on the nested component configs.

    Example:
        # 8 prompt groups/step x 8 siblings = 64 rollouts/step; up to 24 groups buffered (3 steps ahead).
        AsyncControlConfig(num_training_steps=100, group_size=8,
                           num_rollout_groups_per_train_step=8, max_offpolicy_steps=3)
    """

    num_training_steps: int = 10
    """Optimizer steps to run."""

    group_size: int = 8
    """Sibling rollouts sampled per prompt (the GRPO group)."""

    num_rollout_groups_per_train_step: int = 8
    """Prompt groups whose surviving rollouts close one train step. Rollout target = this * group_size."""

    max_offpolicy_steps: int | None = 3
    """Policy-version lag budget; also derives buffer capacity. None logs lag without dropping."""

    num_rollout_workers: int | None = None
    """Rollout worker coroutines. None uses the derived buffer capacity."""

    buffer: AsyncRolloutBuffer.Config = field(default_factory=AsyncRolloutBuffer.Config)
    episode_builder: EpisodeBuilder.Config = field(
        default_factory=EpisodeBuilder.Config
    )
    episode_batcher: EpisodeBatcher.Config = field(
        default_factory=EpisodeBatcher.Config
    )
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    def __post_init__(self):
        if self.max_offpolicy_steps == 0:
            raise ValueError(
                "max_offpolicy_steps=0 (true on-policy) is not supported yet: the generator keeps sampling at "
                "the current version during the optimizer step, so those episodes become lag-1 the instant the "
                "trainer advances. Use >=1, or None to disable dropping."
            )
            # TODO(async-rl): support max_offpolicy_steps=0 via a strict on-policy mode — pause generation during
            # the step, or switch to a trainer-pull / rendezvous queue so the batch is filtered exactly when the
            # trainer asks.


class RLTrainer(Configurable):
    """Top-level RL training orchestrator.

    Owns a `PolicyTrainer` actor (gradient updates), a `VLLMGenerator` actor
    (sampling), and a `Rollouter` (datasets + rubric + env construction). Each
    training step samples groups of rollouts, scores them via the rollouter's rubric,
    builds GRPO advantages, and syncs trainer weights to the generator.

    Example:

        cfg = config_registry.rl_grpo_qwen3_0_6b_varlen()
        trainer = cfg.build()
        await trainer.setup_async()
        await trainer.run()
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Top-level config for RL training."""

        model_spec: ModelSpec | None = None
        """Model specification shared by trainer and generator.
        Set programmatically via config_registry (not from CLI)."""

        hf_assets_path: str = "./tests/assets/tokenizer"
        """Path to HF assets folder (model weights, tokenizer, config files)."""

        dump_folder: str = "outputs/rl"
        """Root output folder for RL artifacts (temp weights, logs, etc.)."""

        async_control: AsyncControlConfig = field(default_factory=AsyncControlConfig)
        """Async-loop knobs: group/step counts, staleness, workers, and the buffer/builder/batcher/validation configs."""

        rollouter: Rollouter.Config
        """The rollouter: its datasets, envs, and rubric."""
        # TODO: support multiple rollouters for data mixing.

        renderer: RendererConfig
        """Message-to-token renderer config."""

        rollout_recorder: RolloutSampleRecorder.Config = field(
            default_factory=RolloutSampleRecorder.Config
        )
        """JSONL recorder to save sampled rollouts to disk for further inspection and debugging."""

        compile: CompileConfig = field(default_factory=CompileConfig)
        """torch.compile config shared by trainer and generator."""

        trainer: PolicyTrainer.Config = field(
            default_factory=lambda: PolicyTrainer.Config(loss=GRPOLoss.Config())
        )
        """PolicyTrainer config. Controls optimizer, training, parallelism."""

        generator: VLLMGenerator.Config = field(default_factory=VLLMGenerator.Config)
        """VLLMGenerator actor configuration (vLLM engine, sampling)."""

        num_generators: int = 1
        """Number of generator replicas to spawn as separate proc meshes.

        This is distinct from intra-generator parallelism controlled by
        ``generator.parallelism``. Total generator GPU/process usage is
        ``num_generators * generator_world_size``.
        """

        generator_router: GeneratorRouter.Config = field(
            default_factory=GeneratorRouter.Config
        )
        """Generator routing strategy configuration."""

        metrics: m.MetricsProcessor.Config = field(
            default_factory=m.MetricsProcessor.Config
        )

        def __post_init__(self):
            if self.num_generators < 1:
                raise ValueError(
                    f"num_generators must be at least 1, got {self.num_generators}"
                )
            if self.generator.checkpoint.enable:
                raise ValueError(
                    "Generator checkpoint must be disabled in the RL loop "
                    "(weights are synced from the trainer via TorchStore). "
                    "Set generator.checkpoint.enable=False."
                )
            # RL policy inputs are shaped by BatchConfig, not TrainingConfig.
            if self.trainer.parallelism.enable_sequence_parallel:
                sp_degree = self.trainer.parallelism.tensor_parallel_degree
                seq_len = self.async_control.episode_batcher.batch.seq_len
                if sp_degree > 1 and seq_len % sp_degree != 0:
                    raise ValueError(
                        f"RL batcher sequence length ({seq_len}) must be divisible "
                        f"by sequence parallel degree ({sp_degree})."
                    )
            # Static (model-free) seq-len guard; setup_async runs the authoritative one with model.max_seq_len.
            seq_len = self.async_control.episode_batcher.batch.seq_len
            rollout_cap = getattr(
                getattr(self.rollouter, "token_env", None),
                "rollout_max_context_len",
                None,
            )
            if (
                rollout_cap is not None
                and rollout_cap + self.generator.sampling.max_tokens > seq_len
            ):
                raise ValueError(
                    f"rollout_max_context_len ({rollout_cap}) + sampling.max_tokens "
                    f"({self.generator.sampling.max_tokens}) exceeds episode_batcher.batch.seq_len ({seq_len}); "
                    f"episodes longer than seq_len are dropped in packing. Raise seq_len or lower the caps."
                )
            if self.trainer.debug.batch_invariant:
                if not self.trainer.debug.deterministic:
                    raise ValueError("batch_invariant requires deterministic=True")
                # TODO: Replace trainer dtype constraint to use mixed
                #  training enabled by FSDP.
                if self.trainer.training.dtype != "bfloat16":
                    raise ValueError(
                        f"batch_invariant requires bfloat16 training dtype, "
                        f"got {self.trainer.training.dtype!r}"
                    )
                if self.generator.model_dtype != "bfloat16":
                    raise ValueError(
                        f"batch_invariant requires bfloat16 generator dtype, "
                        f"got {self.generator.model_dtype!r}"
                    )
                if self.trainer.parallelism.enable_sequence_parallel:
                    raise ValueError(
                        "batch_invariant mode doesn't support SP now. "
                        "SP uses reduce-scatter which only supports Ring in NCCL "
                        "and has not been validated for determinism."
                    )

            if (
                not self.generator_router.hot_swap
                and not self.generator.reset_prefix_cache_on_weight_sync
            ):
                raise ValueError(
                    "generator_router.hot_swap=False requires "
                    "generator.reset_prefix_cache_on_weight_sync=True, else requests admitted after a "
                    "pull reuse KV cached under the old weights."
                )

    def __init__(self, config: Config):
        self.config = config
        self.trainer: PolicyTrainer | None = None
        self.generator_router: GeneratorRouter | None = None
        self._proc_meshes = []
        self.metrics_processor: m.MetricsProcessor = config.metrics.build(
            log_dir=config.dump_folder,
            job_config=config.to_dict(),
        )
        self.renderer = config.renderer.build(tokenizer_path=config.hf_assets_path)

        # Carry the base seed and renderer stop tokens on the sampling config so
        # the generator reads them off each request; the rollouter offsets the
        # seed per sample. Avoids the generator depending on request_id format.
        self._sampling = replace(
            config.generator.sampling,
            seed=config.generator.debug.seed,
            stop_token_ids=list(self.renderer.get_stop_token_ids()),
        )
        # TODO: pass our own tokenizer to the renderer and read pad/eos off it
        # once `renderers` supports bring-your-own-tokenizer
        # (https://github.com/PrimeIntellect-ai/renderers/pull/70).
        # Until then, reach into the renderer's tokenizer for the pad id (eos doubles as pad).
        self._pad_id = self.renderer._tokenizer.eos_token_id
        self._rollouter: Rollouter = config.rollouter.build()
        self.rollout_recorder = config.rollout_recorder.build(
            dump_dir=config.dump_folder
        )

        # Derived async-loop sizing (one public knob, max_offpolicy_steps, sets the run-ahead bound).
        async_control = config.async_control
        self._target_rollouts_per_training_batch = (
            async_control.num_rollout_groups_per_train_step * async_control.group_size
        )
        buffer_depth_train_steps = (
            async_control.max_offpolicy_steps
            if async_control.max_offpolicy_steps is not None
            else 2
        )
        self._max_buffered_rollout_groups = max(
            1,
            math.ceil(
                buffer_depth_train_steps
                * async_control.num_rollout_groups_per_train_step
            ),
        )
        self._num_rollout_workers = (
            async_control.num_rollout_workers or self._max_buffered_rollout_groups
        )

    async def close(self):
        """Best-effort: tear down actors, close metric backends, then stop proc meshes."""
        logger.info("Closing: tearing down actors and process meshes.")

        if self.trainer is not None:
            try:
                await self.trainer.close.call()
            except Exception:
                logger.exception("trainer.close failed")

        if self.generator_router is not None:
            close_results = await self.generator_router.fanout(
                "close", return_exceptions=True
            )
            for idx, result in enumerate(close_results):
                if isinstance(result, BaseException):
                    actor_name = (
                        "generator" if len(close_results) == 1 else f"generator[{idx}]"
                    )
                    logger.error(
                        "%s.close failed",
                        actor_name,
                        exc_info=(type(result), result, result.__traceback__),
                    )

        try:
            self.metrics_processor.close()
        except Exception:
            logger.exception("metrics_processor close failed")

        for i, mesh in enumerate(self._proc_meshes):
            try:
                await mesh.stop()
            except Exception:
                logger.exception("mesh.stop[%d] failed", i)
        self._proc_meshes = []

    def _get_rank_0_value(self, result):
        """Extract rank 0 result from a Monarch ValueMesh.

        Monarch actor endpoints return results from all ranks in the mesh.
        This method picks out rank 0's result. This should be used in cases
        where all ranks return the same result.
        """
        return result.get(0)

    def _make_generate_fn(self, metrics_prefix: str) -> GenerateFn:
        """Build the rollouter's `GenerateFn`: route a completion via the generator router, namespacing
        metrics with `metrics_prefix` and pinning sticky routing on `routing_session_id` (a sample's
        turns reuse one generator's prefix KV)."""

        @sl.log_trace_span("generate")
        async def generate(
            prompt_token_ids: list[int],
            *,
            request_id: str,
            routing_session_id: str | None = None,
            sampling_config: SamplingConfig | None = None,
        ) -> Completion | None:
            result = await self.generator_router.route(
                "generate",
                prompt_token_ids,
                request_id=request_id,
                sampling_config=sampling_config,
                metrics_prefix=metrics_prefix,
                routing_ctx=RoutingContext(
                    estimated_cost=len(prompt_token_ids),
                    session_id=routing_session_id,
                ),
            )
            return self._get_rank_0_value(result)

        return generate

    @sl.log_trace_span("setup_async")
    async def setup_async(
        self,
        *,
        trainer_mesh: ProcMesh,
        generator_meshes: list[ProcMesh],
    ):
        """Spawn Monarch actors on separate meshes and initialize weights.

        Kept separate from ``__init__`` because actor spawning, torch
        elastic env setup, TorchStore initialization, and the initial
        weight push/pull are all ``await``-based runtime side effects
        that cannot run in a synchronous constructor.

        The trainer and generator meshes are provisioned by the caller
        (see ``create_proc_mesh``) on disjoint GPUs; this method only
        spawns the actors on them and synchronizes initial weights from
        trainer to generator. Must be called before :meth:`run`.

        Args:
            trainer_mesh: ProcMesh the trainer actor is spawned on.
            generator_meshes: ProcMesh objects the generator actors are spawned on.
        """
        # Thread pool for TokenEnv's asyncio.to_thread renderer calls — one worker per
        # concurrent rollout, capped by CPUs.
        max_concurrent_rollouts = max(
            self._num_rollout_workers * self.config.async_control.group_size,
            self.config.async_control.validation.num_samples,
        )
        max_workers = max(1, min(max_concurrent_rollouts, os.cpu_count() or 1))
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=max_workers)
        )

        config = self.config
        if not generator_meshes:
            raise ValueError("setup_async requires at least one generator mesh")

        # An episode is at most context + completion tokens; seq_len must fit the longest one or it is dropped
        # in packing and crashes the trainer. With a rollout cap, the longest is cap + sampling.max_tokens;
        # uncapped, it is the model's max_seq_len. (The model-free half of this runs in Config.__post_init__.)
        seq_len = config.async_control.episode_batcher.batch.seq_len
        model_max_seq_len = config.model_spec.model.max_seq_len
        rollout_cap = getattr(
            getattr(config.rollouter, "token_env", None),
            "rollout_max_context_len",
            None,
        )
        max_episode_tokens = (
            rollout_cap + config.generator.sampling.max_tokens
            if rollout_cap is not None
            else model_max_seq_len
        )
        if max_episode_tokens > seq_len:
            raise ValueError(
                f"episode_batcher.batch.seq_len ({seq_len}) is smaller than the longest episode a rollout can "
                f"produce ({max_episode_tokens} tokens; rollout cap={rollout_cap}, "
                f"sampling.max_tokens={config.generator.sampling.max_tokens}, model max_seq_len={model_max_seq_len}). "
                f"Raise seq_len, or lower rollout_max_context_len / sampling.max_tokens."
            )

        trainer_parallelism = config.trainer.parallelism
        dp_shard = max(trainer_parallelism.data_parallel_shard_degree, 1)
        self.trainer_dp_degree = (
            trainer_parallelism.data_parallel_replicate_degree * dp_shard
        )

        # TODO(observability): the mesh_spawn span wraps ~80 LoC of branching
        # provisioner logic. Pull a PerHostProvisioner.spawn_meshes(...) helper and
        # shrink this span to a single call.
        with sl.log_trace_span("mesh_spawn"):
            # Store proc meshes for cleanup
            self._proc_meshes = [trainer_mesh, *generator_meshes]

            await setup_torch_elastic_env_async(trainer_mesh)
            for generator_mesh in generator_meshes:
                await setup_torch_elastic_env_async(generator_mesh)

            # Spawn actors on their respective meshes
            self.trainer = trainer_mesh.spawn(
                "trainer",
                PolicyTrainer,
                config.trainer,
                model_spec=config.model_spec,
                hf_assets_path=config.hf_assets_path,
                generator_dtype=config.generator.model_dtype,
                compile_config=config.compile,
                output_dir=config.dump_folder,
            )

            # The generator runs vLLM's own cudagraph/inductor path; its per-layer torch.compile must
            # stay aot_eager (inductor here crashes the engine). So the generator is pinned to
            # aot_eager and `config.compile.backend` controls ONLY the trainer (which wants inductor).
            # TODO: verify trainer model compile=True works in RL (it shares the model path with the generator).
            generator_compile = replace(config.compile, backend="aot_eager")
            generators = []
            for idx, generator_mesh in enumerate(generator_meshes):
                actor_name = (
                    "generator" if len(generator_meshes) == 1 else f"generator_{idx}"
                )
                generator = generator_mesh.spawn(
                    actor_name,
                    VLLMGenerator,
                    config.generator,
                    model_spec=config.model_spec,
                    model_path=config.hf_assets_path,
                    compile_config=generator_compile,
                    # TODO(async-rl): derive max_num_seqs from the run-ahead depth + the generator's profiled
                    # KV concurrency, and keep the cudagraph capture-size cap separate from this request cap.
                    # 512 for now (cudagraph capture must not be unbounded on this path).
                    max_num_seqs=512,
                    output_dir=config.dump_folder,
                )
                generators.append(generator)
            self.generator_router = config.generator_router.build(generators=generators)

        # Initialize TorchStore for weight sync between trainer and generator.
        # StorageVolumes are spawned on the trainer mesh so they are colocated
        # with the weight source for faster data access in the non-RDMA path.
        # LocalRankStrategy: routes each process to a storage volume based on
        #   LOCAL_RANK, so colocated processes share the same volume.
        # https://github.com/meta-pytorch/torchstore
        with sl.log_trace_span("torchstore_init"):
            await ts.initialize(mesh=trainer_mesh, strategy=ts.LocalRankStrategy())

        # TODO: can we avoid this and just have both trainer and generator initialize from the same checkpoint
        # in parallel?
        # Initial weight sync from trainer to generator
        with sl.log_trace_span("trainer_push_model_state_dict"):
            await self.trainer.push_model_state_dict.call(version=0)
        with sl.log_trace_span("generator_initial_weight_sync"):
            await self.generator_router.pull_model_state_dict(policy_version=0)

    @sl.log_trace_span("_collect_validation_rollouts")
    async def _collect_validation_rollouts(
        self, *, num_groups: int, sampling: SamplingConfig, step: int
    ) -> tuple[list[RolloutGroup], list[m.Metric]]:
        """Sample held-out prompts, run each greedily (n=1) concurrently, and emit validation metrics.

        Separate from the training batcher path, so a validation pass never mixes into off-policy training
        state. `group_id`s use a `val/` namespace so they never collide with a train request still in flight.
        """
        generate = self._make_generate_fn("validation_generator")
        samples = [self._rollouter.get_validation_sample() for _ in range(num_groups)]
        group_results = await asyncio.gather(
            *(
                self._rollouter.run_group_rollouts(
                    generate_fn=generate,
                    sample=sample,
                    group_id=f"val/step={step}/group={i}",
                    group_size=1,
                    sampling=sampling,
                    renderer=self.renderer,
                )
                for i, sample in enumerate(samples)
            ),
            return_exceptions=True,
        )

        rollout_groups: list[RolloutGroup] = []
        num_failed_groups = 0
        for i, result in enumerate(group_results):
            if isinstance(result, BaseException):
                logger.error(
                    "validation group val/step=%d/group=%d failed; dropping",
                    step,
                    i,
                    exc_info=result,
                )
                num_failed_groups += 1
                continue
            rollout_groups.append(result)

        metrics = _rollout_turn_metrics(rollout_groups)
        metrics.append(
            m.Metric("validation/group_failures", m.Sum(float(num_failed_groups)))
        )
        metrics += prepare_rollout_metrics(
            "validation",
            [rollout for group in rollout_groups for rollout in group.rollouts],
        )
        return rollout_groups, metrics

    # TODO: we currently determine validation.num_samples
    # but what if i want to run the entire dataset?
    @sl.log_trace_span("validate")
    async def validate(self, *, step: int) -> list[m.Metric]:
        """Run greedy validation on held-out prompts.

        Args:
            step: Training step this validation pass belongs to (0 for the
                pre-training pass); tagged into logged rollout samples.

        Returns:
            Validation rollout metrics, generation metrics, and validation
            timing.
        """
        # TODO: investigate using pass@k for validation.
        t_validate_start = time.perf_counter()
        num_samples = self.config.async_control.validation.num_samples
        if num_samples == 0:  # skip validation (e.g. loss guard CI)
            return []
        greedy = replace(self._sampling, temperature=0.0, top_p=1.0)

        rollout_groups, validation_metrics = await self._collect_validation_rollouts(
            num_groups=num_samples, sampling=greedy, step=step
        )
        rollouts = [rollout for group in rollout_groups for rollout in group.rollouts]

        self.rollout_recorder.record(
            step=step, is_validation=True, rollout_groups=rollout_groups
        )

        validation_metrics.append(
            m.Metric("validation/num_samples", m.NoReduce(float(len(rollouts))))
        )

        t_validate_s = time.perf_counter() - t_validate_start
        validation_metrics.append(m.Metric("timing/validate", m.NoReduce(t_validate_s)))
        return validation_metrics

    async def run(self) -> None:
        """Orchestrate the async loop: `_trainer_loop` drives; when it finishes or any stage crashes,
        cancel and close the rest."""
        num_training_steps = self.config.async_control.num_training_steps
        logger.info(
            f"Pre-training validation; then {num_training_steps} steps of async RL training"
        )

        pre_validation = await self._validate_and_log(step=0)
        sl.log_trace_instant("training_start")

        # Two policy version pointers: the trainer advances at the optimizer step;
        # the generator version advances when a weight pull completes.
        self._trainer_policy_version = 0
        self._generator_policy_version = 0

        self._rollout_buffer = self.config.async_control.buffer.build(
            max_buffered_rollout_groups=self._max_buffered_rollout_groups,
        )
        episode_builder = self.config.async_control.episode_builder.build()
        episode_batcher = self.config.async_control.episode_batcher.build(
            target_rollouts_per_training_batch=self._target_rollouts_per_training_batch,
            max_offpolicy_steps=self.config.async_control.max_offpolicy_steps,
            # read live: staleness is judged against the version about to train
            trainer_policy_version=lambda: self._trainer_policy_version,
            dp_degree=self.trainer_dp_degree,
            pad_id=self._pad_id,
        )
        packed_training_batch_queue: asyncio.Queue[
            PackedTrainingBatch | None
        ] = asyncio.Queue(maxsize=1)

        generate_fn = self._make_generate_fn("generator")
        input_task = asyncio.create_task(
            self._data_input_loop(self._rollout_buffer), name="data_input"
        )
        rollout_tasks = [
            asyncio.create_task(
                self._rollout_loop(
                    worker_id=worker_id,
                    buffer=self._rollout_buffer,
                    generate_fn=generate_fn,
                ),
                name=f"rollout_{worker_id}",
            )
            for worker_id in range(self._num_rollout_workers)
        ]
        episode_batcher_task = asyncio.create_task(
            self._episode_batcher_loop(
                buffer=self._rollout_buffer,
                episode_builder=episode_builder,
                episode_batcher=episode_batcher,
                packed_training_batch_queue=packed_training_batch_queue,
            ),
            name="episode_batcher",
        )
        trainer_task = asyncio.create_task(
            self._trainer_loop(
                packed_training_batch_queue, num_training_steps=num_training_steps
            ),
            name="trainer",
        )
        background_tasks = [input_task, *rollout_tasks, episode_batcher_task]
        try:
            # The trainer drives; if a producer crashes first, surface it rather than hang on the trainer.
            done, _ = await asyncio.wait(
                [trainer_task, *background_tasks], return_when=asyncio.FIRST_COMPLETED
            )
            for task in background_tasks:
                if task not in done:
                    continue
                if task.cancelled():
                    raise asyncio.CancelledError
                if (exc := task.exception()) is not None:
                    raise exc
            await trainer_task
        finally:
            # Always close the buffer (unblocks every loop) AND cancel/await the trainer too — a producer
            # failing first must not leave the trainer blocked on the packed-training-batch queue.
            await self._rollout_buffer.close()
            for task in (*background_tasks, trainer_task):
                task.cancel()
            await asyncio.gather(
                *background_tasks, trainer_task, return_exceptions=True
            )

        # TODO(async): the engine may still hold abandoned train requests when post-validation
        #   starts; they resolve harmlessly (validation uses a `val/` id namespace) but share the
        #   engine. Drain them first if it skews validation timing.
        post_validation = await self._validate_and_log(step=num_training_steps)
        self._log_reward_delta(pre_validation, post_validation)

    async def _validate_and_log(self, *, step: int) -> dict[str, float]:
        """Run one validation pass, log it, and return its aggregated values for the pre/post delta."""
        metrics = await self.validate(step=step)
        self.metrics_processor.log(step=step, metrics=metrics, is_validation=True)
        return m.MetricsProcessor._aggregate_metrics(metrics)

    def _log_reward_delta(self, pre: dict[str, float], post: dict[str, float]) -> None:
        """Console pre/post reward summary, visible without scrolling back through the loop."""
        reward_keys = sorted(key for key in set(pre) | set(post) if "reward" in key)
        logger.info("=" * 60)
        logger.info("Validation reward (pre / post):")
        for key in reward_keys:
            logger.info(
                f"  {key}:  {pre.get(key, float('nan')):+.3f}  /  {post.get(key, float('nan')):+.3f}"
            )
        logger.info("=" * 60)

    async def _data_input_loop(self, buffer: AsyncRolloutBuffer) -> None:
        """Read training samples off the event loop and admit them as rollout-group work.

        Blocks on a free buffer slot BEFORE reading a sample, so dataset reads are backpressured too.
        Returns when the buffer closes.
        """
        group_index = 0
        while await buffer.wait_for_rollout_group_slot():
            with sl.log_trace_span(
                "get_training_sample"
            ):  # dataset reads can block; keep them off the loop
                sample = await asyncio.to_thread(self._rollouter.get_training_sample)
            await buffer.add_rollout_group_work(
                RolloutGroupWork(
                    group_id=f"step={self._generator_policy_version + 1}/group={group_index}",
                    sample=sample,
                )
            )
            group_index += 1

    async def _rollout_loop(
        self, *, worker_id: int, buffer: AsyncRolloutBuffer, generate_fn: GenerateFn
    ) -> None:
        """Generate + score one group at a time; a failed group becomes an empty group + a failure metric.

        Staleness is enforced later (in the batcher), so this loop carries no version logic. Raw rollouts
        are recorded before any drop, so dropped groups stay inspectable on disk.
        """
        while True:
            work = await buffer.claim_next_rollout_group()
            if work is None:  # buffer closed
                return
            try:
                with sl.log_trace_span(
                    "rollout_group"
                ):  # auto-emits an _error record on exception
                    group = await self._rollouter.run_group_rollouts(
                        generate_fn=generate_fn,
                        sample=work.sample,
                        group_id=work.group_id,
                        group_size=self.config.async_control.group_size,
                        sampling=self._sampling,
                        renderer=self.renderer,
                    )
                group.metrics = [
                    *prepare_rollout_metrics("rollout", group.rollouts),
                    *_rollout_turn_metrics([group]),
                ]
                self.rollout_recorder.record(
                    step=self._generator_policy_version + 1,
                    is_validation=False,
                    rollout_groups=[group],
                )
            except Exception:
                logger.exception("rollout group %s failed; dropping", work.group_id)
                group = RolloutGroup(
                    group_id=work.group_id,
                    rollouts=[],
                    metrics=[m.Metric("rollout/group_failures", m.Sum(1.0))],
                )
            await buffer.record_rollout_group_result(group)

    async def _episode_batcher_loop(
        self,
        *,
        buffer: AsyncRolloutBuffer,
        episode_builder: EpisodeBuilder,
        episode_batcher: EpisodeBatcher,
        packed_training_batch_queue: "asyncio.Queue[PackedTrainingBatch | None]",
    ) -> None:
        """Take generated groups, build episodes, accumulate them, and queue each ready training batch.

        Packing runs in a worker thread (torch releases the GIL), so the event loop keeps serving the rollout
        workers while a batch packs. A clean close drains the buffer, flushes the remainder, then sends a
        `None` sentinel so the trainer stops.
        """
        while True:
            rollout_group = await buffer.take_generated_rollout_group()
            if rollout_group is None:  # closed and drained
                break
            with sl.log_trace_span("episode_builder"):
                episode_builder_output = episode_builder.build_episodes(
                    rollout_group=rollout_group
                )
            with sl.log_trace_span("episode_batcher_pack"):
                maybe_training_batches = await asyncio.to_thread(
                    episode_batcher.add_episode_group,
                    episode_builder_output=episode_builder_output,
                )
            for packed_training_batch in maybe_training_batches:
                await packed_training_batch_queue.put(packed_training_batch)
        if (remaining := episode_batcher.flush_remaining()) is not None:
            await packed_training_batch_queue.put(remaining)
        await packed_training_batch_queue.put(None)

    async def _trainer_loop(
        self,
        packed_training_batch_queue: "asyncio.Queue[PackedTrainingBatch | None]",
        *,
        num_training_steps: int,
    ) -> None:
        """Train rollout-count batches and sync generator weights inline (push then pull, both awaited).

        The `get()` wait is the only intended idle bubble (`perf/trainer_idle_ratio`).
        """
        for step in range(1, num_training_steps + 1):
            sl.set_step(step)  # propagate the step counter to the actors
            await self.trainer.sync_log_step.call(step)
            await self.generator_router.fanout("sync_log_step", step)
            step_metrics = StepMetricsRecorder()

            with step_metrics.record("timing/step/wait_for_training_batch"):
                packed = await packed_training_batch_queue.get()
            if packed is None:
                logger.info("Episode batcher closed and drained; stopping training")
                break

            # TODO(async): can't stream microbatches (interleave pack->train) — the loss is normalized by
            #   packed.num_global_valid_tokens (sum over ALL microbatches), needed before any fwd/bwd. To
            #   support streaming, accumulate raw loss/token counts across microbatches and scale before optim.
            with sl.log_trace_span("forward_backward"), step_metrics.record(
                "timing/step/train"
            ):
                microbatch_metrics = [
                    self._get_rank_0_value(
                        await self.trainer.forward_backward.call(
                            microbatch, packed.num_global_valid_tokens
                        )
                    )
                    for microbatch in packed.microbatches
                ]
                fwd_bwd = combine_microbatch_metrics(microbatch_metrics)
                if not math.isfinite(fwd_bwd["loss/mean"]):
                    logger.error("Loss is NaN/Inf; training diverged")
                    break
            with sl.log_trace_span("optim_step"), step_metrics.record(
                "timing/step/optim"
            ):
                optim = self._get_rank_0_value(await self.trainer.optim_step.call())
            self._trainer_policy_version = optim.policy_version

            # Inline weight sync: publish new weights before the next train step (push then pull, both awaited).
            # TODO(perf): overlap weight sync (today both are awaited synchronously). (a) Delay the PULL await
            #   until the next push: fire the pull as a task; the generator loads v_N while the trainer runs step
            #   N+1; await before the next push. Needs the model_state_dict_{version%2} double-buffer + a
            #   generator-side pull-duration return. (b) Delay the PUSH await until the next optim step (stage
            #   GPU->CPU on a side stream during fwd/bwd). Gated by timing/step/weight_sync/{push,pull}. Requires
            #   direct_rdma=False (under direct_rdma=True the generator reads live GPU weights, so an overlapped
            #   next optim corrupts the in-flight read).
            with sl.log_trace_span("weight_sync_push"), step_metrics.record(
                "timing/step/weight_sync/push"
            ):
                await self.trainer.push_model_state_dict.call(
                    version=optim.policy_version
                )
            with sl.log_trace_span("weight_sync_pull"), step_metrics.record(
                "timing/step/weight_sync/pull"
            ):
                await self.generator_router.pull_model_state_dict(
                    policy_version=optim.policy_version
                )
            self._generator_policy_version = optim.policy_version

            self.metrics_processor.log(
                step=step,
                is_validation=False,
                metrics=[
                    *packed.metrics,
                    *[
                        m.Metric(key, m.NoReduce(value))
                        for key, value in fwd_bwd.items()
                    ],
                    *[
                        m.Metric(key, m.NoReduce(value))
                        for key, value in optim.metrics.items()
                    ],
                    *self._rollout_buffer.metrics(),
                    *step_metrics.metrics(),
                    *_perf_ratio_metrics(
                        num_global_valid_tokens=packed.num_global_valid_tokens,
                        durations=step_metrics.durations,
                    ),
                ],
            )

            # TODO(async-rl): periodic mid-training validation — if validation.every_n_steps and
            # step % validation.every_n_steps == 0, run a validation pass here. Needs separate generator
            # scheduling to overlap the rollout workers; off by default (every_n_steps=0) -> pre/post only for now.
