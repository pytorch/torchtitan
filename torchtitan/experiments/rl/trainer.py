# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TLDR:
_data_input_loop -> _rollout_loop -> _batcher_loop -> training_batch_queue -> _trainer_loop
           v               v                    v
           |               |                    |
           |               ^                    ^
           +-------RolloutGroupWorkBuffer-------+

Detailed diagram:

_data_input_loop                                      _rollout_loop[N] (group workers)
+--------------------------------------------------+  +--------------------------------------------------+
| group_buffer.wait_for_slot()                     |  | work = group_buffer.claim_next()                  |
| sample = rollouter.get_training_sample()         |  | group = rollouter.run_group_rollouts(work.sample) |
| work = RolloutGroupWork(group_id, sample)        |  | group_buffer.finalize_work(group)                 |
| group_buffer.add_work(work)                      |  +-----------------------+--------------------------+
+-----------------------+--------------------------+                          ^ |
                        |                                                     | |
                        | adds work entry                                     | | updates same entry
                        v                                                     | v
RolloutGroupWorkBuffer
+-----------------------+-----------------------------------------------------+-+------------------------+
| capacity = max_offpolicy_steps * num_rollout_groups_per_train_step  (= S*B)                            |
|                                                                                                        |
| caller                  group_buffer call                       RolloutGroupWork _WorkState           |
| _data_input_loop        add_work(RolloutGroupWork)              WAITING                                |
| _rollout_loop[N]        claim_next()                            WAITING -> INFLIGHT                    |
| _rollout_loop[N]        finalize_work(RolloutGroup)             INFLIGHT -> FINALIZED                  |
| _batcher_loop           RolloutGroup = take_finalized()         FINALIZED -> removed                   |
+--------------------------------------------------------------------------------+----------------------+
                                                  |
                                                  | group = group_buffer.take_finalized()
                                                  v
_batcher_loop
+--------------------------------------------------------------------------------+
| training_sample_group = training_sample_builder.build_from_group(rollout_group=group) |
| maybe_training_batch = batcher.add_training_samples(training_sample_group)      |
| training_batch_queue.put(TrainingBatch)                                         |
+-----------------------+------------------------------+-------------------------+
                        |  ^                           |
          add group     |  | maybe_training_batch      | put batch
                        v  |                           v
Batcher                                     training_batch_queue
+----------------------------------------------+   +----------------------------------------------+
| accumulated TrainingSampleGroups             |   | size 1; holds TrainingBatch | None           |
| pack at target_groups_per_batch (= B) groups |   +---------------------+------------------------+
+----------------------------------------------+                         |
                                                                         | packed = training_batch_queue.get()
                                                                         v
_trainer_loop
+--------------------------------------------------------------------------------+
| train packed batch -> optim step -> push weights -> pull weights               |
+--------------------------------------------------------------------------------+

Backpressure (each stage: who fills it / who drains it):
_data_input_loop
  waits for: a free RolloutGroupWorkBuffer slot (producer: _data_input_loop.add_work)
  unblocked by:  _batcher_loop -> group_buffer.take_finalized() (consumer)

_rollout_loop[N]
  waits for: a claimable (WAITING) RolloutGroupWork (consumer: group_buffer.claim_next)
  unblocked by: _data_input_loop -> group_buffer.add_work() (producer)

_batcher_loop (take finalized group)
  waits for: the oldest group becoming FINALIZED (consumer: group_buffer.take_finalized)
  unblocked by: _rollout_loop[N] -> group_buffer.finalize_work() (producer)

_batcher_loop (put training batch)
  waits for: a free training_batch_queue slot (producer: training_batch_queue.put)
  unblocked by:  _trainer_loop -> training_batch_queue.get() (consumer)

_trainer_loop
  waits for: a TrainingBatch in training_batch_queue (consumer: training_batch_queue.get)
  unblocked by: _batcher_loop -> training_batch_queue.put() (producer)
"""

import asyncio
import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

# PYTORCH_CUDA_ALLOC_CONF is set in torchtitan/experiments/rl/__init__.py (before torch is imported)
# and in train.py; see the note there.
import torch  # noqa: F401
import torchstore as ts
from monarch.actor import ProcMesh
from monarch.spmd import setup_torch_elastic_env_async

from torchtitan.config import CompileConfig, Configurable
from torchtitan.experiments.rl.actors.generator import SamplingConfig, VLLMGenerator
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.batcher import Batcher
from torchtitan.experiments.rl.components.metrics_utils import (
    combine_microbatch_metrics,
    compute_perf_ratio_metrics,
    compute_policy_age_metrics,
    compute_rollout_metrics,
    MetricsTimer,
)
from torchtitan.experiments.rl.components.training_sample_builder import (
    TrainingSampleBuilder,
)
from torchtitan.experiments.rl.components.work_buffer import (
    RolloutGroupWork,
    RolloutGroupWorkBuffer,
)
from torchtitan.experiments.rl.generator_router import GeneratorRouter, RoutingContext
from torchtitan.experiments.rl.losses import GRPOLoss
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.rollout import RolloutGroup
from torchtitan.experiments.rl.rollout.rollouter import Rollouter
from torchtitan.experiments.rl.rollout.types import GenerateFn
from torchtitan.experiments.rl.rollout_recorder import RolloutSampleRecorder
from torchtitan.experiments.rl.types import Completion, TrainingBatch
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols.model_spec import ModelSpec

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class ValidationConfig:
    """Held-out validation that runs at the start and end of training"""

    # TODO: enable periodic validation with proper overlapping

    num_samples: int = 20
    """Held-out prompts scored greedily (temp=0, n=1) per validation pass. 0 skips validation."""


@dataclass(kw_only=True, slots=True)
class AsyncControlConfig(Configurable.Config):
    num_training_steps: int = 10
    """Optimizer steps to run."""

    num_rollout_groups_per_train_step: int = 8
    """Prompt groups whose surviving rollouts compose one train step (the batch target, in groups).
    NOTE: The number of tokens will vary from batch to batch and, due to packing, the global batch size will
    vary as well. Therefore, the number of accumulated microbatches changes at each step."""

    group_size: int = 8
    """Sibling rollouts sampled per prompt (the GRPO group)."""

    max_offpolicy_steps: int = 3
    """Max train-steps a rollout may lag the trainer. Sets the rollout buffer size (S*B) and its
    stalling behavior."""

    num_group_workers: int | None = None
    """Concurrent rollout-group workers. None -> num_rollout_groups_per_train_step (one train step's
    worth). Decoupled from the buffer depth; raise it to feed the generator more decode concurrency."""

    group_buffer: RolloutGroupWorkBuffer.Config = field(
        default_factory=RolloutGroupWorkBuffer.Config
    )
    training_sample_builder: TrainingSampleBuilder.Config = field(
        default_factory=TrainingSampleBuilder.Config
    )
    batcher: Batcher.Config = field(default_factory=Batcher.Config)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    def __post_init__(self):
        # Runtime sizing (buffer depth, worker count) is derived on the controller, not here:
        # AsyncControlConfig is slots=True, so it can't hold non-field derived attributes.
        # TODO(async-rl): support max_offpolicy_steps=0 via a strict on-policy mode.
        if self.max_offpolicy_steps == 0:
            raise ValueError(
                "max_offpolicy_steps=0 (true on-policy) is not supported yet"
            )


class RLController(Configurable):
    """Top-level RL async training orchestrator.

    Owns a `PolicyTrainer` actor (gradient updates), a `VLLMGenerator` actor
    (sampling), and a `Rollouter` (datasets + rubric + env construction).

    Check the docstring at the top of the file for more details.

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
        """Async-loop knobs: group/step counts, staleness, workers, and the group_buffer/builder/batcher/validation configs."""

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

        # TODO: put generator, num generators and generator router in a separate config
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

        # TODO: rename it to metrics_processor
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
                seq_len = self.async_control.batcher.batch.seq_len
                if sp_degree > 1 and seq_len % sp_degree != 0:
                    raise ValueError(
                        f"RL batcher sequence length ({seq_len}) must be divisible "
                        f"by sequence parallel degree ({sp_degree})."
                    )

            # TODO: add a check so that all seq_len related variables make sense
            # e.g. rollout max length cannot be larger than the model max_seq_len
            # or the packing len, etc.

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

        # Async-loop sizing, derived here (not on the slots config):
        #   buffer depth = max_offpolicy_steps * num_rollout_groups_per_train_step  (= S*B)
        #   workers      = num_group_workers (default = one train step of groups = B), decoupled from depth
        async_control = config.async_control
        self._max_buffered_rollout_groups = (
            async_control.max_offpolicy_steps
            * async_control.num_rollout_groups_per_train_step
        )
        self._num_group_workers = (
            async_control.num_group_workers
            if async_control.num_group_workers is not None
            else async_control.num_rollout_groups_per_train_step
        )
        if self._num_group_workers <= 0:
            raise ValueError(
                f"num_group_workers must be positive, got {self._num_group_workers}"
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
        generation metrics with `metrics_prefix` and pinning sticky routing on `routing_session_id` (a sample's
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
        rollout_concurrency = max(
            self._num_group_workers * self.config.async_control.group_size,
            self.config.async_control.validation.num_samples,
        )
        max_workers = max(1, min(rollout_concurrency, os.cpu_count() or 1))
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=max_workers)
        )

        config = self.config
        if not generator_meshes:
            raise ValueError("setup_async requires at least one generator mesh")

        trainer_parallelism = config.trainer.parallelism
        dp_shard = max(trainer_parallelism.data_parallel_shard_degree, 1)
        self.trainer_dp_degree = (
            trainer_parallelism.data_parallel_replicate_degree * dp_shard
        )

        # total number of generators
        generator_dp = max(config.generator.parallelism.data_parallel_degree, 1)
        num_generators = len(generator_meshes) * generator_dp

        # Ceiling (not target) for the generator's max_num_seqs: the per-generator share of peak
        # concurrent decode (num_group_workers * group_size siblings, or the validation pass), clamped.
        # Bounds the KV budget + the number of CUDA graphs captured.
        max_num_seqs = min(math.ceil(rollout_concurrency / num_generators), 512)

        logger.info(
            "max_num_seqs=%d per generator (rollout_concurrency=%d / generators=%d)",
            max_num_seqs,
            rollout_concurrency,
            num_generators,
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

            # TODO: enable torch.compile for trainer + generator. Generator inductor crashed the vLLM
            # engine (it shares the model path with vLLM's own inductor/cudagraph); find a working config.
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
                    compile_config=config.compile,
                    max_num_seqs=max_num_seqs,
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

        # Initial weight sync: only the trainer loads weights (HF/DCP formats), then publishes them for
        # the generators to pull -- so the generator never needs to know about checkpoint formats.
        with sl.log_trace_span("weight_sync_push"):
            await self.trainer.push_model_state_dict.call(version=0)
        with sl.log_trace_span("weight_sync_pull"):
            await self.generator_router.pull_model_state_dict(policy_version=0)

    @sl.log_trace_span("_collect_validation_rollouts")
    async def _collect_validation_rollouts(
        self, *, num_groups: int, sampling: SamplingConfig, step: int
    ) -> tuple[list[RolloutGroup], list[m.Metric]]:
        """Sample held-out prompts, run each greedily (n=1) concurrently, and emit validation metrics."""
        generate = self._make_generate_fn(metrics_prefix="validation_generator")
        samples = [self._rollouter.get_validation_sample() for _ in range(num_groups)]
        group_results = await asyncio.gather(
            *(
                self._rollouter.run_group_rollouts(
                    generate_fn=generate,
                    sample=sample,
                    # Negative ids keep validation disjoint from training group ids, so their
                    # request_ids can't collide in the shared engine (e.g. post-validation).
                    group_id=-(i + 1),
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
                    "validation group %d (step=%d) failed; dropping",
                    -(i + 1),
                    step,
                    exc_info=result,
                )
                num_failed_groups += 1
                continue
            rollout_groups.append(result)

        metrics = compute_rollout_metrics(
            prefix="validation",
            rollouts=[
                rollout for group in rollout_groups for rollout in group.rollouts
            ],
        )
        metrics.append(
            m.Metric("validation/group_failures", m.Sum(float(num_failed_groups)))
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

        self.rollout_recorder.record(
            step=step, is_validation=True, rollout_groups=rollout_groups
        )

        t_validate_s = time.perf_counter() - t_validate_start
        validation_metrics.append(m.Metric("timing/validate", m.NoReduce(t_validate_s)))
        return validation_metrics

    async def run(self) -> None:
        """Start all async loops: `_trainer_loop` drives; when it finishes or any stage crashes,
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

        # buffer
        self._rollout_buffer = self.config.async_control.group_buffer.build(
            max_buffered_rollout_groups=self._max_buffered_rollout_groups,
        )

        # training_sample_builder
        training_sample_builder = (
            self.config.async_control.training_sample_builder.build()
        )

        # batcher
        batcher = self.config.async_control.batcher.build(
            target_groups_per_batch=self.config.async_control.num_rollout_groups_per_train_step,
            max_offpolicy_steps=self.config.async_control.max_offpolicy_steps,
            # lambda: lets the batcher read the current trainer policy version at any time.
            trainer_policy_version_getter=lambda: self._trainer_policy_version,
            dp_degree=self.trainer_dp_degree,
            pad_id=self._pad_id,
        )

        # training_batch_queue
        training_batch_queue: asyncio.Queue[TrainingBatch | None] = asyncio.Queue(
            maxsize=1
        )

        # rollout_loop
        generate_fn = self._make_generate_fn(metrics_prefix="generator")

        rollout_tasks = [
            asyncio.create_task(
                self._rollout_loop(
                    group_buffer=self._rollout_buffer,
                    generate_fn=generate_fn,
                ),
                name=f"rollout_worker_{group_worker_id}",
            )
            for group_worker_id in range(self._num_group_workers)
        ]

        # data_input_loop
        data_input_task = asyncio.create_task(
            self._data_input_loop(self._rollout_buffer), name="data_input"
        )

        # training_sample_batcher_loop
        batcher_task = asyncio.create_task(
            self._batcher_loop(
                group_buffer=self._rollout_buffer,
                training_sample_builder=training_sample_builder,
                batcher=batcher,
                training_batch_queue=training_batch_queue,
            ),
            name="batcher",
        )

        # trainer_loop
        trainer_task = asyncio.create_task(
            self._trainer_loop(
                training_batch_queue, num_training_steps=num_training_steps
            ),
            name="trainer",
        )

        # run everything until trainer finishes its number of steps
        # or some other loop breaks
        background_tasks = [
            data_input_task,
            *rollout_tasks,
            batcher_task,
        ]
        try:
            done, _ = await asyncio.wait(
                [trainer_task, *background_tasks], return_when=asyncio.FIRST_COMPLETED
            )
            # The trainer is the finite clock: it runs num_training_steps then returns -> training is done.
            # Producers loop forever, so a producer in `done` means it crashed (or wrongly returned);
            # await it to surface the exception instead of hanging on the still-running trainer. Check
            # producers even when the trainer also finished this turn, so a simultaneous crash isn't hidden.
            for task in done:
                if task is trainer_task:
                    continue
                await task  # re-raises a real crash
                raise RuntimeError(f"{task.get_name()} exited unexpectedly")
            if trainer_task in done:
                await trainer_task
        finally:
            # close() wakes the buffer's waiters; cancellation wakes anything blocked on the queue.
            await self._rollout_buffer.close()
            for task in (*background_tasks, trainer_task):
                task.cancel()
            await asyncio.gather(
                *background_tasks, trainer_task, return_exceptions=True
            )

        # TODO(async): the engine may still hold abandoned train requests when post-validation
        #   starts; they resolve harmlessly (validation uses negative group ids, disjoint from
        #   training) but share the engine. Drain them first if it skews validation timing.
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

    async def _data_input_loop(self, group_buffer: RolloutGroupWorkBuffer) -> None:
        """Upon a slot becoming available in the `RolloutGroupWorkBuffer`, gets a training prompt
        from the `rollouter` a puts a `RolloutGroupWork` in the buffer. This will notify
        the `_rollout_loop` to claim the work and start generating.

        waits_for: a free `RolloutGroupWorkBuffer` slot
        unblocked_by: `_batcher_loop` calling `group_buffer.take_finalized()`, which frees a slot.
        """
        group_index = 0
        while await group_buffer.wait_for_slot():
            with sl.log_trace_span("get_training_sample"):
                # to_thread: Dont block on dataset reads
                sample = await asyncio.to_thread(self._rollouter.get_training_sample)
            await group_buffer.add_work(
                RolloutGroupWork(
                    group_id=group_index,
                    sample=sample,
                )
            )
            group_index += 1

    async def _rollout_loop(
        self, *, group_buffer: RolloutGroupWorkBuffer, generate_fn: GenerateFn
    ) -> None:
        """Generate + score one group at a time; a failed group becomes an empty group + a failure metric.

        Staleness is enforced later (in the batcher), so this loop carries no version logic. Raw rollouts
        are recorded before any drop, so dropped groups stay inspectable on disk.

        waits_for: a claimable WAITING group (group_buffer.claim_next)
        unblocked_by: `_data_input_loop` calling `group_buffer.add_work()`
        """
        while True:
            work = await group_buffer.claim_next()
            if work is None:  # group_buffer closed/shutdown signal
                return
            try:
                with sl.log_trace_span("rollout_group"):
                    group = await self._rollouter.run_group_rollouts(
                        generate_fn=generate_fn,
                        sample=work.sample,
                        group_id=work.group_id,
                        group_size=self.config.async_control.group_size,
                        sampling=self._sampling,
                        renderer=self.renderer,
                    )
                group.metrics = compute_rollout_metrics(
                    prefix="rollout", rollouts=group.rollouts
                )
                # Record under the version this group was sampled at (the generator's current version).
                # Per-turn min/max_policy_version on the rollout carry the exact attribution.
                self.rollout_recorder.record(
                    step=self._generator_policy_version,
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
            await group_buffer.finalize_work(group)

    async def _batcher_loop(
        self,
        *,
        group_buffer: RolloutGroupWorkBuffer,
        training_sample_builder: TrainingSampleBuilder,
        batcher: Batcher,
        training_batch_queue: "asyncio.Queue[TrainingBatch | None]",
    ) -> None:
        """Take finalized groups, build training_samples, accumulate them, and queue each ready training batch.

        Packing runs in a worker thread (torch releases the GIL), so the event loop keeps serving the rollout
        workers while a batch packs. On a clean close the group_buffer drains and returns None; we forward a
        `None` sentinel so the trainer stops.

        waits_for (in):  the oldest group becoming FINALIZED (group_buffer.take_finalized)
        unblocked_by:    `_rollout_loop` calling `group_buffer.finalize_work()`
        waits_for (out): a free training_batch_queue slot (training_batch_queue.put)
        unblocked_by:    `_trainer_loop` calling `training_batch_queue.get()`
        """
        while True:
            rollout_group = await group_buffer.take_finalized()
            if rollout_group is None:  # closed and drained
                break
            with sl.log_trace_span("training_sample_builder"):
                training_sample_group = training_sample_builder.build_from_group(
                    rollout_group=rollout_group
                )

            # We put a group in. We may get a batch back
            # if there are enough accumulated trainable groups to return one.
            with sl.log_trace_span("batcher_pack"):
                maybe_training_batch = await asyncio.to_thread(
                    batcher.add_training_samples,
                    training_sample_group=training_sample_group,
                )
            if maybe_training_batch is not None:
                await training_batch_queue.put(maybe_training_batch)
        await training_batch_queue.put(None)
        # TODO(async-rl): if finite datasets are supported, drain a final partial batch here.

    async def _trainer_loop(
        self,
        training_batch_queue: "asyncio.Queue[TrainingBatch | None]",
        *,
        num_training_steps: int,
    ) -> None:
        """Train group-count batches and sync generator weights inline (push then pull, both awaited).

        The `get()` wait is the only intended idle bubble (`perf/trainer_idle_ratio`).

        waits_for: a TrainingBatch in training_batch_queue (training_batch_queue.get)
        unblocked_by: `_batcher_loop` calling `training_batch_queue.put()`
        """
        for step in range(1, num_training_steps + 1):
            sl.set_step(step)  # propagate the step counter to the actors
            await self.trainer.sync_log_step.call(step)
            await self.generator_router.fanout("sync_log_step", step)
            metrics_timer = MetricsTimer()

            with sl.log_trace_span("wait_for_training_batch"), metrics_timer.record(
                "timing/step/wait_for_training_batch"
            ):
                packed = await training_batch_queue.get()
            if packed is None:
                logger.info("Batcher closed and drained; stopping training")
                break

            # Policy age is computed HERE, at consumption time, against the live trainer version, so it is
            # faithful to what this step trains on -- not the version when the batch was packed.
            policy_age_panel = compute_policy_age_metrics(
                trainer_policy_version=self._trainer_policy_version,
                min_policy_versions=packed.min_policy_versions,
            )

            # TODO(async): can't stream microbatches (interleave pack->train) — the loss is normalized by
            #   packed.num_global_valid_tokens (sum over ALL microbatches), needed before any fwd/bwd. To
            #   support streaming, accumulate raw loss/token counts across microbatches and scale before optim.
            with sl.log_trace_span("forward_backward"), metrics_timer.record(
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
            with sl.log_trace_span("optim_step"), metrics_timer.record(
                "timing/step/optim"
            ):
                optim = self._get_rank_0_value(await self.trainer.optim_step.call())
            self._trainer_policy_version = optim.policy_version

            # Weight sync: publish new weights before the next train step (push then pull, both awaited).
            # TODO(perf): overlap weight sync (today both are awaited synchronously). (a) Delay the PULL await
            #   until the next push: fire the pull as a task; the generator loads v_N while the trainer runs step
            #   N+1; await before the next push. Needs the model_state_dict_{version%2} double-buffer + a
            #   generator-side pull-duration return. (b) Delay the PUSH await until the next optim step (stage
            #   GPU->CPU on a side stream during fwd/bwd). Gated by timing/step/weight_sync/{push,pull}. Requires
            #   direct_rdma=False (under direct_rdma=True the generator reads live GPU weights, so an overlapped
            #   next optim corrupts the in-flight read).
            with sl.log_trace_span("weight_sync_push"), metrics_timer.record(
                "timing/step/weight_sync/push"
            ):
                await self.trainer.push_model_state_dict.call(
                    version=optim.policy_version
                )
            with sl.log_trace_span("weight_sync_pull"), metrics_timer.record(
                "timing/step/weight_sync/pull"
            ):
                await self.generator_router.pull_model_state_dict(
                    policy_version=optim.policy_version
                )
            self._generator_policy_version = optim.policy_version

            # Snapshot durations before metrics() drains the timer (the perf panel reads them below).
            durations = dict(metrics_timer.durations)
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
                    *metrics_timer.metrics(),
                    *policy_age_panel,
                    *compute_perf_ratio_metrics(
                        num_global_valid_tokens=packed.num_global_valid_tokens,
                        durations=durations,
                    ),
                ],
            )
