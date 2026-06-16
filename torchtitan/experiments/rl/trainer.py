# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Runs the async RL dataflow.

Stages:
    input: read training samples
    rollout: generate and score GRPO sibling rollouts
    batcher: drop stale episodes and pack batches
    trainer: run fwd/bwd, optimizer step, and generator weight sync
"""

import asyncio
import logging
import math
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import NamedTuple

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
from torchtitan.experiments.rl.admission_budget import AdmissionBudget
from torchtitan.experiments.rl.batcher import Batcher, PackedBatch
from torchtitan.experiments.rl.episode_buffer import EpisodeBuffer
from torchtitan.experiments.rl.generator_router import GeneratorRouter, RoutingContext
from torchtitan.experiments.rl.losses import GRPOLoss
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.rollout import (
    prepare_rollout_metrics,
    rollout_to_episodes,
    RolloutGroup,
)
from torchtitan.experiments.rl.rollout.rollouter import Rollouter
from torchtitan.experiments.rl.rollout.types import GenerateFn
from torchtitan.experiments.rl.rollout_recorder import RolloutSampleRecorder
from torchtitan.experiments.rl.types import Completion, Episode, RolloutID
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols.model_spec import ModelSpec

logger = logging.getLogger(__name__)


def _generation_metrics(groups: list[RolloutGroup]) -> list[m.Metric]:
    """Flatten every turn's per-generation metrics (latencies, output tokens) across the groups.

    Shared by the rollout loop (`_rollout_loop`, one group) and validation
    (`_collect_rollouts`, a round of groups).
    """
    return [
        metric
        for group in groups
        for rollout in group.rollouts
        for rollout_turn in rollout.turns
        for metric in rollout_turn.metrics
    ]


class _PendingPull(NamedTuple):
    """A generator weight-load fired into the background; awaited before the next publish (§5.1)."""

    task: asyncio.Task
    version: int


def _reduce_microbatches(per_microbatch: list[dict[str, float]]) -> dict[str, float]:
    """Reduce per-microbatch loss metrics over the grad-accumulation: mean/frac keys are pre-normalized
    by num_global_valid_tokens so summing them reconstructs the global value; max keys take the max.

    Matches `_mean`/`_frac` as well as `/mean`/`/frac` so the GRPO loss keys (`loss/ratio_mean`,
    `loss/ratio_clipped_frac`, `loss/generator_logprob_nan_frac`) are summed too — they are all
    pre-normalized, so without this they would report ~1/grad_accum of the truth.

    Example:

        _reduce_microbatches([{"loss/ratio_clipped_frac": 0.1, "x/max": 2.0},
                              {"loss/ratio_clipped_frac": 0.2, "x/max": 5.0}])
        # -> {"loss/ratio_clipped_frac": 0.3, "x/max": 5.0}
    """
    reduced: dict[str, float] = {}
    for microbatch in per_microbatch:
        for key, value in microbatch.items():
            if key not in reduced:
                reduced[key] = value
            elif key.endswith("/max"):
                reduced[key] = max(reduced[key], value)
            elif key.endswith(("/mean", "_mean", "/frac", "_frac")):
                reduced[key] += value
    return reduced


def _build_step_metrics(
    *,
    batch: PackedBatch,
    fwd_bwd: dict[str, float],
    optimizer: dict[str, float],
    wait_s: float,
    train_s: float,
    push_s: float,
    pull_wait_s: float,
    sync_s: float,
    step_s: float,
) -> list[m.Metric]:
    """All metrics for one async train step (pure: no I/O, unit-testable).

    `batch.metrics` already carry the buffer's `perf/buffer_*` panel keys + rollout/generation/packing
    drill-down that rode out with the batch; this adds fwd/bwd, optimizer, the 4 trainer-side `perf/`
    panel ratios, and `timing/*` drill-down.

    The gap between goodput (`/ step_s`) and active (`/ train_s`) IS the trainer-idle + sync bubble.

    Example:

        # step_s=10, wait_s=8, train_s=1, sync_s=0.5, num_global_valid_tokens=100
        # -> perf/trainer_idle_ratio=0.8, perf/weight_sync_overhead_ratio=0.05,
        #    perf/goodput_tokens_per_second=10, perf/active_tokens_per_second=100
    """
    tokens = batch.num_global_valid_tokens
    metrics: list[m.Metric] = list(batch.metrics)
    metrics += [m.Metric(key, m.NoReduce(value)) for key, value in fwd_bwd.items()]
    metrics += [m.Metric(key, m.NoReduce(value)) for key, value in optimizer.items()]
    derived = [
        ("perf/trainer_idle_ratio", wait_s / step_s if step_s else 0.0),
        ("perf/goodput_tokens_per_second", tokens / step_s if step_s else 0.0),
        ("perf/active_tokens_per_second", tokens / train_s if train_s else 0.0),
        ("perf/weight_sync_overhead_ratio", sync_s / step_s if step_s else 0.0),
        ("timing/step", step_s),
        ("timing/get_batch", wait_s),
        ("timing/train", train_s),
        ("timing/weight_sync", sync_s),
        # push (GPU->CPU stage) vs pull_wait (un-overlapped tail of the prior generator load): which
        # half of the sync dominates — the drill-down that gates overlapping the push too (RFC §12).
        ("timing/weight_sync/push", push_s),
        ("timing/weight_sync/pull_wait", pull_wait_s),
    ]
    metrics += [m.Metric(key, m.NoReduce(value)) for key, value in derived]
    return metrics


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

        num_steps: int = 10
        """Number of RL training steps."""

        dump_folder: str = "outputs/rl"
        """Root output folder for RL artifacts (temp weights, logs, etc.)."""

        num_rollout_workers: int = 16
        """Rollout workers generating GRPO groups concurrently (one group each).
        Rollouts in flight = `num_rollout_workers * group_size`. Recommendation: Size it to overproduce ~4x the
        trainer's per-step consumption (`~4 * groups_consumed_per_step`) and let the stale-drop trim
        the surplus, rather than sizing it exactly."""

        max_queued_batches: int = 1
        """Packed batches the batcher may park ahead of the trainer (the `batch_queue` depth)."""

        max_num_seqs: int | None = None
        """vLLM engine's max concurrent sequences (KV budget + cudagraph sizes). None =
        `num_rollout_workers * group_size` (legacy: no engine queue). Set it BELOW the
        in-flight rollout count to keep a FCFS refill queue, so the engine stays at this batch
        continuously instead of dipping during group tails (continuous feeding)."""

        group_size: int = 8
        """Sibling rollouts sampled per dataset row (the GRPO group). The generator
        is always called with `n=1`; prompts are pre-expanded by `group_size`."""

        num_validation_samples: int = 20
        """Number of held-out prompts scored greedily (temp=0, n=1) per validation pass."""

        max_offpolicy_steps: int | None = 3
        """Off-policy bound: drop an episode whose oldest token is more than this many versions behind
        the trainer's optimizer version. The generator adopts weights >=1 version behind the trainer,
        so 1 is the effective on-policy floor. None = never drop (log staleness only). See `EpisodeBuffer`."""

        drop_rollout_group_if_any_stale: bool = False
        """Drop a whole GRPO group if ANY of its episodes is stale, keeping groups intact (instead of
        dropping stale episodes individually). No effect when `max_offpolicy_steps is None`."""

        max_buffered_batches: int = 2
        """Depth bound: batches the rollout producer may bank ahead before backpressure.
        Set >= `max_offpolicy_steps + 1` to use the full staleness budget."""

        max_active_rollout_groups: int | None = None
        """Pre-generation gate: rollout groups a worker may have in flight before the trainer consumes
        them (`AdmissionBudget`). None = unbounded. A scale knob — bound it to ~2 batches of groups so
        generation can't run far ahead and produce born-stale work; leave None at small scale where
        the depth bound suffices and a tight gate would idle workers."""

        validation_freq: int = 0
        """Steps between mid-training validation passes. 0 = validate only before/after training.
        TODO(async): not yet wired into run()'s async loop (only pre/post validation runs); periodic
        validation is generation-only and must overlap the rollout workers (see `_trainer_loop`)."""

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

        batcher: Batcher.Config = field(default_factory=Batcher.Config)
        """Batcher config: local_batch_size, seq_len."""

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
                seq_len = self.batcher.batch.seq_len
                if sp_degree > 1 and seq_len % sp_degree != 0:
                    raise ValueError(
                        f"RL batcher sequence length ({seq_len}) must be divisible "
                        f"by sequence parallel degree ({sp_degree})."
                    )
            if self.max_offpolicy_steps == 0:
                raise ValueError(
                    "max_offpolicy_steps=0 deadlocks the async loop: staleness is measured against the "
                    "trainer's optimizer version, which is >=1 ahead of the version the generator has "
                    "adopted, so no freshly-sampled episode is ever at the trainer's current version. "
                    "Use >=1 (1 is the effective on-policy floor), or None to disable dropping."
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
        self.batcher = Batcher(
            config.batcher, pad_id=self.renderer._tokenizer.eos_token_id
        )
        self._rollouter: Rollouter = config.rollouter.build()
        self.rollout_recorder = config.rollout_recorder.build(
            dump_dir=config.dump_folder
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
        metrics with `metrics_prefix` and routing stickily on `rollout_id.group_id` (so a GRPO group's
        turns reuse one generator's prefix KV)."""

        @sl.log_trace_span("generate")
        async def generate(
            prompt_token_ids: list[int],
            *,
            rollout_id: RolloutID,
            sampling_config: SamplingConfig | None = None,
        ) -> Completion | None:
            result = await self.generator_router.route(
                "generate",
                prompt_token_ids,
                request_id=rollout_id.request_id,
                sampling_config=sampling_config,
                metrics_prefix=metrics_prefix,
                routing_ctx=RoutingContext(
                    estimated_cost=len(prompt_token_ids),
                    session_key=rollout_id.group_id,
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
            self.config.num_rollout_workers * self.config.group_size,
            self.config.num_validation_samples,
        )
        max_workers = max(1, min(max_concurrent_rollouts, os.cpu_count() or 1))
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=max_workers)
        )

        config = self.config
        if not generator_meshes:
            raise ValueError("setup_async requires at least one generator mesh")

        # We have 3 knobs for sequence length:
        # a) the packed batch `batcher.batch.seq_len`
        # b) the generator's context `model.max_seq_len`
        # c) the rollout's max tokens `token_env.max_rollout_tokens`
        # We need it to be `a >= b >= c`.
        seq_len = config.batcher.batch.seq_len
        model_max_seq_len = config.model_spec.model.max_seq_len
        rollout_cap = getattr(
            getattr(config.rollouter, "token_env", None), "max_rollout_tokens", None
        )
        if seq_len < model_max_seq_len and (
            rollout_cap is None or rollout_cap > seq_len
        ):
            raise ValueError(
                f"batcher seq_len ({seq_len}) is smaller than the longest episode a rollout can "
                f"produce (generator max_seq_len={model_max_seq_len}, rollouter "
                f"max_rollout_tokens={rollout_cap}); episodes longer than seq_len are silently "
                f"dropped during packing and crash the trainer. Set batcher.batch.seq_len >= "
                f"{model_max_seq_len}, or rollouter.token_env.max_rollout_tokens <= {seq_len}."
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
                    max_num_seqs=(
                        config.max_num_seqs
                        if config.max_num_seqs is not None
                        else max(
                            config.num_rollout_workers * config.group_size,
                            config.num_validation_samples,
                        )
                    ),
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

    @sl.log_trace_span("_collect_rollouts")
    async def _collect_rollouts(
        self,
        *,
        is_validation: bool,
        num_groups: int,
        group_size: int,
        sampling: SamplingConfig,
        step: int,
        group_offset: int,
    ) -> tuple[list[RolloutGroup], list[m.Metric]]:
        """Sample dataset items, run each group's rollouts concurrently, emit metrics.

        Hands each prompt to `Rollouter.run_group_rollouts` with a `GenerateFn` bound to the
        generator, which returns a RolloutGroup.

        Args:
            is_validation: Sample from the validation dataset (else train).
            num_groups: Number of prompt groups to collect this call.
            group_size: Sibling rollouts per group (generator runs n=1).
            sampling: SamplingConfig for every generate call.
            step: Training step, tagged into group_id for metrics + debugging.
            group_offset: Starting group index so group_ids stay unique across rounds in a step.

        Returns:
            Scored rollout groups plus rollout/generator metrics.
        """
        generation_metrics_prefix = (
            "validation_generator" if is_validation else "generator"
        )
        rollout_metrics_prefix = "validation" if is_validation else "rollout"
        generate = self._make_generate_fn(generation_metrics_prefix)

        # Validation ids live in their own namespace so they never collide with a train
        # request still in flight in the long-lived continuous-batching engine.
        group_prefix = "val/" if is_validation else ""

        # Sample one dataset item per group, then run every group's rollouts concurrently.
        samples = [
            (
                self._rollouter.get_validation_sample()
                if is_validation
                else self._rollouter.get_training_sample()
            )
            for _ in range(num_groups)
        ]
        group_results = await asyncio.gather(
            *(
                self._rollouter.run_group_rollouts(
                    generate_fn=generate,
                    sample=sample,
                    group_id=f"{group_prefix}step={step}/group={group_offset + i}",
                    group_size=group_size,
                    sampling=sampling,
                    renderer=self.renderer,
                )
                for i, sample in enumerate(samples)
            ),
            return_exceptions=True,
        )

        # Keep the groups that succeeded; log + count the ones that raised.
        rollout_groups: list[RolloutGroup] = []
        num_failed_groups = 0
        for i, result in enumerate(group_results):
            if isinstance(result, BaseException):
                logger.error(
                    "group %sstep=%d/group=%d failed; dropping",
                    group_prefix,
                    step,
                    group_offset + i,
                    exc_info=result,
                )
                num_failed_groups += 1
                continue
            rollout_groups.append(result)

        # Metrics ride with the rollout: flatten each turn's per-generation metrics, add failures.
        # TODO: it is confusing what metrics belong in the rollout turn and what metrics
        # should be calculated here in the controller. It seems that we should move
        # all metrics calculation to the rollout loop, so there is no confusion or metric duplication
        # TODO: we may also have some metrics at the Rollout level, i.e. not turn specific.
        # TODO: we also need a "logs" field, so that if there were errors/warnings
        # they can be made available for the rollout_logger
        metrics = _generation_metrics(rollout_groups)
        metrics.append(
            m.Metric(
                f"{rollout_metrics_prefix}/group_failures",
                m.Sum(float(num_failed_groups)),
            )
        )
        metrics += prepare_rollout_metrics(
            rollout_metrics_prefix,
            [rollout for group in rollout_groups for rollout in group.rollouts],
        )
        return rollout_groups, metrics

    @staticmethod
    @sl.log_trace_span("_build_episodes")
    def _build_episodes(
        rollout_groups: list[RolloutGroup],
    ) -> tuple[list[Episode], list[m.Metric]]:
        """Flatten scored rollout groups into training episodes.

        Skips rollouts without training tokens and emits episode-level metrics.

        Args:
            rollout_groups: Scored rollout groups from one round.

        Returns:
            Train episodes plus episode-level metrics.
        """
        episodes: list[Episode] = []
        group_stds: list[float] = []
        # Iteratate over `Rollouts` and `Episode`s. More than one `Episode` may be produced
        # per rollout if the history up to turn N is not a prefix of turn N+1, indicating history branching.
        branches_per_rollout: list[float] = []
        advantages_per_rollout: list[float] = []
        for group in rollout_groups:
            # Drop the whole group if no sibling has trainable tokens
            if any(
                not any(turn.completion_token_ids for turn in rollout.turns)
                for rollout in group.rollouts
            ):
                logger.warning(
                    "group %s has an untrainable rollout; dropping the group",
                    group.group_id,
                )
                continue

            # Advantage was already filled by the Rollouter's advantage estimator; here
            # we only collect each group's reward std for the metric emitted below.
            group_stds.append(
                statistics.pstdev([rollout.reward for rollout in group.rollouts])
            )
            for rollout in group.rollouts:
                rollout_episodes = rollout_to_episodes(rollout)
                episodes.extend(rollout_episodes)
                branches_per_rollout.append(float(len(rollout_episodes)))
                advantages_per_rollout.append(rollout.advantage)

        # TODO: drop groups with zero std
        zero_std = [1.0 if std == 0.0 else 0.0 for std in group_stds]
        episode_metrics: list[m.Metric] = [
            m.Metric("advantage", m.SummaryStats.from_list(advantages_per_rollout)),
            m.Metric("rollout_reward/group_zero_std_frac", m.Mean.from_list(zero_std)),
            m.Metric(
                "rollout/branches_per_rollout", m.Mean.from_list(branches_per_rollout)
            ),
            m.Metric(
                "rollout/branches_per_rollout", m.Max.from_list(branches_per_rollout)
            ),
        ]

        # Per-rollout policy versions. We log max/min in case episodes come
        # from multiple rollout versions.
        policy_versions = [episode.policy_version for episode in episodes]
        if policy_versions:
            episode_metrics.extend(
                [
                    m.Metric(
                        "rollout/policy_version", m.Min.from_list(policy_versions)
                    ),
                    m.Metric(
                        "rollout/policy_version", m.Max.from_list(policy_versions)
                    ),
                ]
            )
        return episodes, episode_metrics

    # TODO: we currently determine num_validation_samples
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
        num_samples = self.config.num_validation_samples
        if num_samples == 0:  # skip validation (e.g. loss guard CI)
            return []
        greedy = replace(self._sampling, temperature=0.0, top_p=1.0)

        rollout_groups, validation_metrics = await self._collect_rollouts(
            is_validation=True,
            num_groups=num_samples,
            group_size=1,
            sampling=greedy,
            step=step,
            group_offset=0,
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
        num_steps = self.config.num_steps
        logger.info(
            f"Pre-training validation; then {num_steps} steps of async RL training"
        )

        pre_validation = await self._validate_and_log(step=0)
        sl.log_trace_instant("training_start")

        # Two policy version pointers: the trainer advances at the optimizer step;
        # the generator version advances when a weight pull completes.
        self._trainer_policy_version = 0
        self._generator_policy_version = 0

        # the in-flight generator weight-load. Used to overlap weight-pull with
        self._pending_pull: _PendingPull | None = None

        # Pre-generation gate: workers acquire before generating a group, released as the group drains.
        self._admission_budget = AdmissionBudget(self.config.max_active_rollout_groups)
        buffer = EpisodeBuffer(
            batcher=self.batcher,
            dp_degree=self.trainer_dp_degree,
            max_offpolicy_steps=self.config.max_offpolicy_steps,
            drop_rollout_group_if_any_stale=self.config.drop_rollout_group_if_any_stale,
            max_buffered_batches=self.config.max_buffered_batches,
            # read live: drops enforce staleness against the version being trained
            train_version=lambda: self._trainer_policy_version,
            on_round_drained=self._admission_budget.release,
        )
        sample_queue: asyncio.Queue = asyncio.Queue(
            maxsize=self.config.num_rollout_workers
        )
        batch_queue: asyncio.Queue[PackedBatch | None] = asyncio.Queue(
            maxsize=self.config.max_queued_batches
        )

        generate = self._make_generate_fn("generator")
        stages = [
            asyncio.create_task(self._input_loop(sample_queue), name="input"),
            *[
                asyncio.create_task(
                    self._rollout_loop(worker, sample_queue, buffer, generate),
                    name=f"rollout_{worker}",
                )
                for worker in range(self.config.num_rollout_workers)
            ],
            asyncio.create_task(
                self._batcher_loop(buffer, batch_queue), name="batcher"
            ),
        ]
        trainer_task = asyncio.create_task(
            self._trainer_loop(batch_queue, num_steps=num_steps), name="trainer"
        )
        try:
            # The trainer drives; if a production stage crashes first, surface it rather than hang.
            await asyncio.wait(
                [trainer_task, *stages], return_when=asyncio.FIRST_COMPLETED
            )
            for stage in stages:
                if stage.done() and not stage.cancelled() and stage.exception():
                    raise stage.exception()
            await trainer_task
        finally:
            for task in (trainer_task, *stages):
                task.cancel()
            # Cancel the in-flight weight-pull too, so an outer cancel (Ctrl-C) doesn't orphan it.
            if self._pending_pull is not None and not self._pending_pull.task.done():
                self._pending_pull.task.cancel()
            await asyncio.gather(trainer_task, *stages, return_exceptions=True)
            await buffer.close()

        # TODO(async): the engine may still hold abandoned train requests when post-validation
        #   starts; they resolve harmlessly (validation uses a `val/` id namespace) but share the
        #   engine. Drain them first if it skews validation timing.
        post_validation = await self._validate_and_log(step=num_steps)
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

    async def _input_loop(self, sample_queue: asyncio.Queue) -> None:
        """Input stage: read training samples off the event loop and queue them, so a rollout worker
        always has a ready sample to start on (the queue depth is the prefetch)."""
        while True:
            sample = await asyncio.to_thread(self._get_training_sample)
            await sample_queue.put(sample)

    @sl.log_trace_span("_get_training_sample")
    def _get_training_sample(self):
        return self._rollouter.get_training_sample()

    async def _rollout_loop(
        self,
        worker: int,
        sample_queue: asyncio.Queue,
        buffer: EpisodeBuffer,
        generate: GenerateFn,
    ) -> None:
        """Rollout stage (the actor stage): pull a sample, generate + score its GRPO group into
        episodes, and add them to the buffer. A group that raises anywhere is logged + dropped as an
        empty round (its `group_failures` rides out with the next batch, and its admission permit still
        releases); the worker keeps going. Staleness is dropped in the batcher loop, so `add_episodes`
        carries no version."""
        group_count = 0
        while True:
            sample = await sample_queue.get()
            # Acquire a group permit BEFORE generating, so generation can't run far ahead of the
            # trainer (released when this group's round drains from the buffer).
            await self._admission_budget.acquire()
            group_id = f"step={self._generator_policy_version + 1}/group=w{worker}_{group_count}"
            group_count += 1
            # One add_episodes per iteration on EVERY path (success or failure) so each acquired permit
            # is matched by exactly one round that drains and releases it.
            try:
                with sl.log_trace_span("run_group_rollouts"):
                    group = await self._rollouter.run_group_rollouts(
                        generate_fn=generate,
                        sample=sample,
                        group_id=group_id,
                        group_size=self.config.group_size,
                        sampling=self._sampling,
                        renderer=self.renderer,
                    )
                episodes, episode_metrics = self._build_episodes([group])
                rollout_metrics = prepare_rollout_metrics("rollout", group.rollouts)
                generation_metrics = _generation_metrics([group])
                self.rollout_recorder.record(
                    step=self._generator_policy_version + 1,
                    is_validation=False,
                    rollout_groups=[group],
                )
                await buffer.add_episodes(
                    episodes, rollout_metrics + generation_metrics + episode_metrics
                )
            except Exception:
                logger.exception("rollout group %s failed; dropping", group_id)
                await buffer.add_episodes(
                    [], [m.Metric("rollout/group_failures", m.Sum(1.0))]
                )

    async def _batcher_loop(
        self, buffer: EpisodeBuffer, batch_queue: asyncio.Queue
    ) -> None:
        """Batcher stage: wait for a full fresh batch of episodes, pack it OFF the event loop, and park
        it on `batch_queue` so the trainer's pull is instant. Forwards the buffer's close as a `None`
        sentinel so the trainer unblocks on a clean stop OR a crash.
        """
        try:
            while True:
                episodes = await buffer.wait_for_batch_episodes()
                if episodes is None:  # buffer closed and drained
                    break
                with sl.log_trace_span("pack_batch"):
                    packed = await asyncio.to_thread(
                        self.batcher.pack_batch,
                        episodes,
                        dp_degree=self.trainer_dp_degree,
                    )
                # pop_consumed_episodes drops the consumed episodes and returns this batch's ride-out
                # metrics (the consumed rounds' rollout/gen metrics + the buffer's staleness/depth).
                ride_out_metrics = await buffer.pop_consumed_episodes(
                    packed.num_episodes_consumed
                )
                ready_batch = replace(packed, metrics=packed.metrics + ride_out_metrics)
                # bounded: blocks => at most max_queued_batches packed ahead
                await batch_queue.put(ready_batch)
        finally:
            # Unblock a trainer parked on `batch_queue.get()`. Non-blocking on purpose: at a clean stop
            # the trainer has already finished and the queue may hold an unconsumed packed batch
            # (full) — a blocking put there would deadlock the cancel (the trainer won't drain it).
            try:
                batch_queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

    async def _trainer_loop(
        self, batch_queue: asyncio.Queue, *, num_steps: int
    ) -> None:
        """Trainer stage / driver. Each step reads a packed batch, trains, and hands the new weights to
        the generators ASYNC so their load overlaps the next step. Phases read top-to-bottom: wait ->
        fwd/bwd -> divergence gate -> optim -> stage weights -> await prior pull (advance version) ->
        fire next pull -> emit. The `batch_queue.get` wait is the trainer-idle bubble
        (`perf/trainer_idle_ratio`).
        """
        # The previous step's generator weight-load runs in the background and overlaps this step.
        # We await it before publishing again, advancing `_generator_policy_version` to the version it
        # adopted. `_pending_pull` is initialized by the caller (`run`).
        for step in range(1, num_steps + 1):
            # propagate the step counter to actors
            sl.set_step(step)
            await self.trainer.sync_log_step.call(step)
            await self.generator_router.fanout("sync_log_step", step)
            t_step = time.perf_counter()

            # --- wait: pulling the packed batch IS the trainer-idle bubble ---
            t_wait = time.perf_counter()
            batch = await batch_queue.get()
            if batch is None:
                logger.info("Episode buffer closed and drained; stopping training")
                break
            wait_s = time.perf_counter() - t_wait

            # --- train: grad-accum microbatches ---
            # TODO(async): can't stream microbatches (interleave pack->train) — the loss is normalized
            #   by batch.num_global_valid_tokens (sum over ALL microbatches), needed before any fwd/bwd.
            #   To fix, grad has to be scaled later
            t_train = time.perf_counter()
            fwd_bwd = _reduce_microbatches(
                [
                    await self._forward_backward(
                        microbatch, batch.num_global_valid_tokens
                    )
                    for microbatch in batch.microbatches
                ]
            )
            if not math.isfinite(fwd_bwd["loss/mean"]):
                logger.error("Loss is NaN/Inf; training diverged")
                break

            with sl.log_trace_span("trainer_optim_step_call"):
                optim = self._get_rank_0_value(await self.trainer.optim_step.call())
            self._trainer_policy_version = optim.policy_version
            train_s = time.perf_counter() - t_train

            # --- weight sync: stage to CPU, await the prior pull,
            # then fire this pull so its load overlaps the NEXT step ---
            t_sync = time.perf_counter()
            with sl.log_trace_span("trainer_push_model_state_dict"):
                await self.trainer.push_model_state_dict.call(
                    version=optim.policy_version
                )
            # the GPU->CPU stage cost (on the critical path)
            push_s = time.perf_counter() - t_sync

            if self._pending_pull is not None:
                await self._pending_pull.task  # prior pull landed: the generator has adopted that version
                self._generator_policy_version = (
                    self._pending_pull.version
                )  # advance ON COMPLETION
            # Un-overlapped tail of the PRIOR pull: ~0 when the generator load hid behind this step,
            # large when the pull can't keep up (the signal that gates the deferred-push follow-up).
            pull_wait_s = time.perf_counter() - t_sync - push_s
            self._pending_pull = _PendingPull(
                task=asyncio.create_task(
                    self.generator_router.pull_model_state_dict(
                        policy_version=optim.policy_version
                    ),
                    name=f"pull_v{optim.policy_version}",
                ),
                version=optim.policy_version,
            )
            sync_s = time.perf_counter() - t_sync
            step_s = time.perf_counter() - t_step

            # --- emit (buffer/rollout/episode/packing + generator-snapshot metrics ride out with
            #     the batch; the generator stats no longer need a controller poll) ---
            self.metrics_processor.log(
                step=step,
                is_validation=False,
                metrics=_build_step_metrics(
                    batch=batch,
                    fwd_bwd=fwd_bwd,
                    optimizer=optim.metrics,
                    wait_s=wait_s,
                    train_s=train_s,
                    push_s=push_s,
                    pull_wait_s=pull_wait_s,
                    sync_s=sync_s,
                    step_s=step_s,
                ),
            )
        if self._pending_pull is not None:
            await self._pending_pull.task  # drain the last weight-load before returning
            self._generator_policy_version = self._pending_pull.version

    async def _forward_backward(
        self, microbatch, num_global_valid_tokens: int
    ) -> dict[str, float]:
        """One grad-accum microbatch forward/backward on the trainer; returns rank-0's metric dict."""
        with sl.log_trace_span("trainer_forward_backward_call"):
            return self._get_rank_0_value(
                await self.trainer.forward_backward.call(
                    microbatch, num_global_valid_tokens
                )
            )
