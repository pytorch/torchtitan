# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
RL trainer used for synchronous grpo training.
"""

import asyncio
import logging
import math
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

# must run before torch import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: F401  # force torch import after PYTORCH_CUDA_ALLOC_CONF is set
import torchstore as ts
from monarch.actor import ProcMesh
from monarch.spmd import setup_torch_elastic_env_async

from torchtitan.config import CompileConfig, Configurable
from torchtitan.experiments.rl.actors.generator import SamplingConfig, VLLMGenerator
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.batcher import Batcher
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
from torchtitan.experiments.rl.rollout_recorder import RolloutSampleRecorder
from torchtitan.experiments.rl.types import Episode
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols.model_spec import ModelSpec

logger = logging.getLogger(__name__)


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
        await trainer.train()
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

        num_groups_per_rollout_batch: int = 5
        """GRPO groups collected per rollout batch; a train step may collect several batches
        until the token target is met. Rollouts per batch = `num_groups_per_rollout_batch * group_size`."""
        # TODO(continuous-batching): this knob exists because we collect to a token budget
        # in discrete sync batches; async/continuous batching streams may change this logic

        group_size: int = 8
        """Sibling rollouts sampled per dataset row (the GRPO group). The generator
        is always called with `n=1`; prompts are pre-expanded by `group_size`."""

        num_validation_samples: int = 20
        """Number of held-out prompts scored greedily (temp=0, n=1) per validation pass."""

        validation_freq: int = 0
        """How often (in training steps) to run a mid-training validation pass. 0
        (the default) only validates once before and once after training; set it to,
        say, 5 to watch the eval metric move as training proceeds."""

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

            if self.trainer.debug.batch_invariant:
                if torch.version.hip is not None:
                    raise ValueError(
                        "batch_invariant mode is not supported on ROCm: the varlen "
                        "attention path cannot force num_splits=1 (rejected by ROCm), "
                        "so split-k reductions are non-deterministic."
                    )
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
        # Step to resume from: 0 for a fresh run, or the restored checkpoint
        # step when resuming. Set in setup_async once the trainer has loaded
        # any existing checkpoint.
        self.start_step = 0
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

    def _shard_episodes(self, episodes: list[Episode]) -> list[list[Episode]]:
        """Round-robin partition episodes across DP ranks."""
        return [
            [episodes[i] for i in range(rank, len(episodes), self.trainer_dp_degree)]
            for rank in range(self.trainer_dp_degree)
        ]

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
        trainer to generator. Must be called before :meth:`train`.

        Args:
            trainer_mesh: ProcMesh the trainer actor is spawned on.
            generator_meshes: ProcMesh objects the generator actors are spawned on.
        """
        # Thread pool for TokenEnv's asyncio.to_thread renderer calls — one worker per
        # concurrent rollout, capped by CPUs.
        max_concurrent_rollouts = max(
            self.config.num_groups_per_rollout_batch * self.config.group_size,
            self.config.num_validation_samples,
        )
        max_workers = max(1, min(max_concurrent_rollouts, os.cpu_count() or 1))
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
                    max_num_seqs=max(
                        config.num_groups_per_rollout_batch * config.group_size,
                        config.num_validation_samples,
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

        # Resume support: the trainer's __init__ already ran
        # CheckpointManager.load(), which (when a checkpoint exists in its
        # folder) restored the model, optimizer, LR scheduler, and
        # policy_version. Read that version back so the loop resumes at the right
        # step and the generators pull weights tagged with the matching version
        # (0 for a fresh run).
        self.start_step = self._get_rank_0_value(
            await self.trainer.get_policy_version.call()
        )
        if self.start_step > 0:
            logger.info("Resuming RL training from step %d", self.start_step)

        # Initial weight sync from trainer to generator
        with sl.log_trace_span("trainer_push_model_state_dict"):
            await self.trainer.push_model_state_dict.call()
        with sl.log_trace_span("generator_pull_model_state_dict"):
            await self.generator_router.pull_model_state_dict(
                policy_version=self.start_step
            )

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
        """Sample examples, run each group's rollouts concurrently, emit metrics.

        Hands each prompt to `Rollouter.run_group_rollouts` with a `GenerateFn` that runs one
        generation, and returns a `RolloutGroup`.

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
        # Validation ids and metrics prefix
        generation_metrics_prefix = (
            "validation_generator" if is_validation else "generator"
        )
        rollout_metrics_prefix = "validation" if is_validation else "rollout"

        group_id_prefix = "val/" if is_validation else ""

        # Pass a callable to the rollouter, so it stays decoupled from the generator router
        async def generate_fn(
            prompt_token_ids,
            *,
            request_id,
            routing_session_id=None,
            sampling_config=None,
        ):
            result = await self.generator_router.route(
                "generate",
                prompt_token_ids,
                request_id=request_id,
                sampling_config=sampling_config,
                metrics_prefix=generation_metrics_prefix,
                routing_ctx=RoutingContext(
                    estimated_cost=len(prompt_token_ids),
                    session_id=routing_session_id,
                ),
            )
            return self._get_rank_0_value(result)

        # Draw one dataset sample per group, then run every group's rollouts concurrently.
        # TODO: "sample" is too confusing. e.g. is this a training sample?
        # A rollout sample? Lets find a better name
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
                    generate_fn=generate_fn,
                    sample=sample,
                    group_id=f"{group_id_prefix}step={step}/group={group_offset + i}",
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
                    group_id_prefix,
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
        # all metrics calculation to the rollout loop.
        # TODO: we may also have some metrics at the Rollout level, i.e. not turn specific.
        # TODO: we also need a "logs" field, so that if there were errors/warnings
        # they can be made available for the rollout_logger
        metrics = [
            metric
            for group in rollout_groups
            for rollout in group.rollouts
            for rollout_turn in rollout.turns
            for metric in rollout_turn.metrics
        ]
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

        # Iteratate over `Rollouts` and produce one or more `Episode`. More than one `Episode`
        # be available if history of turn N is not a prefix of turn N+1, indicating history branching.
        branches_per_rollout: list[float] = []
        advantages_per_rollout: list[float] = []
        for group in rollout_groups:
            # Drop the whole group if no sibling has trainable tokens
            if all(
                not any(turn.completion_token_ids for turn in rollout.turns)
                for rollout in group.rollouts
            ):
                logger.warning(
                    "group %s has no trainable rollout; dropping the group",
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

        num_groups = len(rollout_groups)

        # TODO: drop groups with zero std
        zero_std_frac = (
            sum(1 for s in group_stds if s == 0.0) / num_groups if num_groups else 0.0
        )

        # TODO: better consolidate where rollout metrics are computed
        episode_metrics: list[m.Metric] = [
            m.Metric("advantage", m.SummaryStats.from_list(advantages_per_rollout)),
            m.Metric("rollout_reward/group_std", m.Mean.from_list(group_stds)),
            m.Metric("rollout_reward/group_std", m.Max.from_list(group_stds)),
            m.Metric("rollout_reward/zero_std_frac", m.NoReduce(zero_std_frac)),
            m.Metric(
                "rollout/branches_per_rollout", m.Mean.from_list(branches_per_rollout)
            ),
            m.Metric(
                "rollout/branches_per_rollout", m.Max.from_list(branches_per_rollout)
            ),
        ]

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

    async def train(self):
        num_steps = self.config.num_steps
        num_groups = self.config.num_groups_per_rollout_batch
        start_step = self.start_step  # 0 fresh, >0 when resuming a checkpoint
        logger.info(
            "Pre-training validation; then RL training steps %d..%d",
            start_step + 1,
            num_steps,
        )

        # Validation before training to compare before/after. On resume the
        # baseline is the restored policy at `start_step`.
        pre_validation_metrics = await self.validate(step=start_step)
        self.metrics_processor.log(
            step=start_step,
            metrics=pre_validation_metrics,
            is_validation=True,
        )
        pre_validation_agg = m.MetricsProcessor._aggregate_metrics(
            pre_validation_metrics
        )

        sl.log_trace_instant("training_start")

        for step in range(start_step + 1, num_steps + 1):
            sl.set_step(step)

            # Propagate the step counter to actors for structured logging.
            await self.trainer.sync_log_step.call(step)
            await self.generator_router.fanout("sync_log_step", step)

            t_step_start = time.perf_counter()

            # --- rollouts ---
            # Collect rollouts until total response tokens reach the
            # token budget. The Batcher then packs, truncates to
            # global_batch_size rows, and splits into microbatches.
            t_rollout_start = time.perf_counter()
            rollout_groups: list[RolloutGroup] = []
            rollout_metrics: list[m.Metric] = []
            collected_tokens = 0
            group_offset = 0

            # num_tokens_target (= global_batch_size * seq_len) is the stop
            # condition for collected tokens before a train step can proceed.
            # NOTE: consecutive _collect_rollouts calls have a barrier between them (each is awaited to
            # completion). This behavior will change in async mode.
            num_tokens_target = self.batcher.num_tokens_target(self.trainer_dp_degree)
            while collected_tokens < num_tokens_target:
                new_rollout_groups, new_metrics = await self._collect_rollouts(
                    is_validation=False,
                    num_groups=num_groups,
                    group_size=self.config.group_size,
                    sampling=self._sampling,
                    step=step,
                    group_offset=group_offset,
                )
                rollout_groups.extend(new_rollout_groups)
                rollout_metrics.extend(new_metrics)

                # Count the packed training tokens per rollout. Each turn carries
                # the full context + new completion, so we only need to count the last.
                # TODO: do it based on Episodes, not rollout, since 1 rollout -> N episodes.
                collected_tokens += sum(
                    len(rollout.turns[-1].prompt_token_ids)
                    + len(rollout.turns[-1].completion_token_ids)
                    - 1
                    for group in new_rollout_groups
                    for rollout in group.rollouts
                    if rollout.turns
                )
                group_offset += num_groups

            episodes, episode_metrics = self._build_episodes(rollout_groups)
            t_rollout_s = time.perf_counter() - t_rollout_start

            # record rollout to jsonl
            # TODO: also record the env input (e.g. AlphabetSortSample) so the dump carries the
            # targets/expected answer — today there's no ground truth to judge correctness against.
            self.rollout_recorder.record(
                step=step, is_validation=False, rollout_groups=rollout_groups
            )

            # --- train ---
            t_train_start = time.perf_counter()
            with sl.log_trace_span("batcher_batch"):
                (
                    microbatches,
                    num_global_valid_tokens,
                    packing_metrics,
                ) = self.batcher.batch(episodes, dp_degree=self.trainer_dp_degree)

            # Aggregate metrics across gradient-accumulation microbatches.
            # "/mean" and "/frac" metrics are pre-normalized by
            # num_global_valid_tokens, so summing reconstructs the global
            # value.  "/max" metrics take the max across microbatches.
            # TODO: make metrics processing a utility or resolve this in the trainer actor.
            fwd_bwd_metrics: dict[str, float] = {}
            for microbatch in microbatches:
                with sl.log_trace_span("trainer_forward_backward_call"):
                    mb_metrics = self._get_rank_0_value(
                        await self.trainer.forward_backward.call(
                            microbatch, num_global_valid_tokens
                        )
                    )
                    for k, v in mb_metrics.items():
                        if k not in fwd_bwd_metrics:
                            fwd_bwd_metrics[k] = v
                        elif k.endswith("/max"):
                            fwd_bwd_metrics[k] = max(fwd_bwd_metrics[k], v)
                        elif k.endswith(("/mean", "/frac", "_mean", "_frac")):
                            fwd_bwd_metrics[k] += v
            with sl.log_trace_span("trainer_optim_step_call"):
                optim_output = self._get_rank_0_value(
                    await self.trainer.optim_step.call()
                )
            trainer_policy_version = optim_output.policy_version
            optimizer_metrics = optim_output.metrics
            t_train_s = time.perf_counter() - t_train_start

            # --- weight sync ---
            # TODO: we should have `push_model_state_dict` return `trainer_policy_version`
            # instead of having `trainer.optim_step` return it

            # trainer push
            t_push_start = time.perf_counter()
            with sl.log_trace_span("trainer_push_model_state_dict"):
                await self.trainer.push_model_state_dict.call()
            t_weight_sync_push_s = time.perf_counter() - t_push_start

            # generator pull
            with sl.log_trace_span("generator_pull_model_state_dict"):
                await self.generator_router.pull_model_state_dict(
                    policy_version=trainer_policy_version
                )
            t_weight_sync_total_s = time.perf_counter() - t_push_start
            t_step_s = time.perf_counter() - t_step_start

            # --- Prepare metrics ---
            total_tokens = sum(len(episode.token_ids) for episode in episodes)

            step_metrics: list[m.Metric] = []

            step_metrics += rollout_metrics
            step_metrics += episode_metrics

            # Actor metrics are already globally reduced and aggregated
            # across microbatches; NoReduce passes them through.
            step_metrics += [
                m.Metric(k, m.NoReduce(v)) for k, v in fwd_bwd_metrics.items()
            ]
            step_metrics += [
                m.Metric(k, m.NoReduce(v)) for k, v in optimizer_metrics.items()
            ]
            step_metrics += packing_metrics

            # timing metrics
            for key, value in [
                ("timing/step", t_step_s),
                ("timing/rollout", t_rollout_s),
                ("timing/train", t_train_s),
                ("timing/weight_sync/push", t_weight_sync_push_s),
                ("timing/weight_sync/total", t_weight_sync_total_s),
            ]:
                step_metrics.append(m.Metric(key, m.NoReduce(value)))

            # TODO(perf-metrics): this is trainer tokens / WHOLE step — a goodput that folds in
            #   rollout + weight-sync idle (the trainer sits idle ~87% of the step). Split into
            #   per-component active throughput (total_tokens/t_train_s; generated_tokens/t_rollout_s)
            #   vs goodput (.../t_step_s); the active-vs-goodput gap is the sync idle bubble. The
            #   rollout's generated_tokens must sum ALL collected rollouts, not just the kept batch.
            step_metrics.append(
                m.Metric("perf/tokens_per_second", m.NoReduce(total_tokens / t_step_s))
            )

            self.metrics_processor.log(
                step=step, metrics=step_metrics, is_validation=False
            )

            # break if diverged
            if not math.isfinite(fwd_bwd_metrics["loss/mean"]):
                logger.error("Loss is NaN/Inf; training diverged")
                break

            # --- checkpoint ---
            # Persist full training state (model + optimizer + LR scheduler +
            # policy_version) so a preempted run can resume. The trainer's
            # CheckpointManager only writes on its configured interval and on
            # the final step; other steps are a no-op. Saved after the
            # divergence check so a NaN step is not checkpointed.
            with sl.log_trace_span("trainer_save_checkpoint"):
                await self.trainer.save_checkpoint.call(
                    step, last_step=(step == num_steps)
                )

            # --- periodic validation ---
            # TODO(async): validation is generation-only, so overlap it with the next
            # step's training instead of blocking here.
            if (
                self.config.validation_freq
                and step % self.config.validation_freq == 0
                and step != num_steps
            ):
                periodic_validation_metrics = await self.validate(step=step)
                self.metrics_processor.log(
                    step=step,
                    metrics=periodic_validation_metrics,
                    is_validation=True,
                )

        post_validation_metrics = await self.validate(step=num_steps)
        self.metrics_processor.log(
            step=num_steps,
            metrics=post_validation_metrics,
            is_validation=True,
        )
        post_validation_agg = m.MetricsProcessor._aggregate_metrics(
            post_validation_metrics
        )

        # Side-by-side pre/post summary so the before/after improvement is
        # visible without scrolling back through the train loop.
        reward_keys = sorted(
            k
            for k in set(pre_validation_agg) | set(post_validation_agg)
            if "reward" in k
        )
        logger.info("=" * 60)
        logger.info("Validation summary (pre / post):")
        for key in reward_keys:
            pre = pre_validation_agg.get(key, float("nan"))
            post = post_validation_agg.get(key, float("nan"))
            logger.info(f"  {key}:  {pre:+.3f}  /  {post:+.3f}")
        logger.info("=" * 60)
