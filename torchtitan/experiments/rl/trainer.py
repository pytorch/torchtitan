# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Async GRPO RL trainer: a rollout producer and a trainer consumer over a shared episode buffer.

    examples (sampled off the event loop)
      _prefetch_examples --> example_queue (maxsize = num workers)
                               |
      _rollout_worker x N : run group -> score -> episodes        (producer)
                               | buffer.put(episodes, metrics, train_version)
                               v
      EpisodeBuffer  [staleness drop (oldest per-token version) + depth backpressure
                      + one-batch peel + pack off event loop]
                               | get_batch() -> PackedEpisodeBatch | None
                               v
      _consume_and_train (driver) : wait (idle bubble) -> fwd/bwd -> divergence gate
                                    -> optim -> weight sync (push -> pull, hotswap) -> advance version

`train` starts the producer (`_run_rollout_producer`) and drives the consumer (`_consume_and_train`);
the consumer's `get_batch` wait is the trainer-idle bubble (`controller/trainer_idle_time_ratio`).
"""

import asyncio
import functools
import logging
import math
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

# must run before torch import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torchstore as ts
from monarch.actor import ProcMesh
from monarch.spmd import setup_torch_elastic_env_async

from torchtitan.config import CompileConfig, Configurable
from torchtitan.experiments.rl.actors.generator import SamplingConfig, VLLMGenerator
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.batcher import Batcher
from torchtitan.experiments.rl.episode_buffer import EpisodeBuffer
from torchtitan.experiments.rl.generator_router import GeneratorRouter, RoutingContext
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
from torchtitan.experiments.rl.types import Episode, TrainingBatch
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols.model_spec import ModelSpec

logger = logging.getLogger(__name__)


class GRPOLoss(Configurable):
    """Per-token clipped surrogate loss for GRPO.

    Computes the PPO-style clipped objective at the token level::

        ratio_t = exp(policy_logprob_t - ref_logprob_t)     # π_θ / π_old
        clipped_t = clamp(ratio_t, 1 - ε, 1 + ε)
        loss_t = -min(ratio_t * A_t, clipped_t * A_t)

    The final scalar loss is the sum of per-token losses over loss
    positions (where ``loss_mask == 1``), divided by
    ``num_global_valid_tokens`` (total loss positions across all
    microbatches and DP ranks).  This normalization ensures that
    gradient accumulation across microbatches produces the same
    result as a single large-batch forward pass.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        clip_eps: float = 0.2
        """PPO clipping epsilon for the probability ratio."""

    def __init__(self, config: Config):
        self.clip_eps = config.clip_eps

    def __call__(
        self,
        policy_logprobs: torch.Tensor,
        generator_logprobs: torch.Tensor,
        loss_mask: torch.Tensor,
        advantages: torch.Tensor,
        num_global_valid_tokens: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute per-token GRPO clipped surrogate loss.

        Args:
            policy_logprobs: [B, L] log π_θ(a_t | s_t) from the current policy.
            generator_logprobs: [B, L] log π_old(a_t | s_t) from the sampling policy.
            loss_mask: [B, L] bool mask; True for response tokens.
            advantages: [B, L] per-token advantages (0.0 for prompt/padding).
            num_global_valid_tokens: total response tokens across all microbatches
                and DP ranks; used as the loss denominator so gradient
                accumulation is equivalent to a single large-batch step.

        Returns:
            (loss, metrics) where loss is a scalar tensor and metrics is a
            dict of scalar tensors pre-normalized for SUM reduction across
            DP ranks.
        """
        # Per-token importance sampling ratio: π_θ / π_old
        log_ratio = policy_logprobs - generator_logprobs

        # TODO: debug why vLLM+cudagraph emits nan generator logprobs
        log_ratio = torch.nan_to_num(log_ratio, nan=0.0)
        log_ratio = torch.clamp(log_ratio, -20.0, 20.0)
        ratio = torch.exp(log_ratio)

        clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
        token_pg_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

        masked_loss = token_pg_loss * loss_mask
        loss_denominator = max(num_global_valid_tokens, 1)
        loss = masked_loss.sum() / loss_denominator

        with torch.no_grad():
            masked_ratio = ratio * loss_mask
            metrics = {
                "loss/mean": loss.detach(),
                "loss/ratio_mean": masked_ratio.sum() / loss_denominator,
                "loss/ratio_clipped_frac": (
                    (torch.abs(ratio - clipped_ratio) > 1e-6).float() * loss_mask
                ).sum()
                / loss_denominator,
                # Fraction of response tokens whose generator (vLLM) logprob is nan.
                "loss/generator_logprob_nan_frac": (
                    (~torch.isfinite(generator_logprobs)).float() * loss_mask
                ).sum()
                / loss_denominator,
            }

        return loss, metrics


def _log_samples(rollout_groups: list[RolloutGroup]) -> None:
    """Log the first scored, trainable rollout per group for debugging."""
    for group in rollout_groups:
        rollout = next(
            (
                r
                for r in group.rollouts
                if r.reward is not None and r.turns and r.turns[0].completion_token_ids
            ),
            None,
        )
        if rollout is None:
            continue
        logger.info("  [%s] reward=%+.1f", group.group_id, rollout.reward)
        message = rollout.turns[-1].completion_message
        text = (message.get("content") or "") if message else ""
        logger.info("       A: %s", text[:300].replace("\n", " ").strip())


def _generation_metrics(groups: list[RolloutGroup]) -> list[m.Metric]:
    """Flatten every turn's per-generation metrics (latencies, output tokens) across the groups.

    Shared by the training producer (`_rollout_worker`, one group) and validation
    (`_collect_rollouts`, a round of groups).
    """
    return [
        metric
        for group in groups
        for rollout in group.rollouts
        for rollout_turn in rollout.turns
        for metric in rollout_turn.metrics
    ]


@dataclass(frozen=True, slots=True)
class _WeightSyncTimings:
    """Wall-clock for one trainer->generator weight hotswap (push, then pull)."""

    push_s: float
    total_s: float

    @property
    def pull_s(self) -> float:
        # The generator keeps sampling during the pull; pull = total - push.
        return self.total_s - self.push_s


@dataclass(frozen=True, slots=True)
class _TrainStepTimings:
    """Wall-clock timings for one async train step."""

    step_s: float
    get_batch_s: float  # the trainer-idle bubble (waiting on the buffer)
    train_s: float
    weight_sync: _WeightSyncTimings


def _build_train_step_metrics(
    *,
    buffer_metrics: list[m.Metric],
    fwd_bwd_metrics: dict[str, float],
    optimizer_metrics: dict[str, float],
    num_global_valid_tokens: int,
    timings: _TrainStepTimings,
) -> list[m.Metric]:
    """All metrics for one async train step, in one place (pure: no I/O, unit-testable).

    `buffer_metrics` already carry the rollout/episode/staleness/depth signals that rode out with
    the batch; this adds fwd/bwd, optimizer, timing, and the derived diagnostic ratios.

    `perf/trainer_active_tokens_per_second` divides by TRAIN time, while goodput
    (`perf/tokens_per_second`) divides by the whole STEP, so their gap is the trainer-idle bubble.

    Example:

        t = _TrainStepTimings(step_s=10.0, get_batch_s=8.0, train_s=1.0,
                              weight_sync=_WeightSyncTimings(push_s=0.4, total_s=1.0))
        # -> controller/trainer_idle_time_ratio = 0.8, timing/weight_sync_overhead_ratio = 0.1,
        #    perf/tokens_per_second = tokens/10, perf/trainer_active_tokens_per_second = tokens/1
    """
    # TODO(observability): generator-side KV-cache usage + preemption counts would separate a
    #   saturated generator from an under-provisioned one when idle_ratio is high.
    step_s = timings.step_s
    weight_sync = timings.weight_sync
    metrics: list[m.Metric] = list(buffer_metrics)
    metrics += [
        m.Metric(key, m.NoReduce(value)) for key, value in fwd_bwd_metrics.items()
    ]
    metrics += [
        m.Metric(key, m.NoReduce(value)) for key, value in optimizer_metrics.items()
    ]
    derived = [
        ("timing/step", step_s),
        ("timing/get_batch", timings.get_batch_s),
        ("timing/train", timings.train_s),
        ("timing/weight_sync/push", weight_sync.push_s),
        ("timing/weight_sync/pull", weight_sync.pull_s),
        ("timing/weight_sync/total", weight_sync.total_s),
        (
            "timing/weight_sync_overhead_ratio",
            weight_sync.total_s / step_s if step_s else 0.0,
        ),
        (
            "controller/trainer_idle_time_ratio",
            timings.get_batch_s / step_s if step_s else 0.0,
        ),
        ("perf/tokens_per_second", num_global_valid_tokens / step_s if step_s else 0.0),
        (
            "perf/trainer_active_tokens_per_second",
            num_global_valid_tokens / timings.train_s if timings.train_s else 0.0,
        ),
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
        """GRPO groups the producer keeps generating concurrently.
        Rollouts in flight = `num_groups_per_rollout_batch * group_size`."""

        group_size: int = 8
        """Sibling rollouts sampled per dataset row (the GRPO group). The generator
        is always called with `n=1`; prompts are pre-expanded by `group_size`."""

        num_validation_samples: int = 20
        """Number of held-out prompts scored greedily (temp=0, n=1) per validation pass."""

        max_offpolicy_steps: int = 1
        """Off-policy bound: drop an episode whose oldest token is more than this many versions
        behind the trainer. 0 = strict on-policy. See `EpisodeBuffer`."""

        max_buffered_batches: int = 2
        """Depth bound: batches the rollout producer may bank ahead before backpressure.
        Set >= `max_offpolicy_steps + 1` to use the full staleness budget."""

        rollouter: Rollouter.Config
        """The rollouter: its datasets, envs, and rubric."""
        # TODO: support multiple rollouters for data mixing.

        renderer: RendererConfig
        """Message-to-token renderer config."""

        log_samples: bool = False
        """Log first completion per rollout during training and validation."""

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

        generator_router: GeneratorRouter.Config = field(
            default_factory=lambda: GeneratorRouter.Config(hot_swap=True)
        )
        """Router over one or more generator actors. `hot_swap=True`: a weight pull doesn't drain
        in-flight generation (the async producer keeps generating across the swap)."""

        metrics: m.MetricsProcessor.Config = field(
            default_factory=m.MetricsProcessor.Config
        )

        def __post_init__(self):
            if self.generator.checkpoint.enable:
                raise ValueError(
                    "Generator checkpoint must be disabled in the RL loop "
                    "(weights are synced from the trainer via TorchStore). "
                    "Set generator.checkpoint.enable=False."
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
        self.trainer = None
        self.generator_router: GeneratorRouter | None = None
        self._proc_meshes = []
        self.metrics_processor: m.MetricsProcessor = config.metrics.build(
            log_dir=config.dump_folder,
            job_config=config.to_dict(),
        )
        self.renderer = config.renderer.build(tokenizer_path=config.hf_assets_path)

        # Renderer stop tokens are injected into the generator at spawn
        self._stop_token_ids = list(self.renderer.get_stop_token_ids())
        self._sampling = config.generator.sampling
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

    @sl.log_trace_span("_generate")
    async def _generate(
        self,
        prompt_token_ids,
        *,
        request_id,
        sampling_config=None,
        metrics_prefix,
    ):
        """Await one completion via the generator router (rank 0 value; None on followers).

        `metrics_prefix` namespaces the per-generation metrics (e.g. `"generator"` for training,
        `"validation_generator"` for validation). Bind it with `functools.partial` to hand a
        `GenerateFn` to the rollouter; the rollouter passes `request_id` per call.
        """
        result = await self.generator_router.route(
            "generate",
            prompt_token_ids,
            request_id=request_id,
            sampling_config=sampling_config,
            metrics_prefix=metrics_prefix,
            routing_ctx=RoutingContext(estimated_cost=len(prompt_token_ids)),
        )
        return self._get_rank_0_value(result)

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

        # Fail fast on a seq-len mismatch: the batcher's `pack()` silently drops any episode longer
        # than `seq_len`, which then under-fills the batch and crashes the trainer mid-run. A rollout
        # can produce episodes up to the generator's context (`model.max_seq_len`) unless it self-caps
        # via `max_rollout_tokens`, so the packed row must be able to hold the longest possible episode.
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
                    stop_token_ids=self._stop_token_ids,
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

        # Initial weight sync from trainer to generator (engine is idle, so no drain needed)
        with sl.log_trace_span("trainer_push_model_state_dict"):
            await self.trainer.push_model_state_dict.call()
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
        """Sample examples, run each group's rollouts concurrently, emit metrics.

        Hands each prompt to `Rollouter.run_group_rollouts` with a `GenerateFn` bound to the
        generator, which returns a RollooutGroup.

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
        generate = functools.partial(
            self._generate, metrics_prefix=generation_metrics_prefix
        )

        # Validation ids live in their own namespace so they never collide with a train
        # request still in flight in the long-lived continuous-batching engine.
        group_prefix = "val/" if is_validation else ""

        # Sample one example per group, then run every group's rollouts concurrently.
        # TODO: "sample" is too confusing. e.g. is this a training sample?
        # A rollout sample? Lets find a better name
        examples = [
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
                    sample=example,
                    group_id=f"{group_prefix}step={step}/group={group_offset + i}",
                    group_size=group_size,
                    sampling=sampling,
                    renderer=self.renderer,
                )
                for i, example in enumerate(examples)
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
        """Build train episodes and GRPO advantages from scored rollout groups.

        Centers each group's rewards by its mean, skips rollouts without
        training tokens, and emits reward/advantage metrics.

        Args:
            rollout_groups: Scored rollout groups from one collection round.

        Returns:
            Train episodes plus episode-level metrics.
        """
        # Mean-baseline advantage per group
        episodes: list[Episode] = []
        group_stds: list[float] = []
        # Episodes a rollout packs into: 1 when the trajectory shares one prefix,
        # >1 when the env edited history mid-rollout and `rollout_to_episodes` branched.
        branches_per_rollout: list[float] = []
        advantages_per_rollout: list[float] = []
        for group in rollout_groups:
            # Drop the whole group if any sibling has no trainable tokens; we need at
            # least one turn with assistant tokens per rollout to build an episode (and
            # the group must stay intact for mean-baseline advantage centering).
            if any(
                not any(turn.completion_token_ids for turn in rollout.turns)
                for rollout in group.rollouts
            ):
                logger.warning(
                    "group %s has an untrainable rollout; dropping the group",
                    group.group_id,
                )
                continue

            rewards = [rollout.reward for rollout in group.rollouts]
            group_mean = sum(rewards) / len(rewards)
            group_stds.append(statistics.pstdev(rewards))

            # Center the advantage per rollout; each rollout packs into one or more episodes.
            for rollout in group.rollouts:
                rollout.advantage = rollout.reward - group_mean
                rollout_episodes = rollout_to_episodes(rollout)
                episodes.extend(rollout_episodes)
                branches_per_rollout.append(float(len(rollout_episodes)))
                advantages_per_rollout.append(rollout.advantage)

        # Fraction of groups whose siblings all scored the same (std 0 -> no GRPO signal). A
        # per-group 0/1 indicator under Mean, not NoReduce: the async producer banks many groups
        # between train steps, so the metric must aggregate across them (NoReduce takes one entry).
        zero_std = [1.0 if std == 0.0 else 0.0 for std in group_stds]
        # Rollout reward rides `rollout_reward` (see `prepare_rollout_metrics`); here we emit
        # only the GRPO-specific signals — advantage, per-group reward std, zero-std fraction.
        episode_metrics: list[m.Metric] = [
            m.Metric("advantage", m.SummaryStats.from_list(advantages_per_rollout)),
            m.Metric("reward/group_std", m.Mean.from_list(group_stds)),
            m.Metric("reward/group_std", m.Max.from_list(group_stds)),
            m.Metric("reward/zero_std_frac", m.Mean.from_list(zero_std)),
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

        if self.config.log_samples:
            _log_samples(rollout_groups)
        self.rollout_recorder.record(
            step=step, is_validation=True, rollout_groups=rollout_groups
        )

        validation_metrics.append(
            m.Metric("validation/num_samples", m.NoReduce(float(len(rollouts))))
        )

        t_validate_s = time.perf_counter() - t_validate_start
        validation_metrics.append(m.Metric("timing/validate", m.NoReduce(t_validate_s)))
        return validation_metrics

    async def train(self) -> None:
        num_steps = self.config.num_steps
        logger.info(
            f"Pre-training validation; then {num_steps} steps of async RL training"
        )

        # Validate before and after so the reward improvement is visible.
        pre_validation_agg = self._log_validation(
            step=0, metrics=await self.validate(step=0)
        )
        sl.log_trace_instant("training_start")

        # The producer banks fresh episodes; the consumer (driver) trains whole batches and
        # hotswaps weights, advancing `_train_version`. The producer supervisor closes the buffer
        # when it stops (crash OR cancel), so the consumer's `get_batch` always unblocks.
        buffer = EpisodeBuffer(
            batcher=self.batcher,
            dp_degree=self.trainer_dp_degree,
            max_offpolicy_steps=self.config.max_offpolicy_steps,
            max_buffered_batches=self.config.max_buffered_batches,
        )
        self._train_version = 0
        producer = asyncio.create_task(
            self._run_rollout_producer(buffer), name="rollout_producer"
        )
        try:
            await self._consume_and_train(buffer, num_steps=num_steps)
        finally:
            producer.cancel()
            try:
                await producer  # re-raises a producer crash; CancelledError on a clean stop
            except asyncio.CancelledError:
                pass

        # TODO(async): the engine may still hold abandoned train requests when post-validation
        #   starts; they resolve harmlessly (validation uses a `val/` id namespace) but share the
        #   engine. Drain them first if it skews validation timing.
        post_validation_agg = self._log_validation(
            step=num_steps, metrics=await self.validate(step=num_steps)
        )
        self._log_validation_summary(pre_validation_agg, post_validation_agg)

    def _log_validation(
        self, *, step: int, metrics: list[m.Metric]
    ) -> dict[str, float]:
        """Log one validation pass; return its aggregated values for the pre/post summary."""
        self.metrics_processor.log(step=step, metrics=metrics, is_validation=True)
        return m.MetricsProcessor._aggregate_metrics(metrics)

    def _log_validation_summary(
        self, pre_agg: dict[str, float], post_agg: dict[str, float]
    ) -> None:
        """Side-by-side pre/post reward summary, visible without scrolling back through the loop."""
        reward_keys = sorted(
            key for key in set(pre_agg) | set(post_agg) if "reward" in key
        )
        logger.info("=" * 60)
        logger.info("Validation summary (pre / post):")
        for key in reward_keys:
            pre = pre_agg.get(key, float("nan"))
            post = post_agg.get(key, float("nan"))
            logger.info(f"  {key}:  {pre:+.3f}  /  {post:+.3f}")
        logger.info("=" * 60)

    async def _run_rollout_producer(self, buffer: EpisodeBuffer) -> None:
        """Run the rollout producer; close the buffer on the way out (crash OR cancel) so a
        consumer parked in `get_batch` always unblocks.

        Example: a producer crash propagates here, `buffer.close()` runs in `finally`, `get_batch`
        returns None (consumer stops), and `train()`'s `await producer` re-raises the real error.
        """
        try:
            await self._produce_rollouts(buffer)
        finally:
            await buffer.close()

    async def _produce_rollouts(self, buffer: EpisodeBuffer) -> None:
        """Producer: a data-prefetch coroutine keeps `num_groups_per_rollout_batch` rollout
        workers fed with ready examples; each worker runs a group rollout, scores it into
        episodes, and `put`s them in the buffer (`put` backpressures at the depth bound).

        The prefetch queue decouples data sampling from generation, and `get_batch`'s threaded
        packing keeps the consumer's batching off the event loop. Runs until `train()` cancels it.
        """
        num_workers = self.config.num_groups_per_rollout_batch
        generate = functools.partial(self._generate, metrics_prefix="generator")

        # DATA stage feeds ROLLOUT stage; one spare example queued per worker.
        example_queue: asyncio.Queue = asyncio.Queue(maxsize=num_workers)
        tasks = [
            asyncio.create_task(self._prefetch_examples(example_queue)),
            *(
                asyncio.create_task(
                    self._rollout_worker(
                        worker_id=worker_id,
                        example_queue=example_queue,
                        buffer=buffer,
                        generate=generate,
                    )
                )
                for worker_id in range(num_workers)
            ),
        ]
        try:
            # Workers run until cancelled; surface the first crash (gather leaves siblings running,
            # so the finally cancels them).
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            # Let the cancellations unwind before returning (don't leave orphan workers running).
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _prefetch_examples(self, example_queue: asyncio.Queue) -> None:
        """DATA stage: sample training examples off the event loop and queue them, so a rollout
        worker always has a ready example to start on (the queue depth is the prefetch)."""
        while True:
            example = await asyncio.to_thread(self._get_training_sample)
            await example_queue.put(example)

    @sl.log_trace_span("_get_training_sample")
    def _get_training_sample(self):
        return self._rollouter.get_training_sample()

    @sl.log_trace_span("_run_group_rollouts")
    async def _run_group_rollouts(
        self,
        *,
        generate: GenerateFn,
        example,
        group_id: str,
    ) -> RolloutGroup:
        return await self._rollouter.run_group_rollouts(
            generate_fn=generate,
            sample=example,
            group_id=group_id,
            group_size=self.config.group_size,
            sampling=self._sampling,
            renderer=self.renderer,
        )

    async def _rollout_worker(
        self,
        *,
        worker_id: int,
        example_queue: asyncio.Queue,
        buffer: EpisodeBuffer,
        generate: GenerateFn,
    ) -> None:
        """ROLLOUT stage: pull a ready example, run its group rollout, score it into episodes,
        and `put` them in the buffer. A group that raises is logged + dropped (its
        `group_failures` rides out with the next batch); the worker keeps going."""
        group_count = 0
        while True:
            example = await example_queue.get()
            group_id = (
                f"step={self._train_version + 1}/group=w{worker_id}_{group_count}"
            )
            group_count += 1
            try:
                group = await self._run_group_rollouts(
                    generate=generate,
                    example=example,
                    group_id=group_id,
                )
            except Exception:
                logger.exception("rollout group %s failed; dropping", group_id)
                await buffer.put(
                    [],
                    [m.Metric("rollout/group_failures", m.Sum(1.0))],
                    train_version=self._train_version,
                )
                continue

            episodes, episode_metrics = self._build_episodes([group])
            rollout_metrics = prepare_rollout_metrics("rollout", group.rollouts)
            generation_metrics = _generation_metrics([group])
            if self.config.log_samples:
                _log_samples([group])
            self.rollout_recorder.record(
                step=self._train_version + 1,
                is_validation=False,
                rollout_groups=[group],
            )
            # Live read: the consumer advances `_train_version` after each weight sync while this
            # worker awaits, so admission staleness is measured against the version current when
            # the rollout finished; do not snapshot it earlier in the loop.
            await buffer.put(
                episodes,
                rollout_metrics + generation_metrics + episode_metrics,
                train_version=self._train_version,
            )

    async def _consume_and_train(
        self, buffer: EpisodeBuffer, *, num_steps: int
    ) -> None:
        """Consumer/driver: each step wait one packed batch, train, hotswap weights, advance the
        policy version. The `get_batch` wait is the trainer-idle bubble.
        """
        for step in range(1, num_steps + 1):
            sl.set_step(step)
            # Propagate the step counter to actors for structured logging.
            await self.trainer.sync_log_step.call(step)
            await self.generator_router.fanout("sync_log_step", step)
            t_step_start = time.perf_counter()

            # --- wait: this get_batch wait IS the trainer-idle bubble ---
            t_get_start = time.perf_counter()
            batch = await buffer.get_batch(train_version=self._train_version)
            if batch is None:
                logger.info("Episode buffer closed and drained; stopping training")
                break
            get_batch_s = time.perf_counter() - t_get_start

            # --- train: grad-accum microbatches; a NaN step must NEVER reach optim or push ---
            t_train_start = time.perf_counter()
            fwd_bwd_metrics = await self._run_microbatches(
                batch.microbatches, batch.num_global_valid_tokens
            )
            if not math.isfinite(fwd_bwd_metrics["loss/mean"]):
                logger.error("Loss is NaN/Inf; training diverged")
                break
            with sl.log_trace_span("trainer_optim_step_call"):
                optim_output = self._get_rank_0_value(
                    await self.trainer.optim_step.call()
                )
            train_s = time.perf_counter() - t_train_start

            # --- weight sync (push -> pull, hotswap), then advance to the published version ---
            weight_sync = await self._sync_generator_weights(
                optim_output.policy_version
            )
            self._train_version = optim_output.policy_version
            step_s = time.perf_counter() - t_step_start

            # --- emit (rollout/episode/staleness/depth metrics ride out with the batch) ---
            timings = _TrainStepTimings(
                step_s=step_s,
                get_batch_s=get_batch_s,
                train_s=train_s,
                weight_sync=weight_sync,
            )
            self.metrics_processor.log(
                step=step,
                is_validation=False,
                metrics=_build_train_step_metrics(
                    buffer_metrics=batch.metrics,
                    fwd_bwd_metrics=fwd_bwd_metrics,
                    optimizer_metrics=optim_output.metrics,
                    num_global_valid_tokens=batch.num_global_valid_tokens,
                    timings=timings,
                ),
            )

    async def _run_microbatches(
        self, microbatches: list[list[TrainingBatch]], num_global_valid_tokens: int
    ) -> dict[str, float]:
        """Forward/backward each grad-accum microbatch and reduce its metrics over the accumulation.

        "/mean" and "/frac" keys are pre-normalized by num_global_valid_tokens, so summing them
        across microbatches reconstructs the global value; "/max" keys take the max.
        """
        fwd_bwd_metrics: dict[str, float] = {}
        for microbatch in microbatches:
            with sl.log_trace_span("trainer_forward_backward_call"):
                mb_metrics = self._get_rank_0_value(
                    await self.trainer.forward_backward.call(
                        microbatch, num_global_valid_tokens
                    )
                )
            for key, value in mb_metrics.items():
                if key not in fwd_bwd_metrics:
                    fwd_bwd_metrics[key] = value
                elif key.endswith("/max"):
                    fwd_bwd_metrics[key] = max(fwd_bwd_metrics[key], value)
                elif key.endswith(("/mean", "/frac")):
                    fwd_bwd_metrics[key] += value
        return fwd_bwd_metrics

    async def _sync_generator_weights(self, policy_version: int) -> _WeightSyncTimings:
        """Push trainer weights, then pull them into the generator (hotswap).

        Push-before-pull is load-bearing: the generator keeps sampling on the old weights during
        the push and swaps to `policy_version` on pull. From here the producer samples at, and the
        buffer measures staleness against, the new version.
        """
        # TODO(perf): trainer awaits the pull before the next step; overlapping it (start step k+1
        #   during the pull) needs versioned weight keys or double-buffered source weights to be safe.
        t_push_start = time.perf_counter()
        with sl.log_trace_span("trainer_push_model_state_dict"):
            await self.trainer.push_model_state_dict.call()
        push_s = time.perf_counter() - t_push_start
        with sl.log_trace_span("generator_pull_model_state_dict"):
            await self.generator_router.pull_model_state_dict(
                policy_version=policy_version
            )
        return _WeightSyncTimings(
            push_s=push_s, total_s=time.perf_counter() - t_push_start
        )
