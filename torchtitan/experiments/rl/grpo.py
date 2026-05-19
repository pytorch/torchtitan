# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Async off-policy GRPO controller for torchtitan + monarch + vLLM.

Two long-lived asyncio tasks share a :class:`ReplayBuffer` actor:

- ``continuous_rollouts`` — N copies, each iterating dataset rows,
  driving a group of envs through ``do_rollout_group``, converting
  rollouts to :class:`ReplaySample`s, and feeding the buffer.
- ``continuous_training`` — one copy, pulling batches from the buffer,
  calling ``trainer.forward_backward``, ``trainer.optim_step``, pushing
  weights to torchstore, signalling the generator to pull.

The generator drains in-flight requests inside its own
``pull_model_state_dict`` endpoint; the controller doesn't synchronise
rollouts against the swap. In-flight rollouts finish on the old
weights and land in the buffer; the buffer's ``max_policy_age`` evicts
them before they reach the trainer.

Validation is a one-shot drain outside the pipeline (greedy sampling,
no buffer).

Command::

    MONARCH_RDMA_DISABLE_IBVERBS=1 MONARCH_RDMA_ALLOW_TCP_FALLBACK=1 \\
        python torchtitan/experiments/rl/grpo.py \\
            --module rl --config rl_grpo_qwen3_0_6b
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

# Must run before torch is imported (vLLM cudaMalloc fragmentation).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torchstore as ts
from monarch.actor import this_host
from monarch.spmd import setup_torch_elastic_env_async

from torchtitan.config import (
    CompileConfig,
    ConfigManager,
    Configurable,
    ParallelismConfig,
)
from torchtitan.experiments.rl.actors.generator import (
    GenerateOutput,
    SamplingConfig,
    VLLMGenerator,
)
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.envs.token_env import TokenEnvConfig
from torchtitan.experiments.rl.envs.types import (
    EnvBuilder,
    EnvDataset,
    EnvExample,
    MessageEnv,
)
from torchtitan.experiments.rl.metrics import (
    aggregate_rewards,
    dump_rollouts_jsonl,
    format_validation,
    log_first_sample,
)
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.replay import (
    BufferClosedError,
    compute_advantages,
    ReplayBuffer,
    rollout_to_replay_samples,
)
from torchtitan.experiments.rl.rollouts import do_rollout_group
from torchtitan.experiments.rl.types import ReplaySample, RolloutOutput, TrainBatch
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols.model_spec import ModelSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GRPO loss — per-token, mask-aware, DAPO global-token-mean normalization
# ---------------------------------------------------------------------------


class GRPOLoss(Configurable):
    """Clipped GRPO surrogate loss with per-token importance weighting.

    Operates on shifted, per-token tensors (``[1, T - 1]``). The mask
    selects loss positions; the loss is summed over masked tokens and
    divided by the global-across-DP-ranks mask sum (DAPO normalization).
    Importance weighting uses the per-token ratio
    ``exp(policy_logprob - behavior_logprob)`` clipped to
    ``[1 - clip_eps, 1 + clip_eps]``.

    Example metrics emitted (all SUM-reduced across DP ranks):
        loss/mean              -- the scalar loss
        loss/ratio/mean        -- mean ratio across loss tokens
        loss/ratio/clipped_frac -- fraction of loss tokens clipped
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        clip_eps: float = 0.2
        """PPO/GRPO probability-ratio clip epsilon."""

    def __init__(self, config: Config) -> None:
        self.clip_eps = config.clip_eps

    def __call__(
        self,
        *,
        policy_logprobs: torch.Tensor,  # [1, T - 1]
        behavior_logprobs: torch.Tensor,  # [1, T - 1]
        advantages_per_token: torch.Tensor,  # [1, T - 1]
        loss_mask: torch.Tensor,  # [1, T - 1]
        num_global_valid_tokens: torch.Tensor,  # scalar
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ratio = torch.exp(policy_logprobs - behavior_logprobs)
        clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
        per_token_pg = -torch.minimum(
            ratio * advantages_per_token,
            clipped * advantages_per_token,
        )
        masked = per_token_pg * loss_mask
        loss = masked.sum() / num_global_valid_tokens

        with torch.no_grad():
            ratio_masked = ratio * loss_mask
            clipped_frac = (
                ((ratio - clipped).abs() > 1e-6).float() * loss_mask
            ).sum() / num_global_valid_tokens
            metrics = {
                "loss/mean": loss.detach(),
                "loss/ratio/mean": ratio_masked.sum() / num_global_valid_tokens,
                "loss/ratio/clipped_frac": clipped_frac,
            }
        return loss, metrics


# ---------------------------------------------------------------------------
# Provisioner — non-overlapping GPU ranges for trainer + generator meshes
# ---------------------------------------------------------------------------


class Provisioner:
    """Allocates non-overlapping GPU ranges for Monarch proc meshes.

    Trainer and generator run on disjoint GPU meshes (e.g. trainer on
    0..3, generator on 4..7 single-host). ``allocate(n)`` returns a
    bootstrap callable that sets ``CUDA_VISIBLE_DEVICES`` before CUDA
    initializes in the spawned process.
    """

    def __init__(self, total_gpus: int = 8) -> None:
        self.total_gpus = total_gpus
        self.next_gpu = 0

    @property
    def available(self) -> int:
        return self.total_gpus - self.next_gpu

    def allocate(self, num_gpus: int) -> Callable[[], None]:
        if num_gpus > self.available:
            raise RuntimeError(
                f"Requested {num_gpus} GPUs but only {self.available} available "
                f"(total={self.total_gpus}, allocated={self.next_gpu})"
            )
        gpu_ids = list(range(self.next_gpu, self.next_gpu + num_gpus))
        self.next_gpu += num_gpus

        def _bootstrap() -> None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
            # TODO(monarch): Remove once concurrent-import-during-unpickling is fixed.
            import torch  # noqa: F401

        return _bootstrap


def _check_batch_invariant(
    trainer: PolicyTrainer.Config, generator: VLLMGenerator.Config
) -> None:
    """Enforce the four bfloat16/determinism preconditions for batch invariance."""
    if not trainer.debug.deterministic:
        raise ValueError("batch_invariant requires deterministic=True")
    if trainer.training.dtype != "bfloat16":
        raise ValueError(
            f"batch_invariant requires bfloat16 training dtype, "
            f"got {trainer.training.dtype!r}"
        )
    if generator.model_dtype != "bfloat16":
        raise ValueError(
            f"batch_invariant requires bfloat16 generator dtype, "
            f"got {generator.model_dtype!r}"
        )
    if trainer.parallelism.enable_sequence_parallel:
        raise ValueError(
            "batch_invariant does not support SP "
            "(NCCL reduce-scatter Ring is the only deterministic mode)."
        )


class _CompletionBatcher:
    """Coalesces concurrent ``generate_tokens`` submissions into one call.

    Every rollout's ``completion_fn(prompt, sampling)`` parks on a
    future; the batcher flushes after a single ``await asyncio.sleep(0)``
    so co-scheduled rollouts on the same event loop can pile their
    prompts into one batched call. The batched ``.call(prompts, ...)``
    sends one ordered message to every TP rank — the schedulers stay
    in lockstep, no NCCL hang.

    All entries in a single flush MUST share the same ``sampling``
    (the actor passes one ``SamplingParams`` to vLLM); training and
    validation use different ``SamplingConfig`` instances so they
    naturally batch separately.
    """

    def __init__(self, generator, unmesh: Callable) -> None:
        self._generator = generator
        self._unmesh = unmesh
        self._pending: list[tuple[list[int], asyncio.Future, SamplingConfig]] = []
        self._lock = asyncio.Lock()
        self._flush_task: asyncio.Task | None = None

    async def submit(
        self, prompt_token_ids: list[int], sampling: SamplingConfig
    ) -> GenerateOutput:
        fut: asyncio.Future[GenerateOutput] = asyncio.get_running_loop().create_future()
        async with self._lock:
            self._pending.append((prompt_token_ids, fut, sampling))
            if self._flush_task is None or self._flush_task.done():
                self._flush_task = asyncio.create_task(self._flush())
        return await fut

    async def _flush(self) -> None:
        # One yield so any co-scheduled rollouts get a chance to submit
        # into the same batch before we send.
        await asyncio.sleep(0)
        async with self._lock:
            batch = list(self._pending)
            self._pending.clear()
            self._flush_task = None
        if not batch:
            return
        sampling = batch[0][2]
        for _, _, sc in batch[1:]:
            assert sc is sampling, (
                "_CompletionBatcher: concurrent submissions with different "
                "sampling configs cannot share a batch (TP ranks need one "
                "SamplingParams per call)"
            )
        prompts = [p for p, _, _ in batch]
        mesh = await self._generator.generate_tokens.call(
            prompts,
            sampling_config=sampling,
        )
        outputs = self._unmesh(mesh)
        for (_, fut, _), out in zip(batch, outputs, strict=True):
            fut.set_result(out)


# ---------------------------------------------------------------------------
# RLTrainer
# ---------------------------------------------------------------------------


class RLTrainer(Configurable):
    """Top-level RL training orchestrator (two-task async pipeline)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        model_spec: ModelSpec | None = None
        """Shared model spec between trainer and generator. Programmatic only."""

        hf_assets_path: str = "./tests/assets/tokenizer"
        """Path to HF assets (weights, tokenizer, config files)."""

        num_steps: int = 10
        """Number of trainer steps."""

        num_prompts_per_step: int = 8
        """Total GRPO groups completed per trainer step (roughly — async)."""

        rollout_group_size: int = 8
        """Number of sibling envs per dataset row (GRPO group size).

        Sibling completions share the task; ``compute_advantages`` centers
        rewards across the group. NOT the same as ``SamplingConfig.n`` —
        the generator always returns one sample per request.
        """

        num_rollout_tasks: int = 4
        """Number of long-lived ``continuous_rollouts`` coroutines."""

        max_rollout_turns: int = 16
        """Hard cap on turns per rollout. AlphabetSort: ~3-5; SumDigits: 1."""

        num_validation_samples: int = 20
        """Held-out examples scored greedily per validation pass."""

        dump_folder: str = "outputs/rl"
        """Root output folder for traces, weights, etc."""

        train_dataset: Configurable.Config = field(default=None)  # type: ignore[assignment]
        """Dataset of training :class:`EnvExample`s."""

        train_builder: Configurable.Config = field(default=None)  # type: ignore[assignment]
        """Builder that materializes ``rollout_group_size`` siblings per row."""

        validation_dataset: Configurable.Config = field(default=None)  # type: ignore[assignment]
        """Held-out dataset for validation; uses ``train_builder`` shape."""

        validation_builder: Configurable.Config = field(default=None)  # type: ignore[assignment]
        """Builder for validation envs; ``group_size=1`` (no GRPO)."""

        renderer: RendererConfig = field(default_factory=RendererConfig)
        """Tokenizer-side renderer; shared across all rollouts."""

        token_env: TokenEnvConfig = field(default_factory=TokenEnvConfig)
        """Termination knobs for :class:`TokenEnv` (parse / length / context / timeout)."""

        log_samples: bool = False
        """Log first completion per group to stdout each step."""

        compile: CompileConfig = field(default_factory=CompileConfig)
        """torch.compile config shared by trainer and generator."""

        trainer: PolicyTrainer.Config = field(
            default_factory=lambda: PolicyTrainer.Config(loss=GRPOLoss.Config())
        )

        generator: VLLMGenerator.Config = field(default_factory=VLLMGenerator.Config)

        replay_buffer: ReplayBuffer.Config = field(default_factory=ReplayBuffer.Config)

        def __post_init__(self) -> None:
            if self.generator.sampling.n != 1:
                raise ValueError(
                    "generator.sampling.n must be 1; siblings come from "
                    "rollout_group_size."
                )
            if self.trainer.debug.batch_invariant:
                _check_batch_invariant(self.trainer, self.generator)

    def __init__(self, config: Config) -> None:
        self.config = config
        self.trainer: PolicyTrainer | None = None
        self.generator: VLLMGenerator | None = None
        self.replay_buffer: ReplayBuffer | None = None
        self._proc_meshes: list = []
        self._renderer_pool = None
        self._completion_fn = None

    # ------------------------------------------------------------------ setup / close

    async def close(self) -> None:
        """Best-effort teardown of actors + proc meshes."""
        logger.info("Closing actors and process meshes")

        # Wake any task parked on replay_buffer.sample(); plain Python.
        if self.replay_buffer is not None:
            try:
                await self.replay_buffer.close()
            except Exception:
                logger.exception("replay_buffer.close failed")

        for name, actor in (
            ("trainer", self.trainer),
            ("generator", self.generator),
        ):
            if actor is None:
                continue
            try:
                await actor.close.call()
            except Exception:
                logger.exception("%s.close failed", name)

        for i, mesh in enumerate(self._proc_meshes):
            try:
                await mesh.stop()
            except Exception:
                logger.exception("mesh.stop[%d] failed", i)
        self._proc_meshes = []

    def _get_rank_0_value(self, result, has_gpus: bool = True):
        """Extract rank-0 result from a Monarch ``ValueMesh``."""
        kwargs = {}
        if self._multi_node:
            kwargs["hosts"] = 0
        if has_gpus:
            kwargs["gpus"] = 0
        return result.item(**kwargs)

    @staticmethod
    def _compute_world_size(p: ParallelismConfig) -> int:
        dp_shard = max(p.data_parallel_shard_degree, 1)
        return (
            p.data_parallel_replicate_degree
            * dp_shard
            * p.tensor_parallel_degree
            * p.pipeline_parallel_degree
            * p.context_parallel_degree
        )

    @sl.log_trace_span("setup")
    async def setup(
        self,
        *,
        host_mesh=None,
        trainer_nodes: int | None = None,
        generator_nodes: int | None = None,
        gpus_per_node: int | None = None,
    ) -> None:
        """Spawn actors and synchronize initial weights.

        Single-host: trainer + generator on disjoint GPU ranges of the
        same machine. Multi-host: dedicate whole nodes per role via
        ``host_mesh``.

        Args:
            host_mesh: Optional multi-node ``HostMesh``.
            trainer_nodes: Trainer node count (required with ``host_mesh``).
            generator_nodes: Generator node count (required with ``host_mesh``).
            gpus_per_node: GPUs per node (required with ``host_mesh``).
        """
        config = self.config

        self.trainer_world_size = self._compute_world_size(config.trainer.parallelism)
        self.generator_world_size = self._compute_world_size(
            config.generator.parallelism
        )
        trainer_parallelism = config.trainer.parallelism
        dp_shard = max(trainer_parallelism.data_parallel_shard_degree, 1)
        self.trainer_dp_degree = (
            trainer_parallelism.data_parallel_replicate_degree * dp_shard
        )

        total_gpus = self.trainer_world_size + self.generator_world_size
        logger.info(
            "%d generator GPUs + %d trainer GPUs = %d total",
            self.generator_world_size,
            self.trainer_world_size,
            total_gpus,
        )

        self._multi_node = host_mesh is not None

        with sl.log_trace_span("mesh_spawn"):
            trainer_mesh, generator_mesh = self._spawn_meshes(
                host_mesh=host_mesh,
                trainer_nodes=trainer_nodes,
                generator_nodes=generator_nodes,
                gpus_per_node=gpus_per_node,
                total_gpus=total_gpus,
            )
            self._proc_meshes = [trainer_mesh, generator_mesh]

            await setup_torch_elastic_env_async(trainer_mesh)
            await setup_torch_elastic_env_async(generator_mesh)

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

            self.generator = generator_mesh.spawn(
                "generator",
                VLLMGenerator,
                config.generator,
                model_spec=config.model_spec,
                model_path=config.hf_assets_path,
                compile_config=config.compile,
                max_num_seqs=max(
                    config.num_prompts_per_step * config.rollout_group_size,
                    config.num_validation_samples,
                ),
                output_dir=config.dump_folder,
            )

        # ReplayBuffer lives on the controller (plain class with
        # ``asyncio.Condition``). add / sample synchronize without a
        # Monarch round-trip; the trainer blocks until samples land
        # instead of sleep-polling.
        self.replay_buffer = config.replay_buffer.build()

        # Initialize torchstore (used by trainer.push / generator.pull).
        with sl.log_trace_span("torchstore_init"):
            await ts.initialize(mesh=trainer_mesh, strategy=ts.LocalRankStrategy())

        # Build the renderer pool on the controller (shared by all rollouts).
        self._renderer_pool = config.renderer.build(config.hf_assets_path)

        self._completion_fn = self._build_completion_fn()

        # Initial weight sync.
        with sl.log_trace_span("trainer_push_model_state_dict"):
            await self.trainer.push_model_state_dict.call()
        with sl.log_trace_span("generator_pull_model_state_dict"):
            await self.generator.pull_model_state_dict.call(0)

    def _spawn_meshes(
        self,
        *,
        host_mesh,
        trainer_nodes,
        generator_nodes,
        gpus_per_node,
        total_gpus,
    ):
        if host_mesh is None:
            provisioner = Provisioner(total_gpus=total_gpus)
            trainer_mesh = this_host().spawn_procs(
                per_host={"gpus": self.trainer_world_size},
                bootstrap=provisioner.allocate(self.trainer_world_size),
            )
            generator_mesh = this_host().spawn_procs(
                per_host={"gpus": self.generator_world_size},
                bootstrap=provisioner.allocate(self.generator_world_size),
            )
            return trainer_mesh, generator_mesh

        # Multi-node path
        if trainer_nodes is None or generator_nodes is None or gpus_per_node is None:
            raise ValueError(
                "trainer_nodes, generator_nodes, and gpus_per_node are required "
                "when host_mesh is provided"
            )
        assert self.trainer_world_size % trainer_nodes == 0
        assert self.generator_world_size % generator_nodes == 0
        trainer_gpus_per_node = self.trainer_world_size // trainer_nodes
        generator_gpus_per_node = self.generator_world_size // generator_nodes

        trainer_host_mesh = host_mesh.slice(hosts=slice(0, trainer_nodes))
        generator_host_mesh = host_mesh.slice(
            hosts=slice(trainer_nodes, trainer_nodes + generator_nodes)
        )

        trainer_provisioner = Provisioner(total_gpus=gpus_per_node)
        generator_provisioner = Provisioner(total_gpus=gpus_per_node)

        trainer_mesh = trainer_host_mesh.spawn_procs(
            per_host={"gpus": trainer_gpus_per_node},
            bootstrap=trainer_provisioner.allocate(trainer_gpus_per_node),
        )
        generator_mesh = generator_host_mesh.spawn_procs(
            per_host={"gpus": generator_gpus_per_node},
            bootstrap=generator_provisioner.allocate(generator_gpus_per_node),
        )
        return trainer_mesh, generator_mesh

    # ------------------------------------------------------------------ rollout-side helpers

    def _build_completion_fn(self):
        """Closure handed to ``do_single_rollout``; hides Monarch + the
        rank-0 unmesh + the controller-side batcher from the driver.

        Concurrent rollouts each call ``completion_fn(prompts, sampling)``;
        the batcher coalesces submissions into one
        ``generator.generate_tokens.call(prompts_list, ...)`` so every TP
        rank's vLLM scheduler sees the same ordered list (required by
        ``external_launcher`` TP — see round 7 §4).
        """
        self._batcher = _CompletionBatcher(self.generator, self._get_rank_0_value)

        async def completion_fn(
            prompt_token_ids: list[int], sampling: SamplingConfig
        ) -> GenerateOutput:
            return await self._batcher.submit(prompt_token_ids, sampling)

        return completion_fn

    async def _run_group(
        self,
        *,
        builder_config: Configurable.Config,
        example: EnvExample,
        group_size: int,
        sampling: SamplingConfig,
    ) -> list[RolloutOutput]:
        """Build ``group_size`` sibling envs from ``example`` and roll them out.

        Shared by training (``continuous_rollouts``: train builder,
        ``rollout_group_size``, sampling config) and validation
        (``validate``: validation builder, ``group_size=1``, greedy).
        """
        builder: EnvBuilder = builder_config.build()
        envs: Sequence[MessageEnv] = await builder.make_envs(
            example,
            group_size=group_size,
        )
        return await do_rollout_group(
            envs=envs,
            renderer=self._renderer_pool,
            completion_fn=self._completion_fn,
            sampling=sampling,
            group_id=example.task_id,
            max_turns=self.config.max_rollout_turns,
            token_env_config=self.config.token_env,
        )

    @staticmethod
    def _build_replay_samples(
        rollouts: Sequence[RolloutOutput],
    ) -> list[ReplaySample]:
        """Convert rollouts to replay samples + stamp group-mean advantages."""
        advs = compute_advantages(rollouts)
        out: list[ReplaySample] = []
        for r in rollouts:
            adv = advs[(r.group_id, r.sample_idx)]
            for s in rollout_to_replay_samples(r):
                s.advantage = adv
                out.append(s)
        return out

    # ------------------------------------------------------------------ trainer-side helpers

    @staticmethod
    def _collate(samples: list[ReplaySample]) -> TrainBatch:
        """Pack samples into a varlen :class:`TrainBatch` (pre-shifted).

        See :class:`TrainBatch` for the shift contract. All three
        per-token tensors (``loss_mask``, ``behavior_logprobs``,
        ``advantages_per_token``) are ``unshifted[1:]`` after packing.
        """
        all_ids: list[int] = []
        all_mask: list[int] = []
        all_lp: list[float] = []
        all_adv: list[float] = []
        seq_lens: list[int] = []
        for s in samples:
            # Load-bearing invariant for the global-shift alignment: the
            # first token of every sample MUST be unmasked (prompt). If
            # this ever breaks (e.g., a future assistant-prefix env),
            # cross-boundary positions would leak gradient.
            assert s.loss_mask and s.loss_mask[0] == 0, (
                f"loss_mask[0] must be 0 (got {s.loss_mask[:5]}); the global-"
                "shift collate relies on every sample starting with a prompt token."
            )
            all_ids.extend(s.token_ids)
            all_mask.extend(s.loss_mask)
            all_lp.extend(s.behavior_logprobs)
            all_adv.extend([s.advantage] * len(s.token_ids))
            seq_lens.append(len(s.token_ids))

        token_ids_t = torch.tensor([all_ids], dtype=torch.long)
        # Shift by 1 to align with next-token logprobs ([:, :-1] / [:, 1:]).
        loss_mask_t = torch.tensor([all_mask[1:]], dtype=torch.float32)
        behavior_lp_t = torch.tensor([all_lp[1:]], dtype=torch.float32)
        adv_per_token_t = torch.tensor([all_adv[1:]], dtype=torch.float32)
        return TrainBatch(
            token_ids=token_ids_t,
            seq_lens=seq_lens,
            loss_mask=loss_mask_t,
            behavior_logprobs=behavior_lp_t,
            advantages_per_token=adv_per_token_t,
        )

    # ------------------------------------------------------------------ validation

    @sl.log_trace_span("validate")
    async def validate(self) -> dict[str, float | dict[str, float]]:
        """Greedy held-out rollouts; no buffer."""
        cfg = self.config
        dataset: EnvDataset = cfg.validation_dataset.build()
        examples = dataset.sample_groups(
            step=0,
            num_groups=cfg.num_validation_samples,
        )
        greedy = SamplingConfig(
            n=1,
            temperature=0.0,
            top_p=1.0,
            max_tokens=cfg.generator.sampling.max_tokens,
        )

        async def _one(example: EnvExample) -> RolloutOutput:
            outputs = await self._run_group(
                builder_config=cfg.validation_builder,
                example=example,
                group_size=1,
                sampling=greedy,
            )
            return outputs[0]

        rollouts = await asyncio.gather(*[_one(ex) for ex in examples])
        return aggregate_rewards(rollouts)

    # ------------------------------------------------------------------ training loop

    async def train(self) -> None:
        cfg = self.config
        num_steps = cfg.num_steps
        shutdown = asyncio.Event()
        self._train_step = 0

        logger.info("Pre-training validation")
        pre_validation = await self.validate()
        logger.info("Pre:  %s", format_validation(pre_validation))
        sl.log_trace_instant("training_start")

        async def continuous_rollouts(task_idx: int) -> None:
            """Run one rollout-producer coroutine.

            Each task picks a unique slot ``task_idx`` from each batch of
            ``num_rollout_tasks`` examples the dataset hands back, so the
            ``N`` concurrent producers always sample distinct prompts
            within one dataset step. Without this, all tasks would call
            ``sample_groups(step=..., num_groups=1)`` and burn compute on
            duplicate rollouts of the same prompt.
            """
            train_dataset: EnvDataset = cfg.train_dataset.build()
            dataset_step = 0
            while not shutdown.is_set():
                examples = train_dataset.sample_groups(
                    step=dataset_step,
                    num_groups=cfg.num_rollout_tasks,
                )
                if not examples or task_idx >= len(examples):
                    return
                example = examples[task_idx]
                with sl.log_trace_span("rollout_group"):
                    rollouts = await self._run_group(
                        builder_config=cfg.train_builder,
                        example=example,
                        group_size=cfg.rollout_group_size,
                        sampling=cfg.generator.sampling,
                    )
                samples = self._build_replay_samples(rollouts)
                if samples:
                    await self.replay_buffer.add(samples)

                if cfg.log_samples:
                    if dataset_step % 4 == 0 and task_idx == 0:
                        log_first_sample(rollouts)
                    dump_rollouts_jsonl(
                        rollouts,
                        dump_folder=cfg.dump_folder,
                        train_step=self._train_step,
                    )
                dataset_step += 1

        skipped_zero_advantage_steps = 0

        async def continuous_training() -> None:
            nonlocal skipped_zero_advantage_steps
            while self._train_step < num_steps and not shutdown.is_set():
                try:
                    batches_per_rank = await self.replay_buffer.sample(
                        curr_policy_version=self._train_step,
                    )
                except BufferClosedError:
                    return
                step_start = time.perf_counter()

                # Skip when no group in the batch has any advantage signal:
                # forward/backward would just multiply gradients by zero and
                # waste compute. Counts toward ``num_steps`` so a fully
                # degenerate task still terminates.
                if not _any_advantage_signal(batches_per_rank):
                    skipped_zero_advantage_steps += 1
                    self._train_step += 1
                    sl.set_step(self._train_step)
                    logger.info(
                        "Step %2d | SKIPPED (no advantage signal; cumulative=%d)",
                        self._train_step,
                        skipped_zero_advantage_steps,
                    )
                    sl.log_trace_scalar(
                        {
                            "train/skipped_zero_advantage_steps": skipped_zero_advantage_steps
                        }
                    )
                    continue

                # Collate one TrainBatch per DP rank.
                train_batches = [self._collate(b) for b in batches_per_rank]
                num_global_valid_tokens = sum(
                    int(b.loss_mask.sum().item()) for b in train_batches
                )

                with sl.log_trace_span("trainer_forward_backward_call"):
                    fb = self._get_rank_0_value(
                        await self.trainer.forward_backward.call(
                            train_batches,
                            num_global_valid_tokens=num_global_valid_tokens,
                        )
                    )
                with sl.log_trace_span("trainer_optim_step_call"):
                    opt = self._get_rank_0_value(await self.trainer.optim_step.call())
                metrics = {**fb, **opt.metrics}

                self._train_step += 1
                sl.set_step(self._train_step)

                with sl.log_trace_span("trainer_push_model_state_dict"):
                    await self.trainer.push_model_state_dict.call()
                with sl.log_trace_span("generator_pull_model_state_dict"):
                    await self.generator.pull_model_state_dict.call(self._train_step)

                step_time = time.perf_counter() - step_start
                if not math.isfinite(metrics.get("loss/mean", 0.0)):
                    logger.error("Loss is NaN/Inf; training diverged")
                    shutdown.set()
                    break
                logger.info(
                    "Step %2d | Loss: %+.4f | ratio_clip=%.3f | grad_norm=%.3f | time=%.1fs",
                    self._train_step,
                    metrics.get("loss/mean", 0.0),
                    metrics.get("loss/ratio/clipped_frac", 0.0),
                    metrics.get("train/grad_norm/mean", 0.0),
                    step_time,
                )

        # Fail-fast: any task error triggers shutdown.
        rollout_tasks = [
            asyncio.create_task(continuous_rollouts(i), name=f"rollout_{i}")
            for i in range(cfg.num_rollout_tasks)
        ]
        training_task = asyncio.create_task(continuous_training(), name="training")

        def _on_done(task, label):
            if task.cancelled():
                return
            exc = task.exception()
            if exc is not None:
                logger.error("%s failed: %s", label, exc, exc_info=exc)
                shutdown.set()

        for i, t in enumerate(rollout_tasks):
            t.add_done_callback(lambda x, i=i: _on_done(x, f"rollout_{i}"))
        training_task.add_done_callback(lambda x: _on_done(x, "training"))

        try:
            await training_task
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("Training interrupted; shutting down")
        finally:
            shutdown.set()
            try:
                await asyncio.wait_for(
                    asyncio.gather(*rollout_tasks, return_exceptions=True),
                    timeout=15.0,
                )
            except asyncio.TimeoutError:
                logger.warning("rollout tasks didn't finish in 15s; cancelling")
                for t in rollout_tasks:
                    t.cancel()
                await asyncio.gather(*rollout_tasks, return_exceptions=True)

        sl.log_trace_instant("training_end")
        post_validation = await self.validate()
        logger.info(
            "Summary:\n  Pre:  %s\n  Post: %s",
            format_validation(pre_validation),
            format_validation(post_validation),
        )


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _any_advantage_signal(
    batches_per_rank: Sequence[Sequence[ReplaySample]], *, eps: float = 1e-12
) -> bool:
    """True iff any sample in any rank's batch has a non-zero advantage.

    The group-mean GRPO baseline produces zero advantages when every
    rollout in a group lands the same reward (the model already solved
    the task or completely failed it). Forwarding such batches just
    multiplies gradients by zero and burns compute, so the training
    loop skips them and tracks the count via
    ``train/skipped_zero_advantage_steps``.
    """
    for batch in batches_per_rank:
        for sample in batch:
            if abs(sample.advantage) > eps:
                return True
    return False


async def main() -> None:
    config = ConfigManager().parse_args()
    sl.init_structured_logger(
        source="rl_controller",
        output_dir=config.dump_folder,
        rank=0,
        # pyrefly: ignore [missing-attribute]
        enable=config.trainer.debug.enable_structured_logging,
    )
    sl.log_trace_instant("structured_logger_started")

    trainer = RLTrainer(config)
    try:
        await trainer.setup()
        await trainer.train()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Interrupted; attempting graceful shutdown...")
    finally:
        await trainer.close()


if __name__ == "__main__":
    asyncio.run(main())
