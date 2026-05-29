# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
RL training loop using Monarch Actors.

This demonstrates:
1. Distributed actor architecture with VLLMGenerator (vLLM) and PolicyTrainer (TorchTitan)
   running on separate GPU meshes
2. Weight synchronization across meshes: trainer gathers full (unsharded) weights,
   generator reshards to match its own parallelism layout via distribute_tensor
3. Envs driven rollouts; reward and advantage computation live inline
   in the controller.

Command to run:
python3 torchtitan/experiments/rl/grpo.py \
    --module rl --config rl_grpo_qwen3_0_6b \
    --hf_assets_path=<path_to_model_checkpoint>
"""

import asyncio
import logging
import math
import os
import statistics
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

# must run before torch import
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
    Completion,
    SamplingConfig,
    VLLMGenerator,
)
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.batcher import Batcher
from torchtitan.experiments.rl.envs import RendererEnv, TokenizedTurn
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.recipes import Task
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.rollouts import (
    DatasetOutput,
    last_assistant_text,
    prepare_rollout_metrics,
    Rollout,
    rollout_to_episode,
    RolloutStatus,
    RolloutTurn,
)
from torchtitan.experiments.rl.types import Episode
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
            }

        return loss, metrics


class Provisioner:
    """Allocates non-overlapping GPU ranges for Monarch proc meshes.

    In non-colocated mode, the trainer and generator run on separate GPU
    meshes (e.g. GPUs 0-3 for training, GPUs 4-7 for generation). Each
    call to `allocate(n)` reserves the next *n* GPUs and returns a
    bootstrap callable that sets `CUDA_VISIBLE_DEVICES` before CUDA
    initializes in the spawned process, ensuring each mesh only sees its
    own devices.
    """

    def __init__(self, total_gpus: int = 8):
        self.total_gpus = total_gpus
        self.next_gpu = 0

    @property
    def available(self) -> int:
        return self.total_gpus - self.next_gpu

    def allocate(self, num_gpus: int) -> Callable[[], None]:
        if num_gpus > self.available:
            raise RuntimeError(
                f"Requested {num_gpus} GPUs but only {self.available} "
                f"available (total={self.total_gpus}, allocated={self.next_gpu})"
            )
        gpu_ids = list(range(self.next_gpu, self.next_gpu + num_gpus))
        self.next_gpu += num_gpus

        def _bootstrap():
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
            # TODO: Remove once Monarch/PyTorch fixes concurrent import during unpickling.
            import torch  # noqa: F401

        return _bootstrap


def _log_samples(episodes: list[Episode]) -> None:
    """Log the first episode per prompt for debugging."""
    seen_prompts: set[int] = set()
    for ep in episodes:
        if ep.prompt_idx in seen_prompts:
            continue
        seen_prompts.add(ep.prompt_idx)
        logger.info(f"  [prompt {ep.prompt_idx}] reward={ep.reward:+.1f}")
        logger.info(f"       A: {ep.text[:300].replace(chr(10), ' ').strip()}")


def _overflow_rollouts(
    group_id: str,
    initial_turns: list[TokenizedTurn],
) -> list[Rollout]:
    """Build placeholder `TRUNCATED_OVERFLOW` rollouts for an initial overflow.

    No turns are attached — there's no usable training data — so
    `_build_episodes` skips them. They still appear in rollout metrics so
    truncation rate includes initial-prompt overflow.
    """
    return [
        Rollout(
            group_id=group_id,
            sample_idx=sample_idx,
            status=RolloutStatus.TRUNCATED_OVERFLOW,
            turns=[],
        )
        for sample_idx, _ in enumerate(initial_turns)
    ]


@sl.log_trace_span("step_group")
async def _do_group_step(
    *,
    envs: list[RendererEnv],
    group_id: str,
    initial_turns: list[TokenizedTurn],
    completions: list[Completion],
) -> list[Rollout]:
    """Step each env in the group and pack the results into rollouts.

    Reward is left unset; the controller calls `task.score_group(...)` after
    this and applies `reward` / `reward_components` to each rollout.

    Args:
        envs: Sibling envs for one prompt group.
        group_id: Stable prompt-group ID used for advantage centering.
        initial_turns: Initial prompt turns for each sibling env.
        completions: Generator completions for each sibling env.

    Returns:
        Unscored rollouts for the group, one per completion.
    """
    # Step all envs in parallel
    stepped_turns: list[TokenizedTurn] = await asyncio.gather(
        *(env.step_completion(c) for env, c in zip(envs, completions, strict=True))
    )

    # TODO: support multi-turn rollouts in the controller.
    for env_idx, st in enumerate(stepped_turns):
        if st.next_token_ids is not None:
            raise RuntimeError(
                f"env {group_id}/{env_idx} returned a non-terminal turn; "
                "the controller does not yet support multi-turn rollouts."
            )

    # Pack into Rollouts (reward unset)
    rollouts: list[Rollout] = []
    for sample_idx, (initial, completion, stepped) in enumerate(
        zip(initial_turns, completions, stepped_turns, strict=True)
    ):
        if initial.next_token_ids is None:
            raise RuntimeError(
                f"env {group_id}/{sample_idx} has no initial prompt tokens"
            )
        rollouts.append(
            Rollout(
                group_id=group_id,
                sample_idx=sample_idx,
                status=stepped.status,
                turns=[
                    RolloutTurn(
                        prompt_token_ids=list(initial.next_token_ids),
                        response_token_ids=list(completion.token_ids),
                        response_logprobs=list(completion.token_logprobs),
                        policy_version=completion.policy_version,
                        response_messages=list(stepped.last_response_messages),
                    )
                ],
            )
        )
    return rollouts


class RLTrainer(Configurable):
    """Top-level RL training orchestrator.

    Owns a `PolicyTrainer` actor (gradient updates), a `VLLMGenerator` actor
    (sampling), one or more `Task`s (rubric + env construction), and a
    `Dataset` per phase (train/validation). Each row from the dataset
    carries `DatasetOutput.task`, which the controller uses to route the
    row to the matching `Task` in `tasks`. Each training step samples
    groups of rollouts, scores them via per-task rubrics, builds GRPO
    advantages, and syncs trainer weights to the generator.

    Example:

        cfg = config_registry.rl_grpo_qwen3_0_6b()
        trainer = cfg.build()
        await trainer.setup_async()
        await trainer.train()
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Top-level config for RL training."""

        model_spec: ModelSpec | None = None
        """Model specification shared by trainer and generator. Set
        programmatically via `config_registry` (not from CLI)."""

        hf_assets_path: str = "./tests/assets/tokenizer"
        """Path to HF assets (model weights, tokenizer, config files)."""

        num_steps: int = 10
        """Number of RL training steps."""

        dump_folder: str = "outputs/rl"
        """Root output folder for RL artifacts."""

        num_prompts_per_step: int = 5
        """Number of distinct prompts (= GRPO groups) drawn per training step.
        Total episodes per step is `num_prompts_per_step * group_size`, where
        `group_size` is `generator.sampling.n`."""

        num_validation_samples: int = 20
        """Number of held-out prompts scored greedily per validation pass."""

        train_dataset: Configurable.Config = field(default=None)  # type: ignore[assignment]
        """Training dataset; rows route to `tasks` by `DatasetOutput.task`."""

        validation_dataset: Configurable.Config = field(default=None)  # type: ignore[assignment]
        """Validation dataset; same routing as `train_dataset`."""

        tasks: dict[str, Task.Config] = field(default_factory=dict)
        """Map from task name to `Task.Config`. Single-task runs use a
        one-key dict."""

        renderer: RendererConfig = field(default_factory=RendererConfig)
        """Message-to-token renderer config."""

        async_executor_max_workers: int = 64
        """Worker count for the default `asyncio.to_thread` executor;
        load-bearing for concurrent renderer calls at
        `num_prompts_per_step * group_size` rollouts."""

        log_samples: bool = False
        """Whether to log the first completion per episode during training and
        validation."""

        compile: CompileConfig = field(default_factory=CompileConfig)
        """`torch.compile` config shared by trainer and generator."""

        batcher: Batcher.Config = field(default_factory=Batcher.Config)
        """Batcher config (local_batch_size, seq_len)."""

        trainer: PolicyTrainer.Config = field(
            default_factory=lambda: PolicyTrainer.Config(loss=GRPOLoss.Config())
        )
        """`PolicyTrainer` actor config (optimizer, training, parallelism)."""

        generator: VLLMGenerator.Config = field(default_factory=VLLMGenerator.Config)
        """`VLLMGenerator` actor config (vLLM engine, sampling)."""

        metrics: m.MetricsProcessor.Config = field(
            default_factory=m.MetricsProcessor.Config
        )
        """Metrics processor config."""

        def __post_init__(self):
            if self.generator.checkpoint.enable:
                raise ValueError(
                    "Generator checkpoint must be disabled in the RL loop "
                    "(weights are synced from the trainer via TorchStore). "
                    "Set generator.checkpoint.enable=False."
                )
            if self.train_dataset is None:
                raise ValueError("train_dataset is required")
            if self.validation_dataset is None:
                raise ValueError("validation_dataset is required")
            if not self.tasks:
                raise ValueError("tasks must not be empty")

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

    def __init__(self, config: Config):
        self.config = config
        self.trainer = None
        self.generator = None
        self._proc_meshes = []
        self.metrics_processor: m.MetricsProcessor = config.metrics.build(
            log_dir=config.dump_folder,
            job_config=config.to_dict(),
        )
        self.renderer = config.renderer.build(model_path=config.hf_assets_path)
        self._stop_token_ids = list(self.renderer.get_stop_token_ids())
        # TODO: pass our own tokenizer to the renderer and read pad/eos off it
        # once `renderers` supports bring-your-own-tokenizer
        # (https://github.com/PrimeIntellect-ai/renderers/pull/70).
        # Until then, reach into the renderer's tokenizer for the pad id (eos doubles as pad).
        self.batcher = Batcher(
            config.batcher, pad_id=self.renderer._tokenizer.eos_token_id
        )
        self._train_dataset = config.train_dataset.build()
        self._validation_dataset = config.validation_dataset.build()
        self._tasks: dict[str, Task] = {
            name: cfg.build() for name, cfg in config.tasks.items()
        }

    async def close(self):
        """Best-effort: tear down actors, close metric backends, then stop proc meshes."""
        logger.info("Closing: tearing down actors and process meshes.")
        for actor_name, actor in (
            ("trainer", self.trainer),
            ("generator", self.generator),
        ):
            if actor is None:
                continue
            try:
                await actor.close.call()
            except Exception:
                logger.exception("%s.close failed", actor_name)

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

    def _get_rank_0_value(self, result, has_gpus: bool = True):
        """Extract rank 0 result, handling both single and multi-node meshes.

        Monarch actor endpoints return results from all ranks in the mesh.
        This picks out rank 0's result by indexing into the host and GPU
        dimensions as needed (multi-node meshes have an extra host dimension).
        This should be used in cases where all ranks return the same result.
        """
        kwargs = {}
        if self._multi_node:
            kwargs["hosts"] = 0
        if has_gpus:
            kwargs["gpus"] = 0
        return result.item(**kwargs)

    @staticmethod
    def _compute_world_size(p: ParallelismConfig) -> int:
        """Compute world size from all parallel dimensions."""
        dp_shard = max(p.data_parallel_shard_degree, 1)
        return (
            p.data_parallel_replicate_degree
            * dp_shard
            * p.tensor_parallel_degree
            * p.pipeline_parallel_degree
            * p.context_parallel_degree
        )

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
        host_mesh=None,
        trainer_nodes: int | None = None,
        generator_nodes: int | None = None,
        gpus_per_node: int | None = None,
    ):
        """Spawn Monarch actors on separate meshes and initialize weights.

        Kept separate from ``__init__`` because actor spawning, torch
        elastic env setup, TorchStore initialization, and the initial
        weight push/pull are all ``await``-based runtime side effects
        that cannot run in a synchronous constructor.

        Creates separate GPU meshes for trainer and generator and
        synchronizes initial weights from trainer to generator. Must be
        called before :meth:`train`.

        Args:
            host_mesh: Optional multi-node HostMesh. When provided,
                whole nodes are dedicated to trainer vs generator
                roles instead of partitioning GPUs on a single host.
            trainer_nodes: Number of nodes for the trainer (required when
                host_mesh is provided).
            generator_nodes: Number of nodes for the generator (required when
                host_mesh is provided).
            gpus_per_node: GPUs per node, assumed to be the same across all
                nodes (no heterogeneous node configurations). Required when
                host_mesh is provided.
        """
        # Scale the default thread executor so concurrent renderer calls
        # (asyncio.to_thread inside RendererEnv) don't queue at the
        # default min(32, os.cpu_count() + 4) cap.
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=self.config.async_executor_max_workers)
        )

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
            f"{self.generator_world_size} generator GPUs + "
            f"{self.trainer_world_size} trainer GPUs = {total_gpus} total"
        )

        self._multi_node = host_mesh is not None

        # TODO(observability): the mesh_spawn span wraps ~80 LoC of branching
        # provisioner logic. Pull a Provisioner.spawn_meshes(...) helper and
        # shrink this span to a single call.
        with sl.log_trace_span("mesh_spawn"):
            if host_mesh is not None:
                # Multi-node mode: dedicate whole nodes to trainer vs generator
                if (
                    trainer_nodes is None
                    or generator_nodes is None
                    or gpus_per_node is None
                ):
                    raise ValueError(
                        "trainer_nodes, generator_nodes, and gpus_per_node are "
                        "required when host_mesh is provided"
                    )
                # Validate that world sizes are evenly divisible by node counts
                assert self.trainer_world_size % trainer_nodes == 0, (
                    f"trainer_world_size ({self.trainer_world_size}) must be "
                    f"evenly divisible by trainer_nodes ({trainer_nodes})"
                )
                assert self.generator_world_size % generator_nodes == 0, (
                    f"generator_world_size ({self.generator_world_size}) must be "
                    f"evenly divisible by generator_nodes ({generator_nodes})"
                )

                # Compute GPUs per node for each role based on the config's
                # world size and number of nodes allocated to that role
                trainer_gpus_per_node = self.trainer_world_size // trainer_nodes
                generator_gpus_per_node = self.generator_world_size // generator_nodes

                trainer_host_mesh = host_mesh.slice(hosts=slice(0, trainer_nodes))
                generator_host_mesh = host_mesh.slice(
                    hosts=slice(trainer_nodes, trainer_nodes + generator_nodes)
                )

                # Use Provisioner to set CUDA_VISIBLE_DEVICES so each role
                # only sees its own GPUs and doesn't conflict with other
                # processes on the node
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
            else:
                # Single-node mode: partition GPUs on this_host() via
                # CUDA_VISIBLE_DEVICES
                provisioner = Provisioner(total_gpus=total_gpus)
                trainer_mesh = this_host().spawn_procs(
                    per_host={"gpus": self.trainer_world_size},
                    bootstrap=provisioner.allocate(self.trainer_world_size),
                )
                generator_mesh = this_host().spawn_procs(
                    per_host={"gpus": self.generator_world_size},
                    bootstrap=provisioner.allocate(self.generator_world_size),
                )

            # Store proc meshes for cleanup
            self._proc_meshes = [trainer_mesh, generator_mesh]

            await setup_torch_elastic_env_async(trainer_mesh)
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

            self.generator = generator_mesh.spawn(
                "generator",
                VLLMGenerator,
                config.generator,
                model_spec=config.model_spec,
                model_path=config.hf_assets_path,
                compile_config=config.compile,
                max_num_seqs=max(
                    config.num_prompts_per_step * config.generator.sampling.n,
                    config.num_validation_samples,
                ),
                output_dir=config.dump_folder,
            )

        # Initialize TorchStore for weight sync between trainer and generator.
        # StorageVolumes are spawned on the trainer mesh so they are colocated
        # with the weight source for faster data access in the non-RDMA path.
        # LocalRankStrategy: routes each process to a storage volume based on
        #   LOCAL_RANK, so colocated processes share the same volume.
        # https://github.com/meta-pytorch/torchstore
        with sl.log_trace_span("torchstore_init"):
            await ts.initialize(mesh=trainer_mesh, strategy=ts.LocalRankStrategy())

        # Initial weight sync from trainer to generator
        with sl.log_trace_span("trainer_push_model_state_dict"):
            self.trainer.push_model_state_dict.call().get()
        with sl.log_trace_span("generator_pull_model_state_dict"):
            self.generator.pull_model_state_dict.call(0).get()

    @sl.log_trace_span("_run_rollouts")
    async def _run_rollouts(
        self,
        *,
        dataset: object,
        num_groups: int,
        group_size: int,
        sampling: SamplingConfig,
        step: int,
        group_offset: int,
        metrics_prefix: str,
    ) -> tuple[list[Rollout], list[m.Metric]]:
        """Build groups, batch-generate, then per group: step (controller) +
        score (task rubric). Single-turn; rows route to `self._tasks` by
        `example.task`. Per-group failures are logged and dropped.

        TODO(continuous-batching): collapse to one `asyncio.gather` over
        `num_groups * group_size` rollouts once vLLM supports it.
        """
        # One example per group, routed to its task; build that group's envs.
        examples: list[DatasetOutput] = []
        tasks_per_group: list[Task] = []
        envs_per_group: list[list[RendererEnv]] = []
        group_ids: list[str] = []
        for g in range(num_groups):
            example = dataset.sample_example()
            if example.task not in self._tasks:
                raise KeyError(
                    f"dataset row tagged task={example.task!r} but no such "
                    f"task in RLTrainer.Config.tasks (have {sorted(self._tasks)})"
                )
            task = self._tasks[example.task]
            examples.append(example)
            tasks_per_group.append(task)
            envs_per_group.append(
                task.make_envs(
                    example=example,
                    group_size=group_size,
                    renderer=self.renderer,
                )
            )
            group_ids.append(f"{example.task}/step={step}/group={group_offset + g}")

        all_envs = [env for envs in envs_per_group for env in envs]
        try:
            # Render initial prompts (siblings in a group share the turn-0 prompt)
            initial_turns: list[TokenizedTurn] = await asyncio.gather(
                *(env.initial_prompt() for env in all_envs)
            )

            # Siblings share the turn-0 prompt; skip groups that overflow it
            # (emitted as TRUNCATED_OVERFLOW below, without a generator call).
            runnable_groups = [
                g_idx
                for g_idx in range(num_groups)
                if initial_turns[g_idx * group_size].next_token_ids is not None
            ]
            prompts = [
                list(initial_turns[g_idx * group_size].next_token_ids)
                for g_idx in runnable_groups
            ]

            # One batched generate
            completions, gen_metrics = self._get_rank_0_value(
                self.generator.generate.call(
                    prompts,
                    sampling_config=sampling,
                    metrics_prefix=metrics_prefix,
                ).get()
            )
            expected = len(prompts) * sampling.n
            if len(completions) != expected:
                raise RuntimeError(
                    f"expected {expected} completions, got {len(completions)}"
                )

            # Rebucket completions back to group index
            completions_by_group: dict[int, list[Completion]] = defaultdict(list)
            for c in completions:
                completions_by_group[runnable_groups[c.prompt_idx]].append(c)

            # Per group: step (controller) + score (task rubric). Broken groups
            # are logged and dropped.
            rollouts: list[Rollout] = []
            num_failed_groups = 0
            for g_idx in range(num_groups):
                group_initial = initial_turns[
                    g_idx * group_size : (g_idx + 1) * group_size
                ]
                if g_idx not in completions_by_group:
                    rollouts.extend(_overflow_rollouts(group_ids[g_idx], group_initial))
                    continue
                try:
                    group_rollouts = await _do_group_step(
                        envs=envs_per_group[g_idx],
                        group_id=group_ids[g_idx],
                        initial_turns=group_initial,
                        completions=completions_by_group[g_idx],
                    )
                    rewards = await tasks_per_group[g_idx].score_group(
                        group_rollouts,
                        examples[g_idx].env_input,
                    )
                    for rollout, reward in zip(group_rollouts, rewards, strict=True):
                        rollout.reward = reward.reward
                        rollout.reward_components = reward.components
                    rollouts.extend(group_rollouts)
                except Exception:
                    logger.exception(
                        "group %s failed; dropping its rollouts",
                        group_ids[g_idx],
                    )
                    num_failed_groups += 1
        finally:
            await asyncio.gather(
                *(env.close() for env in all_envs), return_exceptions=True
            )

        gen_metrics = list(gen_metrics)
        gen_metrics.append(
            m.Metric("rollout/group_failures", m.Sum(float(num_failed_groups)))
        )
        return rollouts, gen_metrics

    @sl.log_trace_span("_collect_rollouts")
    async def _collect_rollouts(
        self,
        num_groups: int,
        step: int,
        group_offset: int,
    ) -> tuple[list[Rollout], list[m.Metric]]:
        """Collect train rollouts and emit rollout-shape metrics.

        Args:
            num_groups: Number of prompt groups to collect in this round.
            step: Current training step (tagged into `group_id` for metrics).
            group_offset: Starting group index so generated `group_id`s
                stay unique across collection rounds within a step.

        Returns:
            Scored rollouts and rollout/generator metrics.
        """
        group_size = self.config.generator.sampling.n
        sampling = replace(
            self.config.generator.sampling, stop_token_ids=self._stop_token_ids
        )
        # TODO: per-prompt max_tokens clamp (context_window - len(prompt)) so we
        # don't over-allocate generation budget on long prompts. Initial-prompt
        # overflow is already handled by RendererEnv + _run_rollouts.
        rollouts, generation_metrics = await self._run_rollouts(
            dataset=self._train_dataset,
            num_groups=num_groups,
            group_size=group_size,
            sampling=sampling,
            step=step,
            group_offset=group_offset,
            metrics_prefix="generator",
        )

        rollout_metrics = prepare_rollout_metrics("rollout", rollouts)
        rollout_metrics += generation_metrics
        return rollouts, rollout_metrics

    @staticmethod
    @sl.log_trace_span("_build_episodes")
    def _build_episodes(
        rollouts: list[Rollout],
    ) -> tuple[list[Episode], list[m.Metric]]:
        """Build train episodes and GRPO advantages from scored rollouts.

        Groups rollouts by `group_id`, centers rewards by group mean, skips
        rollouts without training tokens, and emits reward/advantage metrics.

        Args:
            rollouts: Scored rollouts from one collection round.

        Returns:
            Train episodes plus episode-level metrics.
        """
        # Group rollouts by group_id
        groups: dict[str, list[Rollout]] = {}
        for rollout in rollouts:
            groups.setdefault(rollout.group_id, []).append(rollout)
        group_id_to_idx = {gid: i for i, gid in enumerate(sorted(groups))}

        # Mean-baseline advantage per group
        episodes: list[Episode] = []
        group_stds: list[float] = []
        for group_id, group in groups.items():
            rewards = [
                float(rollout.reward) for rollout in group if rollout.reward is not None
            ]
            if not rewards:
                continue
            group_mean = sum(rewards) / len(rewards)
            group_stds.append(statistics.pstdev(rewards))
            for rollout in group:
                # Skip rollouts with no usable training data (e.g. initial-prompt
                # overflow → empty turns).
                if rollout.reward is None or not rollout.turns:
                    continue
                rollout.advantage = rollout.reward - group_mean
                episode = rollout_to_episode(rollout, text=last_assistant_text(rollout))
                episodes.append(replace(episode, prompt_idx=group_id_to_idx[group_id]))

        num_groups = len(groups)
        zero_std_frac = (
            sum(1 for s in group_stds if s == 0.0) / num_groups if num_groups else 0.0
        )
        episode_metrics: list[m.Metric] = [
            m.Metric(
                "reward",
                m.SummaryStats.from_list([ep.reward for ep in episodes]),
            ),
            m.Metric(
                "advantage",
                m.SummaryStats.from_list([ep.advantage for ep in episodes]),
            ),
            m.Metric("reward/group_std", m.Mean.from_list(group_stds)),
            m.Metric("reward/group_std", m.Max.from_list(group_stds)),
            m.Metric("reward/zero_std_frac", m.NoReduce(zero_std_frac)),
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

    @sl.log_trace_span("validate")
    async def validate(self) -> list[m.Metric]:
        """Run greedy validation on held-out prompts.

        Returns:
            Validation rollout metrics, generation metrics, and validation
            timing.
        """
        # TODO: investigate using pass@k for validation.
        t_validate_start = time.perf_counter()
        num_samples = self.config.num_validation_samples
        greedy = SamplingConfig(
            n=1,
            temperature=0.0,
            top_p=1.0,
            max_tokens=self.config.generator.sampling.max_tokens,
            stop_token_ids=self._stop_token_ids,
        )

        rollouts, generation_metrics = await self._run_rollouts(
            dataset=self._validation_dataset,
            num_groups=num_samples,
            group_size=1,
            sampling=greedy,
            step=0,
            group_offset=0,
            metrics_prefix="validation_generator",
        )

        if self.config.log_samples:
            preview = [
                rollout_to_episode(r, text=last_assistant_text(r))
                for r in rollouts
                if r.reward is not None
            ]
            _log_samples(preview)

        validation_metrics = prepare_rollout_metrics("validation", rollouts)
        validation_metrics.append(
            m.Metric("validation/num_samples", m.NoReduce(float(len(rollouts))))
        )
        validation_metrics += generation_metrics

        t_validate_s = time.perf_counter() - t_validate_start
        validation_metrics.append(m.Metric("timing/validate", m.NoReduce(t_validate_s)))
        return validation_metrics

    async def train(self):
        num_steps = self.config.num_steps
        num_groups = self.config.num_prompts_per_step
        logger.info(f"Pre-training validation; then {num_steps} steps of RL training")

        # collect validation metrics before training
        # so we can compare before/after
        pre_validation_metrics = await self.validate()
        self.metrics_processor.log(
            step=0,
            metrics=pre_validation_metrics,
            is_validation=True,
        )
        pre_validation_agg = m.MetricsProcessor._aggregate_metrics(
            pre_validation_metrics
        )

        sl.log_trace_instant("training_start")

        for step in range(1, num_steps + 1):
            sl.set_step(step)
            # Propagate the step counter to actors for structured logging.
            self.trainer.sync_log_step.call(step)
            self.generator.sync_log_step.call(step)
            # Cancellation point for Ctrl-C (KeyboardInterrupt) handling.
            # This yields to the event loop to check for cancellation, which
            # doesn't happen with `.get` calls.
            # TODO: investigate replacing `.get()` with `await
            await asyncio.sleep(0)

            t_step_start = time.perf_counter()

            # --- rollouts ---
            # Collect rollouts until total response tokens reach the
            # token budget. The Batcher then packs, truncates to
            # global_batch_size rows, and splits into microbatches.
            t_rollout_start = time.perf_counter()
            rollouts: list[Rollout] = []
            rollout_metrics: list[m.Metric] = []
            collected_tokens = 0
            group_offset = 0
            # num_tokens_target (= global_batch_size * seq_len) is the stop
            # condition for collected tokens before a train step can proceed.
            # NOTE: this is a proxy — packing adds padding to fill fixed-length
            # rows, so actual token consumption may exceed collected_tokens.
            num_tokens_target = self.batcher.num_tokens_target(self.trainer_dp_degree)
            while collected_tokens < num_tokens_target:
                new_rollouts, new_metrics = await self._collect_rollouts(
                    num_groups, step=step, group_offset=group_offset
                )
                rollouts.extend(new_rollouts)
                rollout_metrics.extend(new_metrics)
                # Both prompt length and completion length are counted.
                collected_tokens += sum(
                    len(t.prompt_token_ids) + len(t.response_token_ids) - 1
                    for r in new_rollouts
                    for t in r.turns
                )
                group_offset += num_groups

            episodes, episode_metrics = self._build_episodes(rollouts)
            t_rollout_s = time.perf_counter() - t_rollout_start

            if self.config.log_samples:
                _log_samples(episodes)

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
            fwd_bwd_metrics: dict[str, float] = {}
            for microbatch in microbatches:
                with sl.log_trace_span("trainer_forward_backward_call"):
                    mb_metrics = self._get_rank_0_value(
                        self.trainer.forward_backward.call(
                            microbatch, num_global_valid_tokens
                        ).get()
                    )
                    for k, v in mb_metrics.items():
                        if k not in fwd_bwd_metrics:
                            fwd_bwd_metrics[k] = v
                        elif k.endswith("/max"):
                            fwd_bwd_metrics[k] = max(fwd_bwd_metrics[k], v)
                        elif k.endswith(("/mean", "/frac")):
                            fwd_bwd_metrics[k] += v
            with sl.log_trace_span("trainer_optim_step_call"):
                optim_output = self._get_rank_0_value(
                    self.trainer.optim_step.call().get()
                )
            trainer_policy_version = optim_output.policy_version
            optimizer_metrics = optim_output.metrics
            t_train_s = time.perf_counter() - t_train_start

            # --- weight sync ---
            # TODO: we should have `push_model_state_dict` return `trainer_policy_version`
            # instead of having `trainer.optim_step` return it
            t_push_start = time.perf_counter()
            with sl.log_trace_span("trainer_push_model_state_dict"):
                self.trainer.push_model_state_dict.call().get()
            t_weight_sync_push_s = time.perf_counter() - t_push_start
            with sl.log_trace_span("generator_pull_model_state_dict"):
                self.generator.pull_model_state_dict.call(trainer_policy_version).get()
            t_weight_sync_total_s = time.perf_counter() - t_push_start
            t_step_s = time.perf_counter() - t_step_start
            # --- divergence check before any logging ---
            if not math.isfinite(fwd_bwd_metrics["loss/mean"]):
                logger.error("Loss is NaN/Inf; training diverged")
                break

            # --- Prepare metrics ---
            total_tokens = sum(
                len(ep.prompt_token_ids) + len(ep.token_ids) for ep in episodes
            )

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

            step_metrics.append(
                m.Metric("perf/tokens_per_second", m.NoReduce(total_tokens / t_step_s))
            )

            self.metrics_processor.log(
                step=step, metrics=step_metrics, is_validation=False
            )

        post_validation_metrics = await self.validate()
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


async def main():
    config = ConfigManager().parse_args()
    sl.init_structured_logger(
        source="rl_controller",
        output_dir=config.dump_folder,
        rank=0,
        # pyrefly: ignore [missing-attribute]
        enable=config.trainer.debug.enable_structured_logging,
    )
    sl.log_trace_instant("structured_logger_started")

    rl_trainer = config.build()
    try:
        await rl_trainer.setup_async()
        await rl_trainer.train()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Interrupted; attempting graceful shutdown...")
    finally:
        await rl_trainer.close()


if __name__ == "__main__":
    asyncio.run(main())
