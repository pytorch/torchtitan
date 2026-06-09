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
from dataclasses import dataclass, field

# must run before torch import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torchstore as ts
from monarch.actor import this_host
from monarch.spmd import setup_torch_elastic_env_async

from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import (
    CompileConfig,
    ConfigManager,
    Configurable,
    ParallelismConfig,
)
from torchtitan.experiments.rl.actors.generator import SamplingConfig, VLLMGenerator
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.types import (
    Completion,
    Episode,
    TrainingBatch,
    Trajectory,
)
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols.model_spec import ModelSpec

logger = logging.getLogger(__name__)


def _token_weighted_mean(
    values: torch.Tensor,
    *,
    num_tokens_by_sample: torch.Tensor,
    num_global_valid_tokens: torch.Tensor,
) -> torch.Tensor:
    """Each rank's share of the global token-weighted mean of `values`.

    Computed as `sum(values * num_tokens_by_sample) / num_global_valid_tokens`.
    SUM-reducing this share across the loss mesh reconstructs the global mean.
    """
    return (
        values * num_tokens_by_sample.to(values.dtype)
    ).sum() / num_global_valid_tokens


class GRPOLoss(Configurable):
    """Clipped GRPO surrogate loss.

    Takes per-sample response logprobs (already extracted from whatever
    packing or padding format the trainer uses).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        clip_eps: float = 0.2
        """PPO clipping epsilon for the probability ratio."""

    def __init__(self, config: Config):
        self.clip_eps = config.clip_eps

    def __call__(
        self,
        policy_logprobs: list[torch.Tensor],
        advantages: torch.Tensor,
        num_global_valid_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        response_lens = torch.tensor(
            [sample_logprobs.numel() for sample_logprobs in policy_logprobs],
            device=advantages.device,
            dtype=advantages.dtype,
        )

        per_sample_mean_logprobs = torch.stack(
            [sample_logprobs.mean() for sample_logprobs in policy_logprobs]
        )
        ratio = torch.exp(per_sample_mean_logprobs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
        # pg = policy gradient.
        sample_pg_losses = -torch.min(ratio * advantages, clipped_ratio * advantages)

        pg_loss = _token_weighted_mean(
            sample_pg_losses,
            num_tokens_by_sample=response_lens,
            num_global_valid_tokens=num_global_valid_tokens,
        )

        with torch.no_grad():
            clipped_frac = (torch.abs(ratio - clipped_ratio) > 1e-6).to(ratio.dtype)
            loss_metrics = {
                "loss/mean": pg_loss.detach(),
                "loss/ratio/mean": _token_weighted_mean(
                    ratio,
                    num_tokens_by_sample=response_lens,
                    num_global_valid_tokens=num_global_valid_tokens,
                ),
                "loss/ratio/clipped_frac": _token_weighted_mean(
                    clipped_frac,
                    num_tokens_by_sample=response_lens,
                    num_global_valid_tokens=num_global_valid_tokens,
                ),
            }

        return pg_loss, loss_metrics


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


def _log_samples(items: list[Episode] | list[Completion]) -> None:
    """Log the first sample per prompt for debugging."""
    seen_prompts: set[int] = set()
    for item in items:
        if item.prompt_idx in seen_prompts:
            continue
        seen_prompts.add(item.prompt_idx)
        reward_str = f" reward={item.reward:+.1f}" if hasattr(item, "reward") else ""
        logger.info(f"  [prompt {item.prompt_idx}]{reward_str}")
        logger.info(f"       A: {item.text[:300].replace(chr(10), ' ').strip()}")


def _prepare_reward_metrics(
    prefix: str,
    trajectories: list[Trajectory],
) -> list[m.Metric]:
    """One ``Mean`` metric per observed reward component across trajectories.

    Example::

        trajectories = [
            Trajectory(
                sample_idx=0,
                prompt_token_ids=p0,
                transitions=[(c0, Step(rewards={"correctness": 1.0, "format": 0.5}, done=True))],
            ),
            Trajectory(
                sample_idx=1,
                prompt_token_ids=p1,
                transitions=[(c1, Step(rewards={"correctness": 0.0}, done=True))],
            ),
        ]
        _prepare_reward_metrics("reward/component", trajectories)
        # -> [
        #      Metric("reward/component/correctness", Mean(sum=1.0, count=2)),  # 0.5
        #      Metric("reward/component/format",      Mean(sum=0.5, count=1)),  # 0.5 - "format" only in trajectory 0
        #    ]
    """
    values_by_name: dict[str, list[float]] = defaultdict(list)
    for trajectory in trajectories:
        for _completion, step in trajectory.transitions:
            for name, value in step.rewards.items():
                values_by_name[name].append(float(value))
    return [
        m.Metric(f"{prefix}/{name}", m.Mean.from_list(values))
        for name, values in sorted(values_by_name.items())
    ]


class RLTrainer(Configurable):
    """Top-level RL training orchestrator."""

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

        num_prompts_per_step: int = 5
        """Number of distinct prompts (= GRPO groups) drawn per training step.

        The total episodes per step is `num_prompts_per_step` * `group_size`,
        where `group_size` is `generator.sampling.n` (completions per prompt).
        """

        num_validation_samples: int = 20
        """Number of held-out prompts scored greedily (temp=0, n=1) per validation pass."""

        env: Configurable.Config = field(default=None)  # type: ignore[assignment]
        """Env config for training rollouts."""

        validation_env: Configurable.Config = field(default=None)  # type: ignore[assignment]
        """Env config for validation rollouts."""

        log_samples: bool = False
        """Log first completion per episode during training and validation."""

        compile: CompileConfig = field(default_factory=CompileConfig)
        """torch.compile config shared by trainer and generator."""

        trainer: PolicyTrainer.Config = field(
            default_factory=lambda: PolicyTrainer.Config(loss=GRPOLoss.Config())
        )
        """PolicyTrainer config. Controls optimizer, training, parallelism."""

        generator: VLLMGenerator.Config = field(default_factory=VLLMGenerator.Config)
        """VLLMGenerator actor configuration (vLLM engine, sampling)."""

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

    def __init__(self, config: Config):
        self.config = config
        self.trainer = None
        self.generator = None
        self._proc_meshes = []
        self.metrics_processor: m.MetricsProcessor = config.metrics.build(
            log_dir=config.dump_folder,
            job_config=config.to_dict(),
        )
        # TODO: Replace this single-turn tokenizer with renderer
        self.tokenizer = HuggingFaceTokenizer(tokenizer_path=config.hf_assets_path)

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

    @staticmethod
    @sl.log_trace_span("_collate_episodes")
    def _collate_episodes(episodes: list[Episode]) -> TrainingBatch:
        """Pack episodes into a single varlen-packed TrainingBatch."""
        all_ids: list[int] = []
        prompt_lens: list[int] = []
        response_lens: list[int] = []

        for ep in episodes:
            all_ids.extend(ep.prompt_token_ids + ep.token_ids)
            prompt_lens.append(len(ep.prompt_token_ids))
            response_lens.append(len(ep.token_ids))

        return TrainingBatch(
            token_ids=torch.tensor([all_ids], dtype=torch.long),
            prompt_lens=prompt_lens,
            response_lens=response_lens,
            seq_lens=[p + r for p, r in zip(prompt_lens, response_lens, strict=True)],
            advantages=torch.tensor(
                [ep.advantage for ep in episodes],
                dtype=torch.float32,
            ),
            token_logprobs=[ep.token_logprobs for ep in episodes],
        )

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

    @sl.log_trace_span("_collect_rollouts")
    def _collect_rollouts(
        self,
        num_groups: int,
        step: int,
    ) -> tuple[list[Trajectory], list[m.Metric]]:
        """Collect group rollouts and emit completion-shape rollout metrics."""
        envs = [
            self.config.env.build(step=step, group_idx=i) for i in range(num_groups)
        ]
        # TODO: Add a check max_tokens = min(max_tokens, context_window - model_input.length)
        # and pass max_tokens to the generator call or skip the call if max_tokens<=0.
        # Do the same for validation.
        tokenized_prompts = [
            self.tokenizer.encode(env.prompt, add_bos=True, add_eos=False)
            for env in envs
        ]
        completions, generation_metrics = self._get_rank_0_value(
            self.generator.generate.call(tokenized_prompts).get()
        )

        trajectories: list[Trajectory] = []
        with sl.log_trace_span("score"):
            for c in completions:
                step_result = envs[c.prompt_idx].step(c.text)
                trajectories.append(
                    Trajectory(
                        sample_idx=c.prompt_idx,
                        prompt_token_ids=tokenized_prompts[c.prompt_idx],
                        transitions=[(c, step_result)],
                    )
                )

        # Metrics
        response_lens = [len(c.token_ids) for c in completions]
        prompt_lens = [len(t.prompt_token_ids) for t in trajectories]
        total_lens = [p + r for p, r in zip(prompt_lens, response_lens, strict=True)]
        truncated = [c.finish_reason == "length" for c in completions]
        rollout_metrics: list[m.Metric] = [
            m.Metric("rollout/response_length", m.Mean.from_list(response_lens)),
            m.Metric("rollout/response_length", m.Max.from_list(response_lens)),
            m.Metric("rollout/prompt_length", m.Mean.from_list(prompt_lens)),
            m.Metric("rollout/prompt_length", m.Max.from_list(prompt_lens)),
            m.Metric("rollout/total_length", m.Max.from_list(total_lens)),
            m.Metric("rollout/truncation_rate", m.Mean.from_list(truncated)),
        ]
        rollout_metrics += generation_metrics
        rollout_metrics += _prepare_reward_metrics(
            prefix="reward/component", trajectories=trajectories
        )
        return trajectories, rollout_metrics

    @staticmethod
    @sl.log_trace_span("_build_episodes")
    def _build_episodes(
        trajectories: list[Trajectory],
    ) -> tuple[list[Episode], list[m.Metric]]:
        """Group trajectories by sample, apply mean-baseline advantage, emit metrics."""
        groups: dict[int, list[Trajectory]] = {}
        for t in trajectories:
            groups.setdefault(t.sample_idx, []).append(t)

        episodes: list[Episode] = []
        group_stds: list[float] = []
        for sample_idx, group in groups.items():
            rewards = [t.total_reward for t in group]
            group_mean = sum(rewards) / len(rewards)
            # Population standard deviation; NaN for an empty group.
            group_stds.append(statistics.pstdev(float(r) for r in rewards))
            for t in group:
                # Single-turn: exactly one (completion, step) per trajectory.
                c, _ = t.transitions[0]
                episodes.append(
                    Episode(
                        policy_version=c.policy_version,
                        prompt_idx=sample_idx,
                        prompt_token_ids=t.prompt_token_ids,
                        text=c.text,
                        token_ids=c.token_ids,
                        token_logprobs=c.token_logprobs,
                        reward=t.total_reward,
                        advantage=t.total_reward - group_mean,
                    )
                )

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
        """Run validation on held-out prompts using greedy sampling.

        TODO: investigate using pass@k.
        """
        t_validate_start = time.perf_counter()
        num_samples = self.config.num_validation_samples
        envs = [
            self.config.validation_env.build(step=0, group_idx=i)
            for i in range(num_samples)
        ]
        greedy = SamplingConfig(
            n=1,
            temperature=0.0,
            top_p=1.0,
            max_tokens=self.config.generator.sampling.max_tokens,
        )

        tokenized_prompts: list[list[int]] = [
            self.tokenizer.encode(env.prompt, add_bos=True, add_eos=False)
            for env in envs
        ]
        completions, generation_metrics = self._get_rank_0_value(
            self.generator.generate.call(
                tokenized_prompts,
                sampling_config=greedy,
                metrics_prefix="validation_generator",
            ).get()
        )

        trajectories = [
            Trajectory(
                sample_idx=i,
                prompt_token_ids=tokenized_prompts[i],
                transitions=[(c, envs[i].step(c.text))],
            )
            for i, c in enumerate(completions)
        ]

        if self.config.log_samples:
            _log_samples(completions)

        validation_metrics: list[m.Metric] = [
            m.Metric(
                "validation/reward",
                m.SummaryStats.from_list([t.total_reward for t in trajectories]),
            ),
            m.Metric(
                "validation/response_length",
                m.Mean.from_list([len(c.token_ids) for c in completions]),
            ),
            m.Metric("validation/num_samples", m.NoReduce(float(len(trajectories)))),
        ]
        validation_metrics += generation_metrics
        validation_metrics += _prepare_reward_metrics(
            prefix="validation/reward/component", trajectories=trajectories
        )

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
            t_rollout_start = time.perf_counter()
            trajectories, rollout_metrics = self._collect_rollouts(
                num_groups, step=step
            )
            episodes, episode_metrics = self._build_episodes(trajectories)
            t_rollout_s = time.perf_counter() - t_rollout_start

            if self.config.log_samples:
                _log_samples(episodes)

            # --- train ---
            t_train_start = time.perf_counter()
            batches = [
                self._collate_episodes(per_rank_episodes)
                for per_rank_episodes in self._shard_episodes(episodes)
            ]
            # Controller has all episodes pre-shard, so it computes
            # the global response-token count instead of an all-reduce.
            num_global_valid_tokens = sum(len(ep.token_ids) for ep in episodes)
            with sl.log_trace_span("trainer_forward_backward_call"):
                fwd_bwd_metrics = self._get_rank_0_value(
                    self.trainer.forward_backward.call(
                        batches,
                        num_global_valid_tokens=num_global_valid_tokens,
                    ).get()
                )
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

            # Actor metrics are already globally reduced; NoReduce passes them through.
            step_metrics += [
                m.Metric(k, m.NoReduce(v)) for k, v in fwd_bwd_metrics.items()
            ]
            step_metrics += [
                m.Metric(k, m.NoReduce(v)) for k, v in optimizer_metrics.items()
            ]

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

    rl_trainer = RLTrainer(config)
    try:
        await rl_trainer.setup_async()
        await rl_trainer.train()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Interrupted; attempting graceful shutdown...")
    finally:
        await rl_trainer.close()


if __name__ == "__main__":
    asyncio.run(main())
