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
import time
from collections.abc import Callable
from dataclasses import dataclass, field

# must run before torch import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torchstore as ts
from monarch.actor import this_host
from monarch.spmd import setup_torch_elastic_env_async

from torchtitan.config import Configurable, ParallelismConfig
from torchtitan.config.configs import CompileConfig
from torchtitan.config.manager import ConfigManager
from torchtitan.experiments.rl.actors.generator import SamplingConfig, VLLMGenerator
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.loss.types import (
    LossOutput,
    sequence_scalar_token_weighted_sum,
)
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.observability.grpo_metrics import (
    _population_std,
    build_reward_component_metrics,
)
from torchtitan.experiments.rl.types import (
    Completion,
    Episode,
    Step,
    TrainBatch,
    Trajectory,
)
from torchtitan.protocols.model_spec import ModelSpec

logger = logging.getLogger(__name__)


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
    ) -> LossOutput:
        response_lens = torch.tensor(
            [policy_lps.numel() for policy_lps in policy_logprobs],
            device=advantages.device,
            dtype=advantages.dtype,
        )

        per_sample_mean_lps = torch.stack(
            [policy_lps.mean() for policy_lps in policy_logprobs]
        )
        ratio = torch.exp(per_sample_mean_lps)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
        sample_pg_losses = -torch.min(ratio * advantages, clipped_ratio * advantages)
        pg_loss = sample_pg_losses.mean()

        with torch.no_grad():
            clipped_frac = (torch.abs(ratio - clipped_ratio) > 1e-6).to(ratio.dtype)
            token_mean_metric_sums = {
                "loss/total": sequence_scalar_token_weighted_sum(
                    sample_pg_losses, response_lens
                ),
                "loss/ratio/mean": sequence_scalar_token_weighted_sum(
                    ratio, response_lens
                ),
                "loss/ratio/clipped_frac": sequence_scalar_token_weighted_sum(
                    clipped_frac, response_lens
                ),
            }
            num_local_valid_tokens = response_lens.sum().detach()

        return LossOutput(
            loss=pg_loss,
            token_mean_metric_sums=token_mean_metric_sums,
            num_local_valid_tokens=num_local_valid_tokens,
        )


class Provisioner:
    """Allocates non-overlapping GPU ranges for Monarch proc meshes.

    In non-colocated mode, the trainer and generator run on separate GPU
    meshes (e.g. GPUs 0-3 for training, GPUs 4-7 for generation). Each
    call to ``allocate(n)`` reserves the next *n* GPUs and returns a
    bootstrap callable that sets ``CUDA_VISIBLE_DEVICES`` before CUDA
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


# Default console keys per context. Override with
# `--metrics.train-console-allow-list=.*` or
# `--metrics.validation-console-allow-list=.*` to print all console keys.
_RL_TRAIN_HEADLINE_METRIC_PATTERNS = [
    r"^loss/total$",
    r"^loss/ratio/clipped_frac$",
    r"^reward/_mean$",
    r"^reward/_max$",
    r"^reward/zero_std_frac$",
    r"^rollout/response_length/max$",
    r"^train/grad_norm/mean$",
    r"^train/lr$",
    r"^perf/tokens_per_second$",
    r"^timing/step$",
]
_RL_VALIDATION_HEADLINE_METRIC_PATTERNS = [
    r"^validation/reward/_mean$",
    r"^validation/reward/_max$",
    r"^validation/response_length/mean$",
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

        The total episodes per step is ``num_prompts_per_step * group_size``,
        where ``group_size`` is ``generator.sampling.n`` (completions per prompt).
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

        metrics: m.MetricsConfig = field(
            default_factory=lambda: m.MetricsConfig(
                train_console_allow_list=list(_RL_TRAIN_HEADLINE_METRIC_PATTERNS),
                validation_console_allow_list=list(
                    _RL_VALIDATION_HEADLINE_METRIC_PATTERNS
                ),
            )
        )

        def __post_init__(self):
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
        self._proc_meshes = []
        self.metric_logger: m.MetricLogger | None = None

    async def cleanup(self):
        """Stop all proc meshes to release GPU memory and close metric backends."""
        if self.metric_logger is not None:
            try:
                self.metric_logger.close()
            except Exception:
                logger.exception("metric_logger close failed")
            self.metric_logger = None
        for mesh in self._proc_meshes:
            try:
                await mesh.stop()
            except Exception:
                pass
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
    def _collate_episodes(episodes: list[Episode]) -> TrainBatch:
        """Pack episodes into a single varlen-packed TrainBatch."""
        all_ids: list[int] = []
        prompt_lens: list[int] = []
        response_lens: list[int] = []

        for ep in episodes:
            all_ids.extend(ep.prompt_token_ids + ep.token_ids)
            prompt_lens.append(len(ep.prompt_token_ids))
            response_lens.append(len(ep.token_ids))

        return TrainBatch(
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

    async def setup(
        self,
        *,
        host_mesh=None,
        trainer_nodes: int | None = None,
        generator_nodes: int | None = None,
        gpus_per_node: int | None = None,
    ):
        """Spawn Monarch actors on separate meshes and initialize weights.

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
        )

        # Initialize TorchStore for weight sync between trainer and generator.
        # StorageVolumes are spawned on the trainer mesh so they are colocated
        # with the weight source for faster data access in the non-RDMA path.
        # LocalRankStrategy: routes each process to a storage volume based on
        #   LOCAL_RANK, so colocated processes share the same volume.
        # https://github.com/meta-pytorch/torchstore
        await ts.initialize(mesh=trainer_mesh, strategy=ts.LocalRankStrategy())

        # push weights from trainer
        self.trainer.push_model_state_dict.call().get()
        # pull weights for policy version 0 (initial weights)
        self.generator.pull_model_state_dict.call(0).get()

        self.metric_logger = m.MetricLogger.build(
            config.metrics,
            log_dir=config.dump_folder,
            config_dict=config.to_dict(),
        )

    def _collect_rollouts(
        self,
        num_groups: int,
        step: int,
    ) -> tuple[list[Trajectory], list[m.Metric]]:
        """Collect group rollouts and emit completion-shape rollout metrics."""
        envs = [
            self.config.env.build(step=step, group_idx=i) for i in range(num_groups)
        ]
        completions = self._get_rank_0_value(
            self.generator.generate.call([env.prompt for env in envs]).get()
        )

        trajectories: list[Trajectory] = []
        steps: list[Step] = []
        for c in completions:
            step_result = envs[c.prompt_idx].step(c.text)
            steps.append(step_result)
            trajectories.append(
                Trajectory(sample_idx=c.prompt_idx, transitions=[(c, step_result)])
            )

        response_lens = [len(c.token_ids) for c in completions]
        prompt_lens = [len(c.prompt_token_ids) for c in completions]
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
        rollout_metrics += build_reward_component_metrics("reward/component", steps)
        return trajectories, rollout_metrics

    @staticmethod
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
            group_stds.append(_population_std([float(r) for r in rewards]))
            for t in group:
                # Single-turn: exactly one (completion, step) per trajectory.
                c, _ = t.transitions[0]
                episodes.append(
                    Episode(
                        policy_version=c.policy_version,
                        prompt_idx=sample_idx,
                        prompt_token_ids=c.prompt_token_ids,
                        text=c.text,
                        token_ids=c.token_ids,
                        token_logprobs=c.token_logprobs,
                        reward=t.total_reward,
                        advantage=t.total_reward - group_mean,
                    )
                )

        num_groups = len(group_stds)
        zero_std_frac = (
            sum(1 for s in group_stds if s == 0.0) / num_groups if num_groups else 0.0
        )
        episode_metrics: list[m.Metric] = [
            m.Metric(
                "reward",
                m.Stats.from_list([ep.reward for ep in episodes]),
            ),
            m.Metric(
                "advantage",
                m.Stats.from_list([ep.advantage for ep in episodes]),
            ),
            m.Metric("reward/group_std", m.Mean.from_list(group_stds)),
            m.Metric("reward/group_std", m.Max.from_list(group_stds)),
            m.Metric("reward/zero_std_frac", m.NoReduce(zero_std_frac)),
        ]
        return episodes, episode_metrics

    async def validate(self) -> list[m.Metric]:
        """Run validation on held-out prompts using greedy sampling.

        TODO: investigate using pass@k.
        """
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
        completions = self._get_rank_0_value(
            self.generator.generate.call(
                [env.prompt for env in envs], sampling_config=greedy
            ).get()
        )

        steps = [env.step(completions[i].text) for i, env in enumerate(envs)]

        if self.config.log_samples:
            _log_samples(completions)

        validation_metrics: list[m.Metric] = [
            m.Metric("validation/reward", m.Stats.from_list([s.reward for s in steps])),
            m.Metric(
                "validation/response_length",
                m.Mean.from_list([len(c.token_ids) for c in completions]),
            ),
            m.Metric("validation/num_samples", m.NoReduce(float(len(steps)))),
        ]
        validation_metrics += build_reward_component_metrics(
            "validation/reward/component", steps
        )
        return validation_metrics

    async def train(self):
        num_steps = self.config.num_steps
        num_groups = self.config.num_prompts_per_step
        logger.info(f"Pre-training validation; then {num_steps} steps of RL training")

        pre_metrics = await self.validate()
        self.metric_logger.log(
            step=0,
            metrics=pre_metrics,
            console_allow_list=self.config.metrics.validation_console_allow_list,
            console_prefix="validate ",
        )

        for step in range(num_steps):
            step_start = time.perf_counter()

            # --- rollouts ---
            rollout_start = time.perf_counter()
            trajectories, rollout_metrics = self._collect_rollouts(
                num_groups, step=step
            )
            episodes, episode_metrics = self._build_episodes(trajectories)
            rollout_s = time.perf_counter() - rollout_start

            if self.config.log_samples:
                _log_samples(episodes)

            # --- train ---
            train_start = time.perf_counter()
            batches = [
                self._collate_episodes(per_rank_episodes)
                for per_rank_episodes in self._shard_episodes(episodes)
            ]
            fwd_metrics = self._get_rank_0_value(
                self.trainer.forward_backward.call(batches).get()
            )
            optim_metrics = self._get_rank_0_value(self.trainer.optim_step.call().get())
            train_s = time.perf_counter() - train_start

            # --- weight sync ---
            push_start = time.perf_counter()
            self.trainer.push_model_state_dict.call().get()
            weight_sync_push_s = time.perf_counter() - push_start
            self.generator.pull_model_state_dict.call(
                optim_metrics["train/policy_version/mean"]
            ).get()
            weight_sync_total_s = time.perf_counter() - push_start

            # --- divergence check before any logging ---
            if not math.isfinite(fwd_metrics["loss/total"]):
                logger.error("Loss is NaN/Inf; training diverged")
                break

            step_s = time.perf_counter() - step_start
            total_tokens = sum(
                len(ep.prompt_token_ids) + len(ep.token_ids) for ep in episodes
            )

            step_metrics: list[m.Metric] = []
            # Actor metrics are already globally reduced; NoReduce passes them through.
            step_metrics += [m.Metric(k, m.NoReduce(v)) for k, v in fwd_metrics.items()]
            step_metrics += [
                m.Metric(k, m.NoReduce(v)) for k, v in optim_metrics.items()
            ]
            step_metrics += rollout_metrics
            step_metrics += episode_metrics
            for key, value in [
                ("timing/step", step_s),
                ("timing/rollout", rollout_s),
                ("timing/train", train_s),
                ("timing/weight_sync/push", weight_sync_push_s),
                ("timing/weight_sync/total", weight_sync_total_s),
            ]:
                step_metrics.append(m.Metric(key, m.NoReduce(value)))
            step_metrics.append(
                m.Metric("perf/tokens_per_second", m.NoReduce(total_tokens / step_s))
            )

            self.metric_logger.log(
                step=step,
                metrics=step_metrics,
                console_allow_list=self.config.metrics.train_console_allow_list,
            )

        post_metrics = await self.validate()
        self.metric_logger.log(
            step=num_steps,
            metrics=post_metrics,
            console_allow_list=self.config.metrics.validation_console_allow_list,
            console_prefix="validate ",
        )


async def main():
    """Run the distributed RL training loop using Monarch."""
    config = ConfigManager().parse_args()
    rl_trainer = RLTrainer(config)
    await rl_trainer.setup()
    await rl_trainer.train()


if __name__ == "__main__":
    asyncio.run(main())
