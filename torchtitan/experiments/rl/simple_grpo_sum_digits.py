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
python3 torchtitan/experiments/rl/simple_grpo_sum_digits.py \
    --module rl --config rl_grpo_qwen3_0_6b \
    --hf_assets_path=<path_to_model_checkpoint>
"""

import asyncio
import logging
import math
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import torch
import torchstore as ts
from monarch.actor import this_host
from monarch.spmd import setup_torch_elastic_env_async

from torchtitan.config import Configurable, ParallelismConfig
from torchtitan.config.manager import ConfigManager
from torchtitan.experiments.rl.actors.generator import VLLMGenerator
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.sum_digits import format_reward, SumDigitsEnv
from torchtitan.experiments.rl.types import Completion, Episode, TrainBatch, Trajectory
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
    ) -> tuple[torch.Tensor, dict[str, float]]:
        per_sample_mean_lps = []
        for policy_lps in policy_logprobs:
            per_sample_mean_lps.append(policy_lps.mean())

        mean_log_ratio = torch.stack(per_sample_mean_lps)
        ratio = torch.exp(mean_log_ratio)

        unclipped_loss = ratio * advantages
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
        clipped_loss = clipped_ratio * advantages
        pg_loss = -torch.min(unclipped_loss, clipped_loss).mean()

        metrics = {
            "pg_loss": pg_loss.item(),
            "ratio_mean": ratio.mean().item(),
            "ratio_clipped_frac": (torch.abs(ratio - clipped_ratio) > 1e-6)
            .float()
            .mean()
            .item(),
        }
        return pg_loss, metrics


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

        log_samples: bool = False
        """Log first completion per episode during training and eval."""

        trainer: PolicyTrainer.Config = field(
            default_factory=lambda: PolicyTrainer.Config(loss=GRPOLoss.Config())
        )
        """PolicyTrainer config. Controls optimizer, training, parallelism."""

        generator: VLLMGenerator.Config = field(default_factory=VLLMGenerator.Config)
        """VLLMGenerator actor configuration (vLLM engine, sampling)."""

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
        self._train_rng = random.Random(42)
        self._proc_meshes = []

    async def cleanup(self):
        """Stop all proc meshes to release GPU memory."""
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
            seq_lens=[p + r for p, r in zip(prompt_lens, response_lens)],
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
        )

        # Generator uses RL-specific parallelize (TP-only, no FSDP, vLLM-compatible)
        from torchtitan.experiments.rl.models.parallelize import parallelize_qwen3

        gen_model_spec = replace(config.model_spec, parallelize_fn=parallelize_qwen3)
        self.generator = generator_mesh.spawn(
            "generator",
            VLLMGenerator,
            config.generator,
            model_spec=gen_model_spec,
            model_path=config.hf_assets_path,
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

    def _collect_rollout(self, num_prompts: int) -> list[Trajectory]:
        """Collect group rollouts: one single-use env per prompt, scored and returned.

        ``generate`` returns ``num_prompts * group_size`` completions
        ordered by ``prompt_idx`` (see ``VLLMGenerator.generate``), so
        completions for prompt i sit at indices ``[i*group_size, (i+1)*group_size)``.
        Each env is bound to one problem, so its group of completions
        all step against the same target.
        """
        group_size = self.config.generator.sampling.n
        envs = [SumDigitsEnv(self._train_rng) for _ in range(num_prompts)]
        completions = self._get_rank_0_value(
            self.generator.generate.call([env.prompt for env in envs]).get()
        )

        trajectories: list[Trajectory] = []
        for sample_idx, env in enumerate(envs):
            # Slice out this env's group_size completions and score each against its env.
            group = completions[sample_idx * group_size : (sample_idx + 1) * group_size]
            for c in group:
                step = env.step(c.text)
                trajectories.append(
                    Trajectory(sample_idx=sample_idx, transitions=[(c, step)])
                )
        return trajectories

    @staticmethod
    def _build_episodes(trajectories: list[Trajectory]) -> list[Episode]:
        """Group trajectories by sample, apply mean-baseline advantage, flatten to Episodes."""
        groups: dict[int, list[Trajectory]] = {}
        for t in trajectories:
            groups.setdefault(t.sample_idx, []).append(t)

        episodes: list[Episode] = []
        for sample_idx, group in groups.items():
            rewards = [t.total_reward for t in group]
            group_mean = sum(rewards) / len(rewards)
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
        return episodes

    async def evaluate(self, num_samples: int = 20) -> dict:
        """Run evaluation on held-out prompts.

        Uses the generator's default (training) sampling config and
        scores the first sample per prompt -- preserves the
        pre-refactor eval signal.

        TODO: report both greedy (temperature=0, n=1) and pass@k
        (training temp, n=k) so mode-sharpening vs. coverage
        regressions are both visible. A single-sample score hides
        both signals.

        Args:
            num_samples: Number of eval prompts to generate.

        Returns:
            Dict with accuracy, correct, total, format_rate, format_ok.
        """
        eval_rng = random.Random(99)
        envs = [SumDigitsEnv(eval_rng) for _ in range(num_samples)]
        completions = self._get_rank_0_value(
            self.generator.generate.call([env.prompt for env in envs]).get()
        )

        # Score the first completion per prompt via the env. Completions come
        # back in prompt_idx order, length num_samples * n.
        group_size = self.config.generator.sampling.n
        correct = 0
        format_ok = 0
        for i, env in enumerate(envs):
            c = completions[i * group_size]
            step = env.step(c.text)
            # Correct iff reward includes the +1.0 correctness component.
            correct += int(step.reward >= 1.0)
            format_ok += int(format_reward(c.text) > 0)

        if self.config.log_samples:
            _log_samples(completions)

        result = {
            "accuracy": correct / num_samples,
            "correct": correct,
            "total": num_samples,
            "format_rate": format_ok / num_samples,
            "format_ok": format_ok,
        }
        logger.info(
            f"Eval: Accuracy={result['accuracy']:.0%} ({correct}/{num_samples}) "
            f"Format={result['format_rate']:.0%} ({format_ok}/{num_samples})"
        )
        return result

    async def train(self):
        num_steps = self.config.num_steps
        logger.info(f"Pre-training eval; then {num_steps} steps of RL training")
        pre_eval = await self.evaluate()

        for step in range(num_steps):
            step_start = time.perf_counter()

            # --- Collect data and create episodes --- #
            trajectories = self._collect_rollout(self.config.num_prompts_per_step)
            episodes = self._build_episodes(trajectories)

            if self.config.log_samples:
                _log_samples(episodes)

            # --- Train step --- #
            batches = [
                self._collate_episodes(per_rank_episodes)
                for per_rank_episodes in self._shard_episodes(episodes)
            ]
            fwd_bwd_metrics = self._get_rank_0_value(
                self.trainer.forward_backward.call(batches).get()
            )
            optim_metrics = self._get_rank_0_value(self.trainer.optim_step.call().get())
            metrics = {**fwd_bwd_metrics, **optim_metrics}

            # --- Weight sync --- #
            t0 = time.perf_counter()
            self.trainer.push_model_state_dict.call().get()
            t_push = time.perf_counter() - t0
            self.generator.pull_model_state_dict.call(metrics["policy_version"]).get()
            t_sync = time.perf_counter() - t0
            logger.info(f"Weight sync: push={t_push:.3f}s, total={t_sync:.3f}s")

            # --- Logging --- #
            token_lens = [len(ep.token_ids) for ep in episodes]
            rewards = [ep.reward for ep in episodes]
            logger.info(
                f"Step {step:2d} | Loss: {metrics['loss']:+.4f} | "
                f"Reward: {sum(rewards) / len(rewards):+.3f} | "
                f"Correct: {sum(1 for r in rewards if r >= 1.0):>2}/{len(rewards)} | "
                f"Avg tokens: {sum(token_lens) / len(token_lens):>3.0f} | "
                f"Logprob diff: mean={metrics['logprob_diff_mean']:.4e}, "
                f"max={metrics['logprob_diff_max']:.4e} | "
                f"Time: {time.perf_counter() - step_start:.1f}s"
            )

            if not math.isfinite(metrics["loss"]):
                logger.error("Loss is NaN/Inf; training diverged")
                break

        logger.info("Post-training eval")
        post_eval = await self.evaluate()
        logger.info(
            f"Pre:  acc={pre_eval['accuracy']:.0%} ({pre_eval['correct']}/{pre_eval['total']}) "
            f"format={pre_eval['format_rate']:.0%} ({pre_eval['format_ok']}/{pre_eval['total']})\n"
            f"Post: acc={post_eval['accuracy']:.0%} ({post_eval['correct']}/{post_eval['total']}) "
            f"format={post_eval['format_rate']:.0%} ({post_eval['format_ok']}/{post_eval['total']})"
        )


async def main():
    """Run the distributed RL training loop using Monarch."""
    config = ConfigManager().parse_args()
    rl_trainer = RLTrainer(config)
    await rl_trainer.setup()
    await rl_trainer.train()


if __name__ == "__main__":
    asyncio.run(main())
