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
3. Separate scoring component for reward and advantage computation

The architecture mirrors monarch's grpo_actor.py but adapted for vLLM rollouts + TorchTitan training.

Command to run:
python3 torchtitan/experiments/rl/sync_grpo_sum_digits/main.py \
    --config rl_grpo_qwen3_0_6b \
    --hf_assets_path=<path_to_model_checkpoint>
"""

import asyncio
import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field

import torch
import torchstore as ts
from monarch.actor import this_host
from monarch.spmd import setup_torch_elastic_env_async

from torchtitan.config import Configurable
from torchtitan.config.manager import ConfigManager
from torchtitan.experiments.rl.actors.generator import VLLMGenerator
from torchtitan.experiments.rl.actors.grader import Grader
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.controller import create_meshes
from torchtitan.experiments.rl.sync_grpo_sum_digits.task import (
    extract_answer,
    SumDigitsTask,
)
from torchtitan.experiments.rl.types import (
    Completion,
    Episode,
    ScoredCompletion,
    TrainBatch,
)
from torchtitan.protocols.model_spec import ModelSpec

logger = logging.getLogger(__name__)


class GRPOLoss:
    """GRPO loss with DAPO-style token-level normalization.

    KL is disabled in this synchronous recipe. The rollout logprobs are used
    as the clipping anchor.
    """

    def __init__(
        self,
        kl_coef: float = 0.0,
        clip_eps: float = 0.2,
    ):
        self.kl_coef = kl_coef
        self.clip_eps = clip_eps

    def __call__(
        self,
        policy_logprobs: torch.Tensor,
        response_mask: torch.Tensor,
        advantages: torch.Tensor,
        old_logprobs: torch.Tensor,
        **_,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        device = policy_logprobs.device
        old_logprobs = old_logprobs.to(device).detach()
        advantages = advantages.to(device)
        response_mask = response_mask.to(device)

        log_ratio = policy_logprobs - old_logprobs
        ratio = torch.exp(log_ratio)
        per_token_adv = advantages.unsqueeze(1).expand_as(policy_logprobs)

        clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
        pg_loss = -torch.min(ratio * per_token_adv, clipped * per_token_adv)

        num_response_tokens = response_mask.sum().clamp(min=1.0)
        pg_loss = (pg_loss * response_mask).sum() / num_response_tokens
        kl_div = ((ratio - 1 - log_ratio) * response_mask).sum() / num_response_tokens

        loss = pg_loss + self.kl_coef * kl_div
        metrics = {
            "pg_loss": pg_loss.item(),
            "kl_div": kl_div.item(),
            "ratio_mean": (ratio * response_mask).sum().item()
            / num_response_tokens.item(),
        }
        return loss, metrics


def compute_advantages(group: list[ScoredCompletion]) -> list[float]:
    """Compute GRPO advantages for a single group."""
    mean_reward = sum(sc.reward for sc in group) / len(group)
    return [sc.reward - mean_reward for sc in group]


def _log_group_sample(
    group: list[ScoredCompletion],
    expected_answer: str,
) -> None:
    """Log the first completion in a group for debugging."""
    sc = group[0]
    extracted = extract_answer(sc.completion.text)
    is_correct = extracted == int(expected_answer) if expected_answer else None
    mark = "+" if is_correct else "-"
    logger.info(
        f"  [{mark}] expected={expected_answer} extracted={extracted} "
        f"reward={sc.reward:+.1f}"
    )
    logger.info(
        f"       A: {sc.completion.text[:300].replace(chr(10), ' ').strip()}"
    )


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

        batch_invariant_mode: bool = True
        """Enable batch-invariant mode for deterministic NCCL collective
        operations and bitwise-reproducible forward/backward passes."""

        num_episodes_per_step: int = 5
        """Number of episodes to create before every training step."""

        log_samples: bool = False
        """Log first completion per episode during training and eval."""

        trainer: PolicyTrainer.Config = field(default_factory=PolicyTrainer.Config)
        """PolicyTrainer config. Controls optimizer, training, parallelism"""

        generator: VLLMGenerator.Config = field(default_factory=VLLMGenerator.Config)
        """VLLMGenerator actor configuration (vLLM engine, sampling)."""

    def __init__(self, config: Config):
        self.config = config

        config.trainer.model_spec = config.model_spec
        config.trainer.hf_assets_path = config.hf_assets_path
        config.trainer.dump_folder = config.dump_folder
        config.generator.model_spec = config.model_spec
        config.generator.hf_assets_path = config.hf_assets_path
        config.generator.batch_invariant_mode = config.batch_invariant_mode

        self.task = SumDigitsTask(seed=42)
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
    def _compute_world_size(p: "ParallelismConfig") -> int:
        """Compute world size from all parallel dimensions."""
        dp_shard = max(p.data_parallel_shard_degree, 1)
        return (
            p.data_parallel_replicate_degree
            * dp_shard
            * p.tensor_parallel_degree
            * p.pipeline_parallel_degree
            * p.context_parallel_degree
        )

    @staticmethod
    def _validate_policy_versions(episodes: list[Episode]) -> int:
        if not episodes:
            raise ValueError("episodes must be non-empty")

        policy_versions = {ep.policy_version for ep in episodes}
        if len(policy_versions) != 1:
            raise ValueError(
                "all episodes in a synchronous GRPO step must share one policy_version, "
                f"got {sorted(policy_versions)}"
            )
        return next(iter(policy_versions))

    def _shard_episodes(self, episodes: list[Episode]) -> list[list[Episode]]:
        """Shard episodes across trainer DP ranks with interleaved indexing."""
        self._validate_policy_versions(episodes)
        return [
            [episodes[i] for i in range(rank, len(episodes), self.trainer_dp_degree)]
            for rank in range(self.trainer_dp_degree)
        ]

    @staticmethod
    def _collate_rank_episodes(
        episodes: list[Episode],
        *,
        policy_version: int,
        pad_token_id: int = 0,
    ) -> TrainBatch:
        """Collate one DP rank's episodes into a padded TrainBatch."""
        if not episodes:
            raise ValueError("episodes must be non-empty")

        prompt_lens = [len(ep.prompt_tokens) for ep in episodes]
        response_lens = [len(ep.response_tokens) for ep in episodes]
        max_len = max(p + r for p, r in zip(prompt_lens, response_lens))

        padded_ids = []
        padded_old_logprobs = []
        for ep in episodes:
            seq = ep.prompt_tokens + ep.response_tokens
            pad_len = max_len - len(seq)
            padded_ids.append(seq + [pad_token_id] * pad_len)
            padded_old_logprobs.append(
                [0.0] * len(ep.prompt_tokens) + ep.logprobs + [0.0] * pad_len
            )

        return TrainBatch(
            token_ids=torch.tensor(padded_ids, dtype=torch.long),
            prompt_lens=torch.tensor(prompt_lens, dtype=torch.long),
            response_lens=torch.tensor(response_lens, dtype=torch.long),
            advantages=torch.tensor(
                [ep.advantage for ep in episodes],
                dtype=torch.float32,
            ),
            old_logprobs=torch.tensor(
                padded_old_logprobs,
                dtype=torch.float32,
            ),
            policy_version=policy_version,
            pad_token_id=pad_token_id,
        )

    async def setup(
        self,
        *,
        host_mesh=None,
        trainer_nodes: int = 0,
        generator_nodes: int = 0,
        gpus_per_node: int | None = None,
    ):
        """Spawn Monarch actors on separate meshes and initialize weights.

        Creates separate GPU meshes for trainer and generator, a CPU mesh for
        the grader, and synchronizes initial weights from trainer to generator.
        Must be called before :meth:`train`.

        Args:
            host_mesh: HostMesh to partition. Defaults to this_host() for
                single-node. For multi-node, pass a HostMesh from a Job
                API or WorkerConnection.
            trainer_nodes: Nodes dedicated to the trainer. 0 (default)
                means trainer and generator share the same host.
            generator_nodes: Nodes dedicated to the generator. 0 (default)
                means shared host.
            gpus_per_node: Physical GPUs per node. Defaults to
                trainer_world_size + generator_world_size for single-node.
        """
        config = self.config

        self.trainer_world_size = self._compute_world_size(config.trainer.parallelism)
        self.generator_world_size = self._compute_world_size(
            config.generator.parallelism
        )
        tp = config.trainer.parallelism
        dp_shard = max(tp.data_parallel_shard_degree, 1)
        self.trainer_dp_degree = tp.data_parallel_replicate_degree * dp_shard

        if host_mesh is None:
            host_mesh = this_host()
        if gpus_per_node is None:
            gpus_per_node = self.trainer_world_size + self.generator_world_size

        self._multi_node = trainer_nodes > 0 or generator_nodes > 0

        logger.info(
            f"{self.generator_world_size} generator GPUs + "
            f"{self.trainer_world_size} trainer GPUs = "
            f"{self.trainer_world_size + self.generator_world_size} total"
        )

        self.system_prompt = self.task.get_system_prompt()

        node_assignments = (
            [trainer_nodes, generator_nodes, 0] if self._multi_node else None
        )
        trainer_mesh, generator_mesh, grader_mesh = create_meshes(
            host_mesh,
            gpus_per_node,
            gpu_requests=[
                self.trainer_world_size,
                self.generator_world_size,
                0,
            ],
            node_assignments=node_assignments,
        )

        # Store proc meshes for cleanup
        self._proc_meshes = [trainer_mesh, generator_mesh, grader_mesh]

        await setup_torch_elastic_env_async(trainer_mesh)
        await setup_torch_elastic_env_async(generator_mesh)

        # Spawn actors on their respective meshes
        self.trainer = trainer_mesh.spawn(
            "trainer",
            PolicyTrainer,
            config.trainer,
            loss_fn=GRPOLoss(),
        )
        self.generator = generator_mesh.spawn(
            "generator",
            VLLMGenerator,
            config.generator,
        )
        self.grader = grader_mesh.spawn(
            "grader",
            Grader,
            self.task.reward_function,
        )

        # Initialize TorchStore for weight sync between trainer and generator.
        # StorageVolumes are spawned on the trainer mesh so they are colocated
        # with the weight source for faster data access in the non-RDMA path.
        # LocalRankStrategy: routes each process to a storage volume based on
        #   LOCAL_RANK, so colocated processes share the same volume.
        # https://github.com/meta-pytorch/torchstore
        await ts.initialize(mesh=trainer_mesh, strategy=ts.LocalRankStrategy())

        # push weights from trainer
        self.trainer.push_weights.call().get()
        # pull weights for policy version 0 (initial weights)
        self.generator.pull_weights.call(0).get()

    async def evaluate(self, num_samples: int = 20) -> dict:
        eval_task = SumDigitsTask(seed=99)
        eval_prompts = []
        eval_answers = []
        for _ in range(num_samples):
            question, answer = eval_task.create_question()
            eval_prompts.append(self.system_prompt + "\n\n" + question)
            eval_answers.append(answer)

        completions: list[Completion] = self._get_rank_0_value(
            self.generator.generate.call(eval_prompts).get()
        )
        samples_per_prompt = self.config.generator.num_samples_per_prompt
        answers_per_completion = [
            answer for answer in eval_answers for _ in range(samples_per_prompt)
        ]
        scored: list[ScoredCompletion] = self._get_rank_0_value(
            self.grader.score.call(completions, answers_per_completion).get(),
            has_gpus=False,
        )

        correct = 0
        for i, answer in enumerate(eval_answers):
            scored_completion = scored[i * samples_per_prompt]
            extracted = extract_answer(scored_completion.completion.text)
            correct += int(extracted == int(answer))

        result = {
            "accuracy": correct / num_samples,
            "correct": correct,
            "total": num_samples,
        }
        return result

    async def train(self):
        num_steps = self.config.num_steps

        # Pre-training evaluation
        logger.info("Evaluating pre-training baseline...")
        pre_eval = await self.evaluate()

        logger.info("=" * 80)
        logger.info(f"Starting RL training for {num_steps} steps")
        logger.info("=" * 80)

        for step in range(num_steps):
            train_prompts = []
            train_answers = []
            for _ in range(self.config.num_episodes_per_step):
                question, answer = self.task.create_question()
                train_prompts.append(self.system_prompt + "\n\n" + question)
                train_answers.append(answer)

            step_start: float = time.perf_counter()

            completions: list[Completion] = self._get_rank_0_value(
                self.generator.generate.call(train_prompts).get()
            )

            answers_per_completion = [
                answer
                for answer in train_answers
                for _ in range(self.config.generator.num_samples_per_prompt)
            ]
            scored: list[ScoredCompletion] = self._get_rank_0_value(
                self.grader.score.call(completions, answers_per_completion).get(),
                has_gpus=False,
            )

            scored_groups: dict[str, list[ScoredCompletion]] = defaultdict(list)
            for sc in scored:
                scored_groups[sc.completion.group_id].append(sc)

            answers_by_group = {
                sc.completion.group_id: answer
                for sc, answer in zip(scored, answers_per_completion)
            }

            episodes = []
            for group_id, group in scored_groups.items():
                advantages = compute_advantages(group)

                if self.config.log_samples:
                    _log_group_sample(group, answers_by_group.get(group_id, ""))

                for sc, adv in zip(group, advantages):
                    c = sc.completion
                    episodes.append(
                        Episode(
                            prompt_tokens=c.prompt_tokens,
                            response_tokens=c.response_tokens,
                            logprobs=c.logprobs,
                            group_id=c.group_id,
                            text=c.text,
                            reward=sc.reward,
                            advantage=adv,
                            policy_version=c.policy_version,
                        )
                    )

            policy_version = self._validate_policy_versions(episodes)
            sharded_episodes = self._shard_episodes(episodes)
            batches = [
                self._collate_rank_episodes(
                    rank_episodes,
                    policy_version=policy_version,
                )
                for rank_episodes in sharded_episodes
            ]
            fwd_bwd_result = self._get_rank_0_value(
                self.trainer.forward_backward.call(batches).get()
            )
            optim_result = self._get_rank_0_value(
                self.trainer.optim_step.call().get()
            )

            t0 = time.perf_counter()
            self.trainer.push_weights.call().get()
            t_push = time.perf_counter() - t0
            self.generator.pull_weights.call(optim_result.policy_version).get()
            t_total = time.perf_counter() - t0
            logger.info(f"Weight sync: push={t_push:.3f}s, total={t_total:.3f}s")

            t_step = time.perf_counter() - step_start

            all_token_lens = [len(ep.response_tokens) for ep in episodes]
            avg_len = sum(all_token_lens) / len(all_token_lens)

            all_rewards = [ep.reward for ep in episodes]
            reward_mean = sum(all_rewards) / len(all_rewards)
            correct_count = sum(1 for r in all_rewards if r > 0)
            total_count = len(all_rewards)

            logger.info(
                f"Step {step:2d} | Loss: {fwd_bwd_result.loss:+.4f} | "
                f"Reward: {reward_mean:+.3f} | "
                f"Correct: {correct_count:>2}/{total_count} | "
                f"Avg tokens: {avg_len:>3.0f} | "
                f"Logprob diff: mean={fwd_bwd_result.metrics['logprob_diff_mean']:.4e}, "
                f"max={fwd_bwd_result.metrics['logprob_diff_max']:.4e} | "
                f"Time: {t_step:.1f}s"
            )

            # Check for divergence
            if not math.isfinite(fwd_bwd_result.loss):
                logger.info("!" * 80)
                logger.info("ERROR: Loss is NaN/Inf! Training diverged.")
                logger.info("!" * 80)
                break

        logger.info("RL Training complete")
        logger.info("Evaluating post-training performance...")
        post_eval = await self.evaluate()

        logger.info("=" * 80)
        logger.info(
            f"Pre-training:  Accuracy={pre_eval['accuracy']:.0%} "
            f"({pre_eval['correct']}/{pre_eval['total']})"
        )
        logger.info(
            f"Post-training: Accuracy={post_eval['accuracy']:.0%} "
            f"({post_eval['correct']}/{post_eval['total']})"
        )
        logger.info("=" * 80)


async def main():
    """Run the distributed RL training loop using Monarch."""
    config = ConfigManager().parse_args(
        default_module="torchtitan.experiments.rl.sync_grpo_sum_digits"
    )
    rl_trainer = RLTrainer(config)
    await rl_trainer.setup()
    await rl_trainer.train()


if __name__ == "__main__":
    asyncio.run(main())
