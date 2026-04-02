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
python3 torchtitan/experiments/rl/simple_grpo_sum_digits.py \
    --module rl --config rl_grpo_qwen3_0_6b \
    --hf_assets_path=<path_to_model_checkpoint>
"""

import asyncio
import logging
import os
import re
import time
from collections import defaultdict
from collections.abc import Callable
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
from torchtitan.experiments.rl.sum_digits import extract_answer, SumDigitsTask
from torchtitan.experiments.rl.types import Episode, TrainBatch
from torchtitan.protocols.model_spec import ModelSpec

logger = logging.getLogger(__name__)


class GRPOLoss:
    """GRPO loss with DAPO-style token-level normalization and KL penalty.

    Uses flat token averaging (every token weighted equally regardless of
    sequence length) instead of GRPO's original per-sequence mean, which
    biases toward shorter outputs. Clipping is applied per-token.

    See: DAPO (arXiv:2503.14476), Dr. GRPO (arXiv:2503.20783)
    """

    def __init__(
        self,
        kl_coef: float = 0.1,
        clip_eps: float = 0.2,
    ):
        self.kl_coef = kl_coef
        self.clip_eps = clip_eps

    def __call__(
        self,
        policy_logprobs: torch.Tensor,
        response_mask: torch.Tensor,
        advantages: torch.Tensor,
        ref_logprobs: torch.Tensor,
        **_,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute GRPO clipped surrogate loss with KL penalty.

        Args:
            policy_logprobs: [B, L] per-token logprobs from the current policy.
            response_mask: [B, L] binary mask, 1.0 for response tokens.
            advantages: [B] per-sequence advantages.
            ref_logprobs: [B, L] per-token logprobs from the reference policy.
            **_: Absorbs extra TrainBatch fields passed via vars().
        """
        device = policy_logprobs.device
        ref_logprobs = ref_logprobs.to(device).detach()
        advantages = advantages.to(device)
        response_mask = response_mask.to(device)

        # Per-token log ratio and importance weight
        log_ratio = policy_logprobs - ref_logprobs
        ratio = torch.exp(log_ratio)

        # Expand per-sequence advantages to per-token [B, L]
        per_token_adv = advantages.unsqueeze(1).expand_as(policy_logprobs)

        # Clipped surrogate objective
        clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
        pg_loss = -torch.min(ratio * per_token_adv, clipped * per_token_adv)

        # Average over response tokens only
        num_response_tokens = response_mask.sum().clamp(min=1.0)
        pg_loss = (pg_loss * response_mask).sum() / num_response_tokens

        # KL penalty (Schulman approximation, flat token average)
        kl_div = ((ratio - 1 - log_ratio) * response_mask).sum() / num_response_tokens

        loss = pg_loss + self.kl_coef * kl_div

        metrics = {
            "pg_loss": pg_loss.item(),
            "kl_div": kl_div.item(),
            "ratio_mean": (ratio * response_mask).sum().item()
            / num_response_tokens.item(),
        }

        return loss, metrics


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


def _log_samples(episodes: list[Episode]) -> None:
    """Log the first completion per group for debugging."""
    seen_groups: set[str] = set()
    for ep in episodes:
        if ep.group_id in seen_groups:
            continue
        seen_groups.add(ep.group_id)
        extracted = extract_answer(ep.text)
        is_correct = (
            extracted == int(ep.expected_answer) if ep.expected_answer else None
        )
        mark = "+" if is_correct else "-"
        logger.info(
            f"  [{mark}] expected={ep.expected_answer} extracted={extracted} "
            f"reward={ep.reward:+.1f}"
        )
        logger.info(f"       A: {ep.text[:300].replace(chr(10), ' ').strip()}")


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

        group_size: int = 8
        """Number of completions per prompt (GRPO group size)."""

        log_samples: bool = False
        """Log first completion per episode during training and eval."""

        trainer: PolicyTrainer.Config = field(default_factory=PolicyTrainer.Config)
        """PolicyTrainer config. Controls optimizer, training, parallelism"""

        generator: VLLMGenerator.Config = field(default_factory=VLLMGenerator.Config)
        """VLLMGenerator actor configuration (vLLM engine, sampling)."""

    def __init__(self, config: Config):
        self.config = config

        # Propagate shared config to sub-configs.
        # The trainer uses the model spec as-is (core Titan's parallelize_fn
        # which includes FSDP). The generator overrides parallelize_fn with
        # the RL-specific TP-only plan for vLLM compatibility.
        config.trainer.model_spec = config.model_spec
        config.trainer.hf_assets_path = config.hf_assets_path
        config.generator.num_samples_per_prompt = config.group_size

        self.task = SumDigitsTask(seed=42)

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

    def _collate(
        self,
        episodes: list[Episode],
        pad_token_id: int = 0,
    ) -> list[TrainBatch]:
        """Pre-shard episodes into per-DP-rank TrainBatches.

        Pads all sequences to the same length within each rank's batch and
        builds [B, L] tensors on CPU. Uses interleaved indexing so each DP
        rank gets a unique slice.

        Returns a list of TrainBatch with one element per DP rank.
        """
        dp_degree = self.trainer_dp_degree
        batches = []
        for rank in range(dp_degree):
            idx = list(range(rank, len(episodes), dp_degree))
            rank_episodes = [episodes[i] for i in idx]

            prompt_lens = [len(ep.prompt_token_ids) for ep in rank_episodes]
            response_lens = [len(ep.token_ids) for ep in rank_episodes]
            max_len = max(p + r for p, r in zip(prompt_lens, response_lens))

            padded_ids = []
            padded_ref_logprobs = []
            for ep in rank_episodes:
                seq = ep.prompt_token_ids + ep.token_ids
                pad_len = max_len - len(seq)
                padded_ids.append(seq + [pad_token_id] * pad_len)
                # ref logprobs: 0 for prompt, ep.logprobs for response, 0 for padding
                ref_lps = (
                    [0.0] * len(ep.prompt_token_ids) + ep.logprobs + [0.0] * pad_len
                )
                padded_ref_logprobs.append(ref_lps)

            batches.append(
                TrainBatch(
                    token_ids=torch.tensor(padded_ids, dtype=torch.long),
                    prompt_lens=torch.tensor(prompt_lens, dtype=torch.long),
                    response_lens=torch.tensor(response_lens, dtype=torch.long),
                    advantages=torch.tensor(
                        [ep.advantage for ep in rank_episodes],
                        dtype=torch.float32,
                    ),
                    ref_logprobs=torch.tensor(padded_ref_logprobs, dtype=torch.float32),
                    pad_token_id=pad_token_id,
                )
            )
        return batches

    async def setup(self):
        """Spawn Monarch actors on separate meshes and initialize weights.

        Creates separate GPU meshes for trainer and generator, a CPU mesh for
        the grader, and synchronizes initial weights from trainer to generator.
        Must be called before :meth:`train`.
        """
        config = self.config

        self.trainer_world_size = self._compute_world_size(config.trainer.parallelism)
        self.generator_world_size = self._compute_world_size(
            config.generator.parallelism
        )

        # DP degree for collating batches (pre-shard data for each DP rank)
        tp = config.trainer.parallelism
        dp_shard = max(tp.data_parallel_shard_degree, 1)
        self.trainer_dp_degree = tp.data_parallel_replicate_degree * dp_shard

        total_gpus = self.trainer_world_size + self.generator_world_size
        logger.info(
            f"{self.generator_world_size} generator GPUs + "
            f"{self.trainer_world_size} trainer GPUs = {total_gpus} total"
        )

        self.system_prompt = self.task.get_system_prompt()

        # Allocate non-overlapping GPU ranges for each mesh
        provisioner = Provisioner(total_gpus=total_gpus)

        # Spawn separate meshes for trainer, generator, and grader
        trainer_mesh = this_host().spawn_procs(
            per_host={"gpus": self.trainer_world_size},
            bootstrap=provisioner.allocate(self.trainer_world_size),
        )
        generator_mesh = this_host().spawn_procs(
            per_host={"gpus": self.generator_world_size},
            bootstrap=provisioner.allocate(self.generator_world_size),
        )
        grader_mesh = this_host().spawn_procs()

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
            model_spec=config.model_spec,
            model_path=config.hf_assets_path,
            batch_invariant_mode=config.batch_invariant_mode,
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
        self.generator.pull_model_state_dict.call(0).get()

    async def evaluate(self, num_samples: int = 20) -> dict:
        """Run evaluation on held-out prompts.

        Generates on eval prompts, scores them, and reports accuracy.

        Args:
            num_samples: Number of eval prompts to generate

        Returns:
            Dict with accuracy, correct, total, format_rate
        """
        eval_task = SumDigitsTask(seed=99)
        eval_prompts = []
        eval_answers = []
        eval_questions = []
        for _ in range(num_samples):
            question, answer = eval_task.create_question()
            eval_prompts.append(self.system_prompt + "\n\n" + question)
            eval_answers.append(answer)
            eval_questions.append(question)

        # Generate on eval prompts
        episodes = (
            self.generator.generate.call(eval_prompts, eval_answers).get().item(gpus=0)
        )

        # Score: check first episode per prompt (episodes are ordered by prompt)
        correct = 0
        format_ok = 0
        samples_per_prompt = self.config.group_size
        for i, (question, answer) in enumerate(zip(eval_questions, eval_answers)):
            ep = episodes[i * samples_per_prompt]
            extracted = extract_answer(ep.text)
            is_correct = extracted == int(answer)
            has_tag = bool(re.search(r"\[ANSWER\]", ep.text))
            correct += int(is_correct)
            format_ok += int(has_tag)

        if self.config.log_samples:
            _log_samples(episodes)

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
        """Run the RL training loop.

        Must call :meth:`setup` first.
        """
        num_steps = self.config.num_steps

        # Pre-training evaluation
        logger.info("Evaluating pre-training baseline...")
        pre_eval = await self.evaluate()

        logger.info("=" * 80)
        logger.info(f"Starting RL training for {num_steps} steps")
        logger.info("=" * 80)

        # Derive number of prompts from batch dimensions:
        # total_episodes = local_batch_size * dp_degree
        # num_prompts = total_episodes / group_size
        local_batch_size = self.config.trainer.training.local_batch_size
        group_size = self.config.group_size
        total_episodes = local_batch_size * self.trainer_dp_degree
        assert total_episodes % group_size == 0, (
            f"total_episodes ({total_episodes} = local_batch_size {local_batch_size} "
            f"* dp_degree {self.trainer_dp_degree}) must be divisible by "
            f"group_size ({group_size})"
        )
        num_prompts = total_episodes // group_size

        for step in range(num_steps):
            train_prompts = []
            train_answers = []
            for _ in range(num_prompts):
                question, answer = self.task.create_question()
                train_prompts.append(self.system_prompt + "\n\n" + question)
                train_answers.append(answer)

            step_start: float = time.perf_counter()

            # Fully sync RL loop (GRPO)
            # 1. Generator produces flat list of Episodes with group_id
            # TODO: Create a queue to use all episodes from all GPUs
            episodes = (
                self.generator.generate.call(train_prompts, train_answers)
                .get()
                .item(gpus=0)
            )

            # 2. Grader computes rewards per episode
            episodes = self.grader.score.call(episodes).get().item()

            # 3. Controller computes GRPO advantages (normalize within group)
            groups: dict[str, list[int]] = defaultdict(list)
            for idx, ep in enumerate(episodes):
                groups[ep.group_id].append(idx)
            for indices in groups.values():
                rewards = torch.tensor([episodes[i].reward for i in indices])
                mean_reward = rewards.mean().item()
                for i in indices:
                    episodes[i].advantage = episodes[i].reward - mean_reward

            if self.config.log_samples:
                _log_samples(episodes)

            # 4. Trainer updates policy using episodes with advantages
            # TODO: ref logprobs are zeros until ReferenceModel actor exists
            batches = self._collate(episodes)
            fb_result = self.trainer.forward_backward.call(batches).get().item(gpus=0)
            optim_result = self.trainer.optim_step.call().get().item(gpus=0)

            # 5. Sync weights
            t0 = time.perf_counter()
            self.trainer.push_weights.call().get()
            t_push = time.perf_counter() - t0
            self.generator.pull_model_state_dict.call(optim_result.policy_version).get()
            t_total = time.perf_counter() - t0
            logger.info(f"Weight sync: push={t_push:.3f}s, total={t_total:.3f}s")

            t_step = time.perf_counter() - step_start

            all_token_lens = [len(ep.token_ids) for ep in episodes]
            avg_len = sum(all_token_lens) / len(all_token_lens)

            all_rewards = [ep.reward for ep in episodes]
            reward_mean = sum(all_rewards) / len(all_rewards)
            correct_count = sum(1 for r in all_rewards if r > 0)
            total_count = len(all_rewards)

            logger.info(
                f"Step {step:2d} | Loss: {fb_result.loss:+.4f} | "
                f"Reward: {reward_mean:+.3f} | "
                f"Correct: {correct_count:>2}/{total_count} | "
                f"Avg tokens: {avg_len:>3.0f} | "
                f"Time: {t_step:.1f}s"
            )

            # Check for divergence
            if not torch.isfinite(torch.tensor(fb_result.loss)):
                logger.info("!" * 80)
                logger.info("ERROR: Loss is NaN/Inf! Training diverged.")
                logger.info("!" * 80)
                break

        # Post-training evaluation
        logger.info("RL Training complete")
        logger.info("Evaluating post-training performance...")
        post_eval = await self.evaluate()

        logger.info("=" * 80)
        logger.info(
            f"Pre-training:  Accuracy={pre_eval['accuracy']:.0%} "
            f"({pre_eval['correct']}/{pre_eval['total']}) "
            f"Format={pre_eval['format_rate']:.0%} ({pre_eval['format_ok']}/{pre_eval['total']})"
        )
        logger.info(
            f"Post-training: Accuracy={post_eval['accuracy']:.0%} "
            f"({post_eval['correct']}/{post_eval['total']}) "
            f"Format={post_eval['format_rate']:.0%} ({post_eval['format_ok']}/{post_eval['total']})"
        )
        logger.info("=" * 80)


async def main():
    """Run the distributed RL training loop using Monarch."""
    config = ConfigManager().parse_args()
    rl_trainer = RLTrainer(config)
    await rl_trainer.setup()
    await rl_trainer.train()


if __name__ == "__main__":
    asyncio.run(main())
