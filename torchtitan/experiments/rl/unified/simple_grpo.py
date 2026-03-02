# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Multiprocess RL training loop using Monarch Actors.

This demonstrates:
1. Distributed actor architecture with VLLMGenerator (vLLM) and PolicyTrainer (TorchTitan) components
2. Weight synchronization between trainer and generator by unwrapping and
    rewrap DTensor. We have strong assumption that trainer and generator has same parallelism
3. Separate scoring component for reward and advantage computation

The architecture mirrors monarch's grpo_actor.py but adapted for vLLM rollouts + TorchTitan training.

Command to run:
python3 torchtitan/experiments/rl/unified/simple_grpo.py \
    --module rl.unified --config rl_grpo_qwen3_0_6b \
    --hf_assets_path=<path_to_model_checkpoint>
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field

import torch
from monarch.actor import this_host
from monarch.utils import setup_env_for_distributed
from torchtitan.config import Configurable
from torchtitan.config.manager import ConfigManager
from torchtitan.experiments.rl.sum_digits import (
    extract_answer,
    sum_digits_reward_function,
    SumDigitsSpec,
)
from torchtitan.experiments.rl.unified.actors.generator import VLLMGenerator
from torchtitan.experiments.rl.unified.actors.grader import Grader
from torchtitan.experiments.rl.unified.actors.trainer import PolicyTrainer
from torchtitan.protocols.model_spec import ModelSpec

logger = logging.getLogger(__name__)


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

        num_prompts_per_step: int = 5
        """Number of prompts to generate per training step."""

        log_samples: bool = False
        """Log per-sample outputs during eval and training steps."""

        trainer: PolicyTrainer.Config = field(default_factory=PolicyTrainer.Config)
        """PolicyTrainer config. Controls optimizer, training, parallelism"""

        generator: VLLMGenerator.Config = field(default_factory=VLLMGenerator.Config)
        """VLLMGenerator actor configuration (vLLM engine, sampling)."""

    def __init__(self, config: Config):
        self.config = config

        # Patch model_spec to use the RL-specific parallelize function.
        # TODO: Switch to canonical Qwen3 parallel plan
        from torchtitan.experiments.rl.unified.models.parallelize import (
            parallelize_qwen3,
        )

        config.model_spec.parallelize_fn = parallelize_qwen3

    async def setup(self):
        """Spawn Monarch actors and initialize weights.

        Creates the process mesh, spawns trainer/generator/grader actors,
        and synchronizes initial weights from trainer to generator.
        Must be called before :meth:`train`.
        """
        config = self.config

        # Validate that trainer and generator have the same parallel plan
        # since they are collocated on the same mesh (our strong assumption now)
        assert config.trainer.parallelism == config.generator.parallelism, (
            f"Trainer and generator must use the same parallel plan.\n"
            f"  Trainer:   {config.trainer.parallelism}\n"
            f"  VLLMGenerator: {config.generator.vllm_engine.parallelism}"
        )

        self.trainer_world_size = (
            config.trainer.parallelism.data_parallel_replicate_degree
            * config.trainer.parallelism.tensor_parallel_degree
        )

        # Task specification for generating prompts
        self.task_spec = SumDigitsSpec(seed=42)
        self.system_prompt = self.task_spec.get_system_prompt()

        # Create process mesh for trainer (generator is collocated on same mesh)
        # TODO: Make the world size according to parallel degrees
        trainer_mesh = this_host().spawn_procs(
            per_host={"gpus": self.trainer_world_size}
        )

        # Set up distributed env vars so that actors are connected via c10d
        await setup_env_for_distributed(
            trainer_mesh,
            master_addr="localhost",  # TODO: figure out what to set
            master_port=29501,  # TODO: figure out what to set
        )

        # Spawn trainer first
        self.trainer_actor = trainer_mesh.spawn(
            "trainer",
            PolicyTrainer,
            config.trainer,
            model_spec=config.model_spec,
            batch_invariant_mode=config.batch_invariant_mode,
            hf_assets_path=config.hf_assets_path,
        )

        # Spawn grader on trainer mesh
        self.grader = trainer_mesh.spawn(
            "grader",
            Grader,
            sum_digits_reward_function,
        )

        # Wait for trainer to be fully initialized on all ranks then collect weights
        initial_weight_mesh = self.trainer_actor.get_weights.call().get()
        initial_weights = {
            gpu: initial_weight_mesh.item(gpus=gpu)
            for gpu in range(self.trainer_world_size)
        }

        self.generator = trainer_mesh.spawn(
            "generator",
            VLLMGenerator,
            config.generator,
            model_spec=config.model_spec,
            model_path=config.hf_assets_path,
            batch_invariant_mode=config.batch_invariant_mode,
        )

        # Initialize generator with trainer weights.
        self.generator.update.call(0, initial_weights).get()

    def _generate_prompts(self) -> tuple[list[str], list[str]]:
        """Generate a batch of prompts and expected answers from the task spec."""
        prompt_texts = []
        expected_answers = []
        for _ in range(self.config.num_prompts_per_step):
            task: Task = self.task_spec.generate_task()
            prompt_texts.append(self.system_prompt + "\n\n" + task.question)
            expected_answers.append(str(task.correct_answer))
        return prompt_texts, expected_answers

    async def evaluate(self, num_samples: int = 20) -> dict:
        """Run evaluation on held-out prompts.

        Generates on eval prompts, scores them, and reports accuracy.

        Args:
            num_samples: Number of eval prompts to generate

        Returns:
            Dict with accuracy, correct, total, format_rate
        """
        eval_spec = SumDigitsSpec(seed=99)  # Different seed from training
        eval_prompts = []
        eval_answers = []
        eval_tasks = []
        for _ in range(num_samples):
            task: Task = eval_spec.generate_task()
            eval_prompts.append(self.system_prompt + "\n\n" + task.question)
            eval_answers.append(str(task.correct_answer))
            eval_tasks.append(task)

        # Generate on eval prompts
        episodes = self.generator.generate.call(eval_prompts).get().item(gpus=0)

        # Score: check first completion per episode
        correct = 0
        format_ok = 0
        for episode, task in zip(episodes, eval_tasks):
            text = episode.completions[0].text
            extracted = extract_answer(text)
            is_correct = extracted == task.correct_answer
            has_tag = bool(re.search(r"\[ANSWER\]", text))
            correct += int(is_correct)
            format_ok += int(has_tag)

            if self.config.log_samples:
                mark = "+" if is_correct else "-"
                logger.info(f"  [{mark}] Q: {task.question}")
                logger.info(f"       A: {text[:200]}")
                logger.info(
                    f"       extracted={extracted} expected={task.correct_answer}"
                )

        result = {
            "accuracy": correct / num_samples,
            "correct": correct,
            "total": num_samples,
            "format_rate": format_ok / num_samples,
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

        for step in range(num_steps):
            # Generate new prompts each step
            self.prompt_texts, self.expected_answers = self._generate_prompts()

            # Fully sync RL loop with separate scoring step
            # 1. VLLMGenerator produces episodes (one per prompt, without rewards)
            # TODO: Create a queue to use all episode from all GPUs
            t0 = time.perf_counter()
            episodes = (
                self.generator.generate.call(self.prompt_texts).get().item(gpus=0)
            )
            t_generate = time.perf_counter() - t0

            # Attach expected answers to each episode
            for episode, answer in zip(episodes, self.expected_answers):
                episode.expected_answer = answer

            # 2. Grader computes rewards per episode
            t0 = time.perf_counter()
            scored_episodes = self.grader.score.call(episodes).get().item(gpus=0)
            t_grade = time.perf_counter() - t0

            # 3. Trainer computes advantages and updates policy
            t0 = time.perf_counter()
            metrics = self.trainer_actor.step.call(scored_episodes).get().item(gpus=0)
            t_train = time.perf_counter() - t0

            # 4. Sync weights back to generator (all TP ranks)
            t0 = time.perf_counter()
            weight_mesh = self.trainer_actor.get_weights.call().get()
            weights = {
                gpu: weight_mesh.item(gpus=gpu)
                for gpu in range(self.trainer_world_size)
            }
            self.generator.update.call(metrics["policy_version"], weights).get()
            t_sync_weights = time.perf_counter() - t0

            # Count correct rewards from scored episodes
            all_rewards = [c.reward for ep in scored_episodes for c in ep.completions]
            correct_count = sum(1 for r in all_rewards if r > 0)
            total_count = len(all_rewards)

            all_token_lens = [
                len(c.token_ids) for ep in scored_episodes for c in ep.completions
            ]
            avg_len = sum(all_token_lens) / len(all_token_lens)

            logger.info(
                f"Step {step:3d} | Loss: {metrics['loss']:.4f} | "
                f"Reward: {metrics['reward_mean']:+.3f} | "
                f"Correct: {correct_count}/{total_count} | "
                f"Avg tokens: {avg_len:.0f} | "
                f"Time: generate={t_generate:.1f}s train={t_train:.1f}s sync_weights={t_sync_weights:.1f}s"
            )

            if self.config.log_samples:
                for ep, answer in zip(scored_episodes, self.expected_answers):
                    idx = 0  # Log first completion per prompt
                    comp = ep.completions[idx]
                    extracted = extract_answer(comp.text)
                    mark = "+" if comp.reward > 0 else "-"
                    question = ep.completions[0].text[:80].replace("\n", " ")
                    logger.info(
                        f"  [{mark}] expected={answer} extracted={extracted} "
                        f"reward={comp.reward:+.1f}"
                    )
                    logger.info(f"       {comp.text[:200].replace(chr(10), ' ')}")

            # Check for divergence
            if not torch.isfinite(torch.tensor(metrics["loss"])):
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
            f"Format={pre_eval['format_rate']:.0%}"
        )
        logger.info(
            f"Post-training: Accuracy={post_eval['accuracy']:.0%} "
            f"({post_eval['correct']}/{post_eval['total']}) "
            f"Format={post_eval['format_rate']:.0%}"
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
