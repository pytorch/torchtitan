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
from dataclasses import dataclass, field

import torch
from monarch.actor import this_host
from monarch.utils import setup_env_for_distributed
from torchtitan.config import Configurable
from torchtitan.config.manager import ConfigManager
from torchtitan.experiments.rl.unified.actors.generator import VLLMGenerator
from torchtitan.experiments.rl.unified.actors.grader import Grader
from torchtitan.experiments.rl.unified.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.unified.sum_digits import extract_answer, SumDigitsTask
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

        num_episodes_per_step: int = 5
        """Number of prompts to generate per training step."""

        log_samples: bool = False
        """Log first completion per episode during training and eval."""

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

        self.task = SumDigitsTask(seed=42)

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

        self.system_prompt = self.task.get_system_prompt()

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
            self.task.reward_function,
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
        episodes = self.generator.generate.call(eval_prompts).get().item(gpus=0)

        # Score: check first completion per episode
        correct = 0
        format_ok = 0
        for episode, question, answer in zip(episodes, eval_questions, eval_answers):
            text = episode.completions[0].text
            extracted = extract_answer(text)
            is_correct = extracted == int(answer)
            has_tag = bool(re.search(r"\[ANSWER\]", text))
            correct += int(is_correct)
            format_ok += int(has_tag)

            if self.config.log_samples:
                mark = "+" if is_correct else "-"
                logger.info(f"  [{mark}] expected={answer} extracted={extracted}")
                logger.info(f"       Q: {question}")
                logger.info(f"       A: {text[:300].replace(chr(10), ' ').strip()}")

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

        for step in range(num_steps):
            # Generate prompts for this step
            train_prompts = []
            train_answers = []
            train_questions = []
            for _ in range(self.config.num_episodes_per_step):
                question, answer = self.task.create_question()
                train_prompts.append(self.system_prompt + "\n\n" + question)
                train_answers.append(answer)
                train_questions.append(question)

            # Fully sync RL loop with separate scoring step
            # 1. VLLMGenerator produces episodes (one per prompt, without rewards)
            # TODO: Create a queue to use all episode from all GPUs
            episodes = (
                self.generator.generate.call(train_prompts).get().item(gpus=0)
            )

            # Attach expected answers to each episode
            for episode, answer in zip(episodes, train_answers):
                episode.expected_answer = answer

            # 2. Grader computes rewards per episode
            scored_episodes = self.grader.score.call(episodes).get().item(gpus=0)

            if self.config.log_samples:
                for ep, question, answer in zip(
                    scored_episodes, train_questions, train_answers
                ):
                    # Log first completion per prompt
                    comp = ep.completions[0]
                    extracted = extract_answer(comp.text)
                    mark = "+" if comp.reward > 0 else "-"
                    logger.info(
                        f"  [{mark}] expected={answer} extracted={extracted} "
                        f"reward={comp.reward:+.1f}"
                    )
                    logger.info(f"       Q: {question}")
                    logger.info(
                        f"       A: {comp.text[:300].replace(chr(10), ' ').strip()}"
                    )

            # 3. Trainer computes advantages and updates policy
            metrics = self.trainer_actor.step.call(scored_episodes).get().item(gpus=0)

            # 4. Sync weights back to generator (all TP ranks)
            weight_mesh = self.trainer_actor.get_weights.call().get()
            weights = {
                gpu: weight_mesh.item(gpus=gpu)
                for gpu in range(self.trainer_world_size)
            }
            self.generator.update.call(metrics["policy_version"], weights).get()

            all_token_lens = [
                len(c.token_ids) for ep in scored_episodes for c in ep.completions
            ]
            avg_len = sum(all_token_lens) / len(all_token_lens)

            all_rewards = [c.reward for ep in scored_episodes for c in ep.completions]
            correct_count = sum(1 for r in all_rewards if r > 0)
            total_count = len(all_rewards)

            logger.info(
                f"Step {step:2d} | Loss: {metrics['loss']:+.4f} | "
                f"Reward: {metrics['reward_mean']:+.3f} | "
                f"Correct: {correct_count:>2}/{total_count} | "
                f"Avg tokens: {avg_len:>3.0f} | "
                f"Logprob diff: mean={metrics['logprob_diff_mean']:.4e}, "
                f"max={metrics['logprob_diff_max']:.4e}"
            )

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
