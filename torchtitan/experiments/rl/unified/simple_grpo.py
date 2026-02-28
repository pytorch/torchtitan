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
from dataclasses import dataclass, field

import torch

from monarch.actor import this_host
from monarch.utils import setup_env_for_distributed
from torchtitan.config import Configurable
from torchtitan.config.manager import ConfigManager
from torchtitan.experiments.rl.unified.actors.generator import VLLMGenerator
from torchtitan.experiments.rl.unified.actors.grader import Grader
from torchtitan.experiments.rl.unified.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.vllm_compat.simple_rl import trivial_reward_function
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

        # Use fake dataset for test. TODO: Implement real RL dataloader.
        logger.debug("Using default prompts")
        prompts_with_answers = [
            ("The capital of France is", "paris"),
            ("What is 7 times 8?", "56"),
            ("The first president of the United States was", "washington"),
            ("The chemical symbol for water is", "h2o"),
            ("The largest planet in our solar system is", "jupiter"),
        ]
        self.prompt_texts = [p[0] for p in prompts_with_answers]
        self.expected_answers = [p[1] for p in prompts_with_answers]

        logger.debug(f"Loaded {len(self.prompt_texts)} prompts")

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
            trivial_reward_function,
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
            prompt_texts=self.prompt_texts,
        )

        # Initialize generator with trainer weights.
        self.generator.update.call(0, initial_weights).get()

    async def train(self):
        """Run the RL training loop.

        Must call :meth:`setup` first.
        """
        num_steps = self.config.num_steps

        logger.info("\n" + "=" * 80)
        logger.info(f"Starting RL training for {num_steps} steps")
        logger.info("=" * 80)

        for step in range(num_steps):
            # Fully sync RL loop with separate scoring step
            # 1. VLLMGenerator produces episodes (one per prompt, without rewards)
            # TODO: Create a queue to use all episode from all GPUs
            episodes = self.generator.generate.call().get().item(gpus=0)
            # Attach expected answers to each episode
            for episode, answer in zip(episodes, self.expected_answers):
                episode.expected_answer = answer
            # 2. Grader computes rewards per episode
            scored_episodes = self.grader.score.call(episodes).get().item(gpus=0)
            # 3. Trainer computes advantages and updates policy
            metrics = self.trainer_actor.step.call(scored_episodes).get().item(gpus=0)
            # 4. Sync weights back to generator (all TP ranks)
            weight_mesh = self.trainer_actor.get_weights.call().get()
            weights = {
                gpu: weight_mesh.item(gpus=gpu)
                for gpu in range(self.trainer_world_size)
            }
            self.generator.update.call(metrics["policy_version"], weights).get()

            logger.info(
                f"\nStep {step:3d} | Loss: {metrics['loss']:.4f} | "
                f"Reward: {metrics['reward_mean']:+.3f} | "
                f"Logprob diff: mean={metrics['logprob_diff_mean']:.4e}, "
                f"max={metrics['logprob_diff_max']:.4e}"
            )
            logger.debug(f"  Sample: {metrics['sample_completion']}...")

            # Check for divergence
            if not torch.isfinite(torch.tensor(metrics["loss"])):
                logger.debug("\n" + "!" * 80)
                logger.debug("ERROR: Loss is NaN/Inf! Training diverged.")
                logger.debug("!" * 80)
                break

        logger.info("\n" + "=" * 80)
        logger.info("RL Training complete")
        logger.info("=" * 80)


async def main():
    """Run the distributed RL training loop using Monarch."""
    config = ConfigManager().parse_args()
    rl_trainer = RLTrainer(config)
    await rl_trainer.setup()
    await rl_trainer.train()


if __name__ == "__main__":
    asyncio.run(main())
