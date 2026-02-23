# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Multiprocess RL training loop using Monarch Actors.

This demonstrates:
1. Distributed actor architecture with Generator (vLLM) and Trainer (TorchTitan) components
2. File based weight synchronization between trainer and generator

The architecture mirrors monarch's grpo_actor.py but adapted for vLLM rollouts + TorchTitan training.

Command to run:
python3 torchtitan/experiments/rl/unified/simple_grpo.py
"""

import asyncio
import logging

import torch

import torchtitan.experiments.rl.unified  # noqa: F401 — registers models with vLLM
from monarch.actor import this_host
from monarch.utils import setup_env_for_distributed
from torchtitan.experiments.rl.unified.actors.generator import Generator
from torchtitan.experiments.rl.unified.actors.trainer import Trainer
from torchtitan.experiments.rl.unified.config_registry import rl_grpo_qwen3_0_6b

logger = logging.getLogger(__name__)


async def main():
    """Run the distributed RL training loop using Monarch."""

    # Step 1: Load config from config_registry
    # TODO: Once the config branch lands, replace with:
    #     from torchtitan.config.manager import ConfigManager
    #     config = ConfigManager().parse_args()
    config = rl_grpo_qwen3_0_6b()
    trainer_cfg = config.trainer

    # Compute world size for trainer and generator
    # TODO: refine the world size computation and check
    trainer_ddp_size = trainer_cfg.parallelism.data_parallel_replicate_degree
    trainer_tp_size = trainer_cfg.parallelism.tensor_parallel_degree

    # RL Training config
    num_steps = trainer_cfg.training.steps

    # Use fake dataset for test. TODO: Implement real RL dataloader.
    logger.info("Using default prompts")
    prompts_with_answers = [
        ("The capital of France is", "paris"),
        ("What is 7 times 8?", "56"),
        ("The first president of the United States was", "washington"),
        ("The chemical symbol for water is", "h2o"),
        ("The largest planet in our solar system is", "jupiter"),
    ]
    prompt_texts = [p[0] for p in prompts_with_answers]
    expected_answers = [p[1] for p in prompts_with_answers]

    logger.info(f"Loaded {len(prompt_texts)} prompts")

    # Create process meshes
    trainer_mesh = this_host().spawn_procs(
        per_host={"gpus": trainer_ddp_size * trainer_tp_size}
    )
    gen_tp_size = config.generator.vllm_engine.parallelism.tensor_parallel_degree
    gen_mesh = this_host().spawn_procs(per_host={"gpus": gen_tp_size})

    # Set up distributed env vars so that actors are connected via c10d
    await setup_env_for_distributed(
        trainer_mesh,
        master_addr="localhost",  # TODO: figure out what to set
        master_port=29500,  # TODO: figure out what to set
    )

    # Set up distributed env vars so that actors are connected via c10d
    await setup_env_for_distributed(
        gen_mesh,
        master_addr="localhost",  # TODO: figure out what to set
        master_port=29501,  # TODO: figure out what to set
    )

    # Spawn actors on trainer and generator mesh
    trainer = trainer_mesh.spawn(
        "trainer",
        Trainer,
        config,
    )

    generator = gen_mesh.spawn(
        "generator",
        Generator,
        config.generator,
        rl_config=config,
        prompt_texts=prompt_texts,
        expected_answers=expected_answers,
    )

    # Initialize generator with trainer weights
    initial_weights = trainer.get_weights.call().get().item(gpus=0)
    generator.update.call(0, initial_weights).get()

    # Training loop
    logger.info("\n" + "=" * 80)
    logger.info(f"Starting RL training for {num_steps} steps")
    logger.info("=" * 80)

    for step in range(num_steps):
        # Fully sync RL loop
        # NOTE: This is only getting Trajectory generated from trainer 0, and trainer 1's data is ignored.
        # .get() is a monarch synchronize API which makes the loop fully sync
        batch = generator.generate.call().get().item(gpus=0)
        metrics = trainer.step.call(batch).get().item(gpus=0)
        weights = trainer.get_weights.call().get().item(gpus=0)
        generator.update.call(metrics["policy_version"], weights).get()

        logger.info(
            f"\nStep {step:3d} | Loss: {metrics['loss']:.4f} | "
            f"Reward: {metrics['reward_mean']:+.3f}"
        )
        logger.info(f"  Sample: {metrics['sample_completion']}...")

        # Check for divergence
        if not torch.isfinite(torch.tensor(metrics["loss"])):
            logger.info("\n" + "!" * 80)
            logger.info("ERROR: Loss is NaN/Inf! Training diverged.")
            logger.info("!" * 80)
            break

    logger.info("\n" + "=" * 80)
    logger.info("RL Training complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
