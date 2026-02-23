# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Multiprocess RL training loop using Monarch Actors.

This demonstrates:
1. Distributed actor architecture with Generator (vLLM), Grader, and Trainer (TorchTitan) components
2. File based weight synchronization between trainer and generator
3. Separate scoring component for reward and advantage computation

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
from torchtitan.experiments.rl.unified.actors.grader import Grader
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
    trainer_world_size = trainer_ddp_size * trainer_tp_size

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

    # Create process mesh for trainer (generator is collocated on same mesh)
    # TODO: Make the world size according to parallel degrees
    trainer_mesh = this_host().spawn_procs(per_host={"gpus": trainer_world_size})

    # Set up distributed env vars so that actors are connected via c10d
    await setup_env_for_distributed(
        trainer_mesh,
        master_addr="localhost",  # TODO: figure out what to set
        master_port=29501,  # TODO: figure out what to set
    )

    # Spawn trainer first and wait for it to be fully initialized on all ranks
    # before spawning generator. This is critical because both actors do NCCL
    # collective operations during init, and Monarch doesn't guarantee init order
    # across ranks — spawning them concurrently can cause cross-rank deadlocks.
    trainer = trainer_mesh.spawn(
        "trainer",
        Trainer,
        config,
    )

    # Spawn grader on trainer mesh (no collective ops in init, safe to co-spawn)
    grader = trainer_mesh.spawn(
        "grader",
        Grader,
        config,
    )

    # Wait for trainer to be fully initialized on all ranks
    # Collect weights from ALL trainer ranks (each holds different TP shards)
    initial_weight_mesh = trainer.get_weights.call().get()
    initial_weights = {
        gpu: initial_weight_mesh.item(gpus=gpu) for gpu in range(trainer_world_size)
    }

    # Now spawn generator — trainer init is complete and process group is ready
    # Make trainer and generator collocated
    generator = trainer_mesh.spawn(
        "generator",
        Generator,
        config.generator,
        rl_config=config,
        prompt_texts=prompt_texts,
        expected_answers=expected_answers,
    )

    # Initialize generator with trainer weights.
    generator.update.call(0, initial_weights).get()

    # Training loop
    logger.info("\n" + "=" * 80)
    logger.info(f"Starting RL training for {num_steps} steps")
    logger.info("=" * 80)

    for step in range(num_steps):
        # Fully sync RL loop with separate scoring step
        # 1. Generator produces episode (without rewards)
        episode = generator.generate.call().get().item(gpus=0)
        # 2. Grader computes rewards
        episode = grader.score.call(episode).get().item(gpus=0)
        # 3. Trainer computes advantages and updates policy
        metrics = trainer.step.call(episode).get().item(gpus=0)
        # 4. Sync weights back to generator (all TP ranks)
        weight_mesh = trainer.get_weights.call().get()
        weights = {gpu: weight_mesh.item(gpus=gpu) for gpu in range(trainer_world_size)}
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
