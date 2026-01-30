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
python3 torchtitan/experiments/rl/unified/simple_grpo.py \
    --job.config_file torchtitan/experiments/rl/unified/run_configs/qwen3_0.6b.toml
"""
import asyncio
import logging
import os

import torch
from monarch.actor import this_host
from monarch.utils import setup_env_for_distributed
from torchtitan.config.manager import ConfigManager
from torchtitan.experiments.rl.unified.actors.generator import Generator
from torchtitan.experiments.rl.unified.actors.trainer import Trainer
from vllm.model_executor.layers.batch_invariant import (
    init_batch_invariance,
    vllm_is_batch_invariant,
)

logger = logging.getLogger(__name__)


async def main():
    """Run the distributed RL training loop using Monarch."""

    # Step 1: Load job config using config manager
    config_manager = ConfigManager()
    job_config = config_manager.parse_args()

    # Set vLLM environment variables from config
    policy_opt = job_config.policy_optimization
    if policy_opt.vllm_batch_invariant:
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
    os.environ["VLLM_ATTENTION_BACKEND"] = policy_opt.vllm_attention_backend

    # RL Training config
    num_steps = job_config.training.steps

    # Parallelism sizes
    trainer_ddp_size = job_config.parallelism.data_parallel_replicate_degree
    trainer_tp_size = job_config.parallelism.tensor_parallel_degree

    init_batch_invariance()
    batch_invariant = vllm_is_batch_invariant()

    # Set up batch invariant
    if batch_invariant:
        logger.info("Batch invariance detected - using vLLM-compatible model")
        from torchtitan.experiments.rl.vllm_compat.batch_invariant_backward import (
            enable_batch_invariant_backward_mode,
        )

        enable_batch_invariant_backward_mode()
    else:
        raise RuntimeError("Batch invariance NOT detected - using standard model")

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
    gen_mesh = this_host().spawn_procs(per_host={"gpus": 1})

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
        job_config,  # Pass full job_config
    )

    generator = gen_mesh.spawn(
        "generator",
        Generator,
        job_config,  # Pass full job_config
        prompt_texts,
        expected_answers,
    )

    # Initialize generator with trainer weights
    initial_weights = trainer.get_weights.call().get().item(gpus=0)
    await generator.update.call(0, initial_weights)

    # Training loop
    logger.info("\n" + "=" * 80)
    logger.info(f"Starting RL training for {num_steps} steps")
    logger.info("=" * 80)

    for step in range(num_steps):
        # Fully sync RL loop
        # NOTE: This is only getting Trajectory generated from trainer 0, and trainer 1's data is ignored.
        # .get() is a blocking method which makes the loop fully sync
        batch = generator.generate.call().get().item(gpus=0)
        metrics = trainer.step.call(batch).get().item(gpus=0)
        weights = trainer.get_weights.call().get().item(gpus=0)
        await generator.update.call(metrics["policy_version"], weights)

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
