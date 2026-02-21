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
VLLM_BATCH_INVARIANT=1 VLLM_ATTENTION_BACKEND=FLASH_ATTN python3 torchtitan/experiments/rl/unified/simple_rl_multiprocess.py
"""
import asyncio
import logging

import torch
from monarch.actor import this_host
from monarch.utils import setup_env_for_distributed
from torchtitan.experiments.rl.unified.actors.generator import Generator
from torchtitan.experiments.rl.unified.actors.trainer import Trainer
from torchtitan.experiments.rl.unified.models.utils import ModelMode
from torchtitan.experiments.rl.vllm_compat.simple_rl import (
    download_and_convert_model,
    load_gsm8k_dataset,
)
from vllm.model_executor.layers.batch_invariant import (
    init_batch_invariance,
    vllm_is_batch_invariant,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum

logger = logging.getLogger(__name__)


async def main():
    """Run the distributed RL training loop using Monarch."""
    # Model Config
    model_name = "Qwen/Qwen3-0.6B"
    cache_dir = "./models"
    output_dir = "./converted"

    # Training config
    group_size = 8
    num_steps = 10
    learning_rate = 1e-5
    max_new_tokens = 20

    # GRPO config
    use_stable_grpo = False
    grpo_beta = 0.1

    # Dataset config
    use_real_dataset = False
    num_dataset_samples = 5

    # Parallelism sizes
    trainer_ddp_size = 2
    trainer_tp_size = 1
    generator_tp_size = 1

    # vLLM compilation config
    vllm_compile_and_cudagraph = True

    init_batch_invariance(AttentionBackendEnum.FLASH_ATTN)
    batch_invariant = vllm_is_batch_invariant()
    mode = ModelMode.UNIFIED

    # Set up batch invariant
    if batch_invariant:
        logger.info("Batch invariance detected - using vLLM-compatible model")
        from torchtitan.experiments.rl.vllm_compat.batch_invariant_backward import (
            enable_batch_invariant_backward_mode,
        )

        enable_batch_invariant_backward_mode()
    else:
        raise RuntimeError("Batch invariance NOT detected - using standard model")

    # Download and convert model
    titan_checkpoint_path, model_path = download_and_convert_model(
        model_name, cache_dir, output_dir
    )

    # Load dataset
    if use_real_dataset:
        logger.info(f"Loading GSM8K dataset ({num_dataset_samples} samples)...")
        # TODO: Refactor into loading torchtitan dataset
        prompt_texts, expected_answers = load_gsm8k_dataset(
            split="train", num_samples=num_dataset_samples
        )
        if prompt_texts is None or len(prompt_texts) == 0:
            use_real_dataset = False

    if not use_real_dataset:
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
    trainer_mesh = this_host().spawn_procs(per_host={"gpus": 2})
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
        titan_checkpoint_path,
        model_path,
        learning_rate,
        mode,
        trainer_ddp_size,
        trainer_tp_size,
    )

    generator = gen_mesh.spawn(
        "generator",
        Generator,
        model_path,
        prompt_texts,
        expected_answers,
        group_size,
        max_new_tokens,
        1.0,  # temperature
        use_real_dataset,
        grpo_beta,
        use_stable_grpo,
        generator_tp_size,
        vllm_compile_and_cudagraph,
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
