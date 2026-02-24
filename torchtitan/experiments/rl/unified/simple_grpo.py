# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Multiprocess RL training loop using Monarch Actors.

This demonstrates:
1. Distributed actor architecture with Generator (vLLM) and PolicyTrainer (TorchTitan) components
2. File based weight synchronization between trainer and generator
3. Separate scoring component for reward and advantage computation

The architecture mirrors monarch's grpo_actor.py but adapted for vLLM rollouts + TorchTitan training.

Command to run:
python3 torchtitan/experiments/rl/unified/simple_grpo.py \
    --module rl.unified --config rl_grpo_qwen3_0_6b \
    --trainer.checkpoint.initial-load-path=<path_to_model_checkpoint>
"""

import asyncio
import logging
from dataclasses import dataclass, field

import torch

from monarch.actor import this_host
from monarch.utils import setup_env_for_distributed
from torchtitan.config import Configurable
from torchtitan.config.manager import ConfigManager
from torchtitan.experiments.rl.unified.actors.generator import Generator
from torchtitan.experiments.rl.unified.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.unified.configs import PolicyOptimizationConfig
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.experiments.rl.unified.actors.grader import Grader

logger = logging.getLogger(__name__)


class RLTrainer(Configurable):
    """Top-level RL training orchestrator.

    Composes a ``PolicyTrainer`` (for model construction, parallelism,
    optimizer, checkpoint, etc.) with RL-specific settings and a
    Generator Monarch actor.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Top-level config for RL training.

        ``trainer`` holds the PolicyTrainer.Config that handles model
        construction, parallelism, optimizer, LR scheduler, checkpoint, etc.
        The remaining fields are RL-specific.
        """

        model_spec: ModelSpec | None = None
        """Model specification shared by trainer and generator.
        Set programmatically via config_registry (not from CLI)."""

        hf_assets_path: str = "./tests/assets/tokenizer"
        """Path to HF assets folder (model weights, tokenizer, config files)."""

        trainer: PolicyTrainer.Config = field(default_factory=PolicyTrainer.Config)
        """PolicyTrainer config. Controls optimizer, training, parallelism,
        lr_scheduler, checkpoint, activation_checkpoint."""

        num_steps: int = 10
        """Number of RL training steps."""

        dump_folder: str = "outputs/rl"
        """Root output folder for RL artifacts (temp weights, logs, etc.)."""

        batch_invariant_mode: bool = True
        """Enable batch-invariant mode for deterministic NCCL collective
        operations and bitwise-reproducible forward/backward passes."""

        policy_optimization: PolicyOptimizationConfig = field(
            default_factory=PolicyOptimizationConfig
        )
        """Policy optimization hyperparameters."""

        generator: Generator.Config = field(default_factory=Generator.Config)
        """Generator actor configuration (vLLM engine, sampling)."""


async def main():
    """Run the distributed RL training loop using Monarch."""

    config = ConfigManager().parse_args()

    # Patch model_spec to use the RL-specific parallelize function.
    # The canonical model_registry returns the standard qwen3 parallelize_fn,
    # but RL training needs the version with inner_attention hooks
    # (PrepareModuleInputOutput) to convert DTensors to local tensors for
    # vLLM's flash attention kernels.
    from torchtitan.experiments.rl.unified.models.parallelize import parallelize_qwen3

    config.model_spec.parallelize_fn = parallelize_qwen3

    trainer_cfg = config.trainer

    # Validate that trainer and generator have the same parallel plan
    # since they are collocated on the same mesh
    assert trainer_cfg.parallelism == config.generator.vllm_engine.parallelism, (
        f"Trainer and generator must use the same parallel plan.\n"
        f"  Trainer:   {trainer_cfg.parallelism}\n"
        f"  Generator: {config.generator.vllm_engine.parallelism}"
    )

    trainer_world_size = (
        trainer_cfg.parallelism.data_parallel_replicate_degree
        * trainer_cfg.parallelism.tensor_parallel_degree
    )

    # RL Training config
    num_steps = config.num_steps

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
        PolicyTrainer,
        config.trainer,
        config.policy_optimization,
        config.model_spec,
        config.hf_assets_path,
    )

    # Spawn grader on trainer mesh (no collective ops in init, safe to co-spawn)
    grader = trainer_mesh.spawn(
        "grader",
        Grader,
        config.policy_optimization,
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
        model_spec=config.model_spec,
        model_path=config.hf_assets_path,
        dump_folder=config.dump_folder,
        batch_invariant_mode=config.batch_invariant_mode,
        policy_optimization=config.policy_optimization,
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
