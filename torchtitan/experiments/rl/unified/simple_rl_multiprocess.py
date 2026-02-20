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
import time

import torch
from monarch.actor import this_host
from monarch.utils import setup_env_for_distributed
from torchtitan.experiments.rl.metrics import (
    CumulativeMetrics,
    PhaseMetrics,
    print_step_metrics,
    print_training_summary,
    update_cumulative_metrics,
)
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

    # Metrics config
    cuda_sync_for_metrics = True
    enable_profiling = True  # Set to False to disable timing/memory output

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
        cuda_sync_for_metrics,
        enable_profiling,
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
        cuda_sync_for_metrics,
        enable_profiling,
    )

    # Initialize generator with trainer weights
    initial_weights = trainer.get_weights.call().get().item(gpus=0)
    await generator.update.call(0, initial_weights)

    # Training loop
    logger.info("\n" + "=" * 80)
    logger.info(f"Starting RL training for {num_steps} steps")
    logger.info("=" * 80)

    cumulative = CumulativeMetrics()

    for step in range(num_steps):
        if enable_profiling:
            step_t0 = time.perf_counter()

            # Rollout phase
            rollout_t0 = time.perf_counter()
        batch = generator.generate.call().get().item(gpus=0)
        if enable_profiling:
            rollout_time_s = time.perf_counter() - rollout_t0

            # Training phase
            train_t0 = time.perf_counter()
        metrics = trainer.step.call(batch).get().item(gpus=0)
        if enable_profiling:
            train_time_s = time.perf_counter() - train_t0

            # Weight sync phase
            wsync_t0 = time.perf_counter()
        weights = trainer.get_weights.call().get().item(gpus=0)
        await generator.update.call(metrics["policy_version"], weights)
        if enable_profiling:
            weight_sync_time_s = time.perf_counter() - wsync_t0
            step_time_s = time.perf_counter() - step_t0

            # Extract timing/memory from actor metrics
            gen_peak_gib = getattr(batch, "gen_peak_active_gib", 0.0)
            gen_peak_pct = getattr(batch, "gen_peak_active_pct", 0.0)
            train_peak_gib = metrics.get("train_peak_active_gib", 0.0)
            train_peak_pct = metrics.get("train_peak_active_pct", 0.0)
            optimizer_time_s = metrics.get("optimizer_time_s", 0.0)
            optimizer_peak_gib = metrics.get("optimizer_peak_active_gib", 0.0)
            optimizer_peak_pct = metrics.get("optimizer_peak_active_pct", 0.0)

            # Accumulate timing using helper function
            update_cumulative_metrics(
                cumulative,
                rollout_time_s,
                train_time_s,
                optimizer_time_s,
                weight_sync_time_s,
                step_time_s,
                rollout_peak_gib=gen_peak_gib,
                train_peak_gib=train_peak_gib,
                optimizer_peak_gib=optimizer_peak_gib,
                weight_sync_peak_gib=0.0,
            )

        logger.info(
            f"\nStep {step:3d} | Loss: {metrics['loss']:.4f} | "
            f"Reward: {metrics['reward_mean']:+.3f}"
        )

        if enable_profiling:
            # Per-phase timing and memory breakdown
            print_step_metrics(
                [
                    PhaseMetrics("rollout", rollout_time_s, gen_peak_gib, gen_peak_pct),
                    PhaseMetrics("train", train_time_s, train_peak_gib, train_peak_pct),
                    PhaseMetrics(
                        "optimizer",
                        optimizer_time_s,
                        optimizer_peak_gib,
                        optimizer_peak_pct,
                    ),
                    PhaseMetrics(
                        "weight_sync",
                        weight_sync_time_s,
                        0.0,
                        0.0,
                    ),
                ],
                step_time_s,
                include_mem_pct=True,
            )

        logger.info(f"  Sample: {metrics['sample_completion']}...")

        # Check for divergence
        if not torch.isfinite(torch.tensor(metrics["loss"])):
            logger.info("\n" + "!" * 80)
            logger.info("ERROR: Loss is NaN/Inf! Training diverged.")
            logger.info("!" * 80)
            break

    # Training summary
    if enable_profiling:
        print_training_summary(cumulative, only_print_memory_if_nonzero=True)

    logger.info("\n" + "=" * 80)
    logger.info("RL Training complete")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
