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
import re

import torch
from monarch.actor import this_host
from monarch.utils import setup_env_for_distributed
from torchtitan.experiments.rl.unified.actors.generator import Generator
from torchtitan.experiments.rl.unified.actors.trainer import Trainer
from torchtitan.experiments.rl.unified.models.utils import ModelMode
from torchtitan.experiments.rl.sum_digits import extract_answer, SumDigitsSpec
from torchtitan.experiments.rl.vllm_compat.simple_rl import download_and_convert_model
from vllm.model_executor.layers.batch_invariant import (
    init_batch_invariance,
    vllm_is_batch_invariant,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum

logger = logging.getLogger(__name__)


async def evaluate(
    generator,
    system_prompt: str,
    num_samples: int = 10,
    seed: int = 99,
    log_samples_per_step: bool = True,
):
    """Run evaluation using the Generator's vLLM engine.

    Args:
        generator: Generator actor
        system_prompt: System prompt for the task
        num_samples: Number of eval samples
        seed: RNG seed (different from training seed to avoid overlap)

    Returns:
        Dict with accuracy, correct, total, format_rate
    """
    spec = SumDigitsSpec(seed=seed)
    tasks = [spec.generate_task() for _ in range(num_samples)]
    prompts = [system_prompt + "\n\n" + t.question for t in tasks]

    # Batch generate all eval prompts in one vLLM call
    responses = generator.generate_text.call(prompts).get().item(gpus=0)

    correct = 0
    format_ok = 0
    for task, response_text in zip(tasks, responses):
        extracted = extract_answer(response_text)
        is_correct = extracted == task.correct_answer
        has_tag = bool(re.search(r"\[ANSWER\]", response_text))

        correct += int(is_correct)
        format_ok += int(has_tag)

        if log_samples_per_step:
            mark = "+" if is_correct else "-"
            logger.info(f"  [{mark}] Q: {task.question}")
            logger.info(f"       A: {response_text[:200]}")
            logger.info(f"       extracted={extracted} expected={task.correct_answer}")

    result = {
        "accuracy": correct / num_samples,
        "correct": correct,
        "total": num_samples,
        "format_rate": format_ok / num_samples,
        "format_ok": format_ok,
    }
    logger.info(
        f"Eval: Accuracy={result['accuracy']:.0%} ({result['correct']}/{result['total']}) "
        f"Format={result['format_rate']:.0%} ({result['format_ok']}/{result['total']})"
    )
    return result


async def main():
    """Run the distributed RL training loop using Monarch."""
    # Model Config
    model_name = "Qwen/Qwen3-0.6B"
    cache_dir = "./models"
    output_dir = "./converted"

    # Training config
    group_size = 8
    num_steps = 12
    learning_rate = 5e-6
    max_new_tokens = 256

    # GRPO config
    use_stable_grpo = False
    grpo_beta = 0.1

    # Whether to log samples per step in eval and training
    log_samples_per_step = False

    # Task config
    num_prompts = 5

    # Parallelism sizes
    trainer_ddp_size = 2
    trainer_tp_size = 1
    generator_tp_size = 1

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

    # Task spec
    task_spec = SumDigitsSpec(seed=42)
    system_prompt = task_spec.get_system_prompt()

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
        group_size,
        max_new_tokens,
        1.0,  # temperature
        grpo_beta,
        use_stable_grpo,
        generator_tp_size,
        log_samples_per_step,
    )

    # Initialize generator with trainer weights
    initial_weights = trainer.get_weights.call().get().item(gpus=0)
    await generator.update.call(0, initial_weights)

    # Pre-training evaluation
    eval_samples = 20
    logger.info("Evaluating pre-training baseline...")
    pre_eval = await evaluate(
        generator, system_prompt, num_samples=eval_samples, log_samples_per_step=log_samples_per_step
    )

    # Training loop
    logger.info("=" * 80)
    logger.info(f"Starting RL training for {num_steps} steps")
    logger.info("=" * 80)

    for step in range(num_steps):
        # Generate new prompts for every step
        prompt_texts = []
        expected_answers = []
        for _ in range(num_prompts):
            task = task_spec.generate_task()
            prompt_texts.append(system_prompt + "\n\n" + task.question)
            expected_answers.append(str(task.correct_answer))

        # Fully sync RL loop
        batch = generator.generate.call(prompt_texts, expected_answers).get().item(gpus=0)
        metrics = trainer.step.call(batch).get().item(gpus=0)
        weights = trainer.get_weights.call().get().item(gpus=0)
        await generator.update.call(metrics["policy_version"], weights)

        correct_count = sum(1 for r in batch.rewards.tolist() if r > 0)
        total_count = len(batch.rewards)
        logger.info(
            f"Step {step:3d} | Loss: {metrics['loss']:.4f} | "
            f"Reward: {metrics['reward_mean']:+.3f} | "
            f"Correct: {correct_count}/{total_count}"
        )
        # Check for divergence
        if not torch.isfinite(torch.tensor(metrics["loss"])):
            logger.info("\n" + "!" * 80)
            logger.info("ERROR: Loss is NaN/Inf! Training diverged.")
            logger.info("!" * 80)
            break

    # Post-training evaluation
    logger.info("RL Training complete")
    logger.info("Evaluating post-training performance...")
    post_eval = await evaluate(
        generator, system_prompt, num_samples=eval_samples, log_samples_per_step=log_samples_per_step
    )

    logger.info("=" * 80)
    logger.info(
        f"Pre-training:  Accuracy={pre_eval['accuracy']:.0%} ({pre_eval['correct']}/{pre_eval['total']}) "
        f"Format={pre_eval['format_rate']:.0%} ({pre_eval['format_ok']}/{pre_eval['total']})"
    )
    logger.info(
        f"Post-training: Accuracy={post_eval['accuracy']:.0%} ({post_eval['correct']}/{post_eval['total']}) "
        f"Format={post_eval['format_rate']:.0%} ({post_eval['format_ok']}/{post_eval['total']})"
    )
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
