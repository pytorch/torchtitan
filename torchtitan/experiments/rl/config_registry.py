# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config entrypoints for the RL experiment.

Each function returns a complete :class:`RLTrainer.Config` and is
discoverable via ``--module rl --config <function_name>``. The
``_trainer``, ``_generator``, and ``_replay`` helpers factor the
shared boilerplate so each entry point only shows its actual
choices (model size, dataset, parallelism, learning rate).
"""

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import (
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.experiments.rl.actors.generator import SamplingConfig, VLLMGenerator
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.alphabet_sort import (
    AlphabetSortBuilder,
    AlphabetSortDataset,
)
from torchtitan.experiments.rl.envs.token_env import TokenEnvConfig
from torchtitan.experiments.rl.grpo import GRPOLoss, RLTrainer
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.replay import ReplayBuffer
from torchtitan.experiments.rl.sum_digits import SumDigitsBuilder, SumDigitsDataset
from torchtitan.models.qwen3 import model_registry


# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------


def _trainer(
    *,
    train_tp: int,
    lr: float,
    warmup_steps: int,
    clip_eps: float = 0.2,
    training_dtype: str = "float32",
    checkpoint_interval: int = 10,
    debug: DebugConfig | None = None,
) -> PolicyTrainer.Config:
    return PolicyTrainer.Config(
        optimizer=OptimizersContainer.Config(lr=lr),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=warmup_steps,
            decay_type="linear",
        ),
        training=TrainingConfig(dtype=training_dtype),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=1,
            tensor_parallel_degree=train_tp,
            # SP off: batch_invariant mode requires it off (NCCL reduce-
            # scatter Ring is the only deterministic mode), and non-
            # batch-invariant runs at TP=2 don't gain enough from SP to
            # justify the inconsistent default.
            enable_sequence_parallel=False,
            disable_loss_parallel=True,
        ),
        checkpoint=CheckpointManager.Config(
            enable=True,
            initial_load_in_hf=True,
            interval=checkpoint_interval,
            last_save_model_only=False,
        ),
        loss=GRPOLoss.Config(clip_eps=clip_eps),
        debug=debug or DebugConfig(),
    )


def _generator(
    *,
    gen_tp: int,
    max_tokens: int,
    gpu_memory_limit: float = 0.9,
    debug: DebugConfig | None = None,
) -> VLLMGenerator.Config:
    return VLLMGenerator.Config(
        model_dtype="bfloat16",
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=1,
            tensor_parallel_degree=gen_tp,
            data_parallel_replicate_degree=1,
            enable_sequence_parallel=False,
            disable_loss_parallel=True,
        ),
        checkpoint=CheckpointManager.Config(enable=False),
        gpu_memory_limit=gpu_memory_limit,
        sampling=SamplingConfig(
            n=1,  # GRPO siblings come from rollout_group_size, not n.
            temperature=0.8,
            top_p=0.95,
            max_tokens=max_tokens,
        ),
        debug=debug or DebugConfig(),
    )


def _replay(
    *,
    batch_size: int = 5,
    max_policy_age: int | None = 1,
    max_buffer_size: int = 256,
) -> ReplayBuffer.Config:
    return ReplayBuffer.Config(
        batch_size=batch_size,
        dp_size=1,
        max_policy_age=max_policy_age,
        max_buffer_size=max_buffer_size,
    )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def rl_grpo_qwen3_0_6b() -> RLTrainer.Config:
    """GRPO config for Qwen3-0.6B (6 GPUs: 4 gen + 2 train) — SumDigits smoke.

    10-step gate. Reward starts ~0.3, should reach ~0.7 by step 10.
    """
    return RLTrainer.Config(
        model_spec=model_registry("0.6B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        num_steps=10,
        num_prompts_per_step=5,
        rollout_group_size=8,
        num_rollout_tasks=4,
        max_rollout_turns=1,
        num_validation_samples=20,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        train_dataset=SumDigitsDataset.Config(seed=42),
        train_builder=SumDigitsBuilder.Config(),
        validation_dataset=SumDigitsDataset.Config(seed=99),
        validation_builder=SumDigitsBuilder.Config(),
        renderer=RendererConfig(name="auto"),
        trainer=_trainer(train_tp=2, lr=2e-6, warmup_steps=2),
        generator=_generator(gen_tp=4, max_tokens=512),
        replay_buffer=_replay(),
    )


def rl_grpo_qwen3_1_7b() -> RLTrainer.Config:
    """GRPO config for Qwen3-1.7B (4 GPUs: 2 gen + 2 train) — SumDigits smoke.

    Encourages brief thinking (Qwen3 chat template emits ``<think>``) so
    the model has scratch space before ``<answer>``, then a larger
    ``max_tokens`` budget so it actually finishes inside the cap.

    Compile disabled: torch.compile aot_eager backend hits the empty
    sources assertion (``s59``) on the 1.7B layer shapes, which yields
    inconsistent graphs across DP ranks and a downstream NCCL timeout.
    The 0.6B config compiles fine; this is model-specific to 1.7B.
    """
    short_thinking_prompt = (
        "You are a careful arithmetic assistant. Given a list of integers, "
        "compute the sum of all their digits. Think briefly (≤30 tokens) "
        "and then respond with exactly `<answer>NUMBER</answer>` where "
        "NUMBER is the digit sum."
    )
    return RLTrainer.Config(
        model_spec=model_registry("1.7B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-1.7B",
        num_steps=10,
        num_prompts_per_step=5,
        rollout_group_size=8,
        num_rollout_tasks=4,
        max_rollout_turns=1,
        num_validation_samples=20,
        log_samples=True,
        compile=CompileConfig(enable=False),
        train_dataset=SumDigitsDataset.Config(seed=42),
        train_builder=SumDigitsBuilder.Config(system_prompt=short_thinking_prompt),
        validation_dataset=SumDigitsDataset.Config(seed=99),
        validation_builder=SumDigitsBuilder.Config(system_prompt=short_thinking_prompt),
        renderer=RendererConfig(name="auto"),
        trainer=_trainer(train_tp=2, lr=2e-6, warmup_steps=2),
        generator=_generator(gen_tp=2, max_tokens=1024, gpu_memory_limit=0.7),
        replay_buffer=_replay(),
    )


def rl_grpo_qwen3_0_6b_batch_invariant() -> RLTrainer.Config:
    """On-policy GRPO config for Qwen3-0.6B (4 GPUs: 2 gen + 2 train).

    Enables deterministic + batch-invariant mode for true on-policy RL.
    Trainer and generator use the same TP degree.
    """
    debug = DebugConfig(batch_invariant=True, deterministic=True)
    return RLTrainer.Config(
        model_spec=model_registry("0.6B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        num_steps=10,
        num_prompts_per_step=5,
        rollout_group_size=8,
        num_rollout_tasks=2,
        max_rollout_turns=1,
        num_validation_samples=4,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        train_dataset=SumDigitsDataset.Config(seed=42),
        train_builder=SumDigitsBuilder.Config(),
        validation_dataset=SumDigitsDataset.Config(seed=99),
        validation_builder=SumDigitsBuilder.Config(),
        renderer=RendererConfig(name="auto"),
        trainer=_trainer(
            train_tp=2,
            lr=2e-6,
            warmup_steps=2,
            training_dtype="bfloat16",
            debug=debug,
        ),
        generator=_generator(gen_tp=2, max_tokens=512, debug=debug),
        replay_buffer=_replay(max_policy_age=0, max_buffer_size=128),
    )


def rl_grpo_qwen3_1_7b_alphabet() -> RLTrainer.Config:
    """AlphabetSort acceptance gate — Qwen3-1.7B, 100 steps, 4-GPU split.

    Multi-turn (3-5 turns/episode), reward 0.0-1.0 via Ratcliff/Obershelp
    similarity. Holistic-power mode (matches prime-rl's run config).
    Target: avg reward materially higher than the baseline; aim for
    >=0.5 by step 100.
    """
    return RLTrainer.Config(
        model_spec=model_registry("1.7B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-1.7B",
        num_steps=100,
        num_prompts_per_step=8,
        rollout_group_size=8,
        num_rollout_tasks=4,
        max_rollout_turns=8,
        num_validation_samples=20,
        # Same torch.compile aot_eager s59-empty-sources bug as the 1.7B
        # SumDigits config; disabled here too.
        compile=CompileConfig(enable=False),
        train_dataset=AlphabetSortDataset.Config(
            seed=1337420,
            min_turns=3,
            max_turns=5,
            min_names_per_turn=1,
            max_names_per_turn=5,
        ),
        train_builder=AlphabetSortBuilder.Config(
            similarity_power=4,
            power_per_turn=False,
        ),
        validation_dataset=AlphabetSortDataset.Config(
            seed=99,
            min_turns=3,
            max_turns=3,
            min_names_per_turn=1,
            max_names_per_turn=4,
        ),
        validation_builder=AlphabetSortBuilder.Config(
            similarity_power=8,
            power_per_turn=False,
        ),
        renderer=RendererConfig(name="auto"),
        token_env=TokenEnvConfig(
            error_reward=0.0,
            truncation_reward=0.0,
            max_trajectory_tokens=2048,
            max_generation_tokens=768,
        ),
        trainer=_trainer(
            train_tp=2,
            lr=1e-5,
            warmup_steps=4,
            checkpoint_interval=50,
        ),
        generator=_generator(gen_tp=2, max_tokens=768, gpu_memory_limit=0.7),
        replay_buffer=_replay(batch_size=8, max_buffer_size=512),
    )
