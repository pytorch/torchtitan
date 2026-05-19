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
from torchtitan.experiments.rl.grpo import RLTrainer
from torchtitan.experiments.rl.loss import DAPOLoss
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
    clip_low: float = 0.2,
    clip_high: float = 0.28,
    dual_clip_c: float = 3.0,
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
        loss=DAPOLoss.Config(
            clip_low=clip_low, clip_high=clip_high, dual_clip_c=dual_clip_c
        ),
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


def rl_grpo_qwen3_0_6b_alphabet() -> RLTrainer.Config:
    """AlphabetSort smoke on Qwen3-0.6B — debug companion to the 1.7B gate.

    0.6B is more numerically stable in mixed-precision training, so this
    config is useful for distinguishing pipeline bugs (would NaN even at
    0.6B) from model-specific instability (1.7B-only NaN).
    """
    return RLTrainer.Config(
        model_spec=model_registry("0.6B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        num_steps=30,
        num_prompts_per_step=8,
        rollout_group_size=8,
        num_rollout_tasks=4,
        max_rollout_turns=5,
        num_validation_samples=20,
        log_samples=True,
        compile=CompileConfig(enable=True, backend="aot_eager"),
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
        trainer=_trainer(train_tp=2, lr=2e-6, warmup_steps=2),
        generator=_generator(gen_tp=2, max_tokens=768, gpu_memory_limit=0.7),
        replay_buffer=_replay(batch_size=4, max_buffer_size=256),
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
        # Bumped 8→16 prompts and 4→8 rollout tasks (codex's measured win
        # on the same 2-GPU 1.7B recipe at branch 939673a9d — mean batch
        # 37 → 70, max 64 → 128). Combined with the CP38 admission/
        # stepping split, this targets continuous batch >=64 instead of
        # batch=1 stalls between multi-turn boundaries.
        num_prompts_per_step=16,
        rollout_group_size=8,
        num_rollout_tasks=8,
        max_rollout_turns=8,
        # Halve weight-sync overhead — see 4B config; the ~25s
        # torchstore TCP-fallback pull was 50% of pre-CP38 step time.
        pull_every_n_steps=2,
        num_validation_samples=20,
        log_samples=True,
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
            train_tp=1,
            # Conservative for 1.7B + batch=4: prime-rl uses lr=1e-5 at
            # batch=64. Step 3 -> 4 saw Loss=NaN at lr=1e-5 batch=8.
            lr=2e-6,
            warmup_steps=0,
            checkpoint_interval=50,
        ),
        # gpu_memory_limit bumped from 0.2 to 0.7 — Opus has GPUs 4,5
        # exclusively (no shared tenants on this branch), so vLLM can
        # claim ~70GB of the 97GB card for KV cache. More KV cache →
        # more concurrent in-flight requests in vLLM's scheduler →
        # bigger continuous batches when combined with CP38 + CP39.
        generator=_generator(gen_tp=1, max_tokens=768, gpu_memory_limit=0.7),
        replay_buffer=_replay(batch_size=4, max_buffer_size=512),
    )


def rl_grpo_qwen3_4b_alphabet() -> RLTrainer.Config:
    """AlphabetSort gate — Qwen3-4B-Instruct-2507, 25 steps, 2 GPUs.

    Single-GPU trainer + single-GPU generator (TP=1 both) to fit the
    Opus 2-GPU budget. Mirrors prime-rl's recipe
    (``frameworks/prime-rl/examples/alphabet_sort/rl.toml``) as closely
    as a full-finetune (no LoRA) 2-GPU run can:

    | knob | prime-rl | here | reason for delta |
    | --- | --- | --- | --- |
    | model | Qwen3-4B-Instruct-2507 | same | identical |
    | lr | 1e-5 (LoRA) | 5e-6 | LoRA-equivalent step ~2x our full-FT |
    | batch_size | 512 prompts | 4 prompts | full-FT 4B fits batch=4 in 1xH100 |
    | rollouts_per_example | 8 | 8 | identical |
    | max_completion_tokens | 768 | 768 | identical |
    | min/max_turns | 3/5 | 3/5 | identical |
    | similarity_power (train) | 4 | 4 | identical |
    | power_per_turn | false | false | identical |
    | loss | DAPO | DAPO | identical (clip 0.2/0.28, dual c=3.0) |
    | num_rollout_tasks | sync | 4 | concurrent producers feeding the buffer |

    25 steps as a fast first-pass; bump to 100 for the full prime-rl
    reproduction (0.30 -> ~0.85) once 25 looks healthy.
    """
    return RLTrainer.Config(
        model_spec=model_registry("4B", attn_backend="varlen"),
        hf_assets_path=(
            "torchtitan/experiments/rl/example_checkpoint/Qwen3-4B-Instruct-2507"
        ),
        num_steps=25,
        # Bumped to feed the CP38 continuous-batching admission window:
        # 4B prompts × 8 group_size × 8 tasks = 256 max in-flight rollouts,
        # well above what vLLM's 2-GPU scheduler can actually run
        # concurrently (~64-128) but ensuring the queue never goes empty.
        # Without CP38 + CP39 the engine drains to batch=1 between turn
        # boundaries; with them we target mean batch ~70 (codex measured).
        num_prompts_per_step=8,
        rollout_group_size=8,
        num_rollout_tasks=8,
        max_rollout_turns=5,
        # Amortize the ~25s torchstore TCP-fallback weight pull across
        # 2 train steps (pull every other step instead of every step).
        # Net effect: ~50% reduction in pull overhead, generator sees
        # weights at most 1 train step stale. DAPO's clipping handles
        # this staleness fine; if convergence breaks, drop back to 1.
        pull_every_n_steps=2,
        num_validation_samples=20,
        log_samples=True,
        # Same s59 empty-sources torch.compile bug as 1.7B layer shapes.
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
            train_tp=1,
            lr=5e-6,
            # Prime-RL uses ConstantSchedulerConfig — no warmup, no
            # decay. Set warmup=0 to match. ``_trainer`` only exposes a
            # linear decay, but at 25-100 steps with lr=5e-6 the
            # decay slope is small enough vs constant.
            warmup_steps=0,
            checkpoint_interval=25,
        ),
        # gpu_memory_limit=0.7 — see 1.7B alphabet for rationale; this
        # branch has exclusive use of GPUs 4,5, so vLLM can claim ~70GB
        # for KV cache (vs the 0.2 used during early shared-tenant runs).
        generator=_generator(gen_tp=1, max_tokens=768, gpu_memory_limit=0.7),
        replay_buffer=_replay(batch_size=4, max_buffer_size=512),
    )
