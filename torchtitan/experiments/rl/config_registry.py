# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Config entry points for the RL/unified experiment.

Each function returns a complete ``RLTrainer.Config`` and is discoverable by
``ConfigManager`` via ``--module rl --config <function_name>``.
"""

import dataclasses
from dataclasses import dataclass

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import (
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.experiments.rl.actors.generator import SamplingConfig, VLLMGenerator
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.batcher import BatchConfig, Batcher
from torchtitan.experiments.rl.examples.alphabet_sort import AlphabetSortRollouter
from torchtitan.experiments.rl.generator_router import (
    GeneratorRouter,
    RoundRobinRoutingStrategy,
)
from torchtitan.experiments.rl.observability.metrics import MetricsProcessor
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.trainer import GRPOLoss, RLTrainer
from torchtitan.models.common.attention import (
    FlexAttention,
    MaskMod,
    SlidingWindowMaskMod,
)
from torchtitan.models.qwen3 import model_registry
from torchtitan.protocols.model import ModelConfigConverter

_BATCH_INVARIANT_DEBUG = DebugConfig(batch_invariant=True, deterministic=True)


class FlexConverter(ModelConfigConverter):
    """Configure FlexAttention layers for the RL loop, applied per flex layer.

    - ``batch_invariant``: pin kernel options (BACKEND=TRITON, BLOCK_M/N=16) so
      the trainer's flex kernel is batch-invariant and matches vLLM's tile size.
      (BACKEND=TRITON avoids the flex_decode kernel.) This is only the
      model-side piece of batch invariance; the trainer/generator debug flags
      and batcher padding are applied by the config builder.
    - ``mask_mod``: a ``MaskMod.Config``set as the within-sequence
    ``mask_mod`` on every flex layer.

    Both are independent and may be combined.
    """

    # the triton BLOCK_N tile size needs to be pinned for stable numerics and
    # needs to match vLLM's for identical results, today vLLM default is 16
    # TODO: run some experiments to determine impact of small vs large tile sizes
    _BLOCK_M = 16
    _BLOCK_N = 16

    @dataclass(kw_only=True, slots=True)
    class Config(ModelConfigConverter.Config):
        batch_invariant: bool = False
        mask_mod: MaskMod.Config | None = None

    def __init__(self, config: Config):
        self.batch_invariant = config.batch_invariant
        self.mask_mod = config.mask_mod

    def convert(self, model_config) -> None:
        for layer_cfg in model_config.layers:
            inner = layer_cfg.attention.inner_attention
            if not isinstance(inner, FlexAttention.Config):
                continue
            if self.batch_invariant:
                inner.kernel_options["BACKEND"] = "TRITON"
                inner.kernel_options["BLOCK_M"] = self._BLOCK_M
                inner.kernel_options["BLOCK_N"] = self._BLOCK_N
            if self.mask_mod is not None:
                inner.mask_mod = self.mask_mod


def rl_grpo_qwen3_0_6b_varlen() -> RLTrainer.Config:
    """GRPO training config for Qwen3-0.6B (6 GPUs: 4 gen + 2 train)."""
    group_size = 8
    return RLTrainer.Config(
        model_spec=model_registry("0.6B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        num_steps=10,
        num_groups_per_rollout_batch=5,
        num_validation_samples=20,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=AlphabetSortRollouter.Config(),
        group_size=group_size,
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        generator_router=GeneratorRouter.Config(
            strategy=RoundRobinRoutingStrategy.Config()
        ),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        batcher=Batcher.Config(
            batch=BatchConfig(local_batch_size=2, global_batch_size=8, seq_len=2048),
        ),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=ParallelismConfig(
                tensor_parallel_degree=4,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=0.95,
                max_tokens=700,
            ),
        ),
    )


def _rl_grpo_qwen3_0_6b_flex(
    *,
    batch_invariant: bool = False,
    mask_mod: MaskMod.Config | None = None,
    sliding_window: int | None = None,
) -> RLTrainer.Config:
    """Shared builder for the Qwen3-0.6B flex-attention configs (4 GPUs: 2 gen + 2 train).

    ``mask_mod`` and ``sliding_window`` are two independent ways to apply a
    within-sequence mask, and are mutually exclusive:

    Args:
        batch_invariant: Pin flex kernel options + enable deterministic /
            batch-invariant debug on trainer and generator (bitwise parity).
        mask_mod: Custom ``MaskMod.Config`` (e.g. ``SlidingWindowMaskMod.Config``)
            applied on every flex layer
        sliding_window: Native sliding-window size (tokens)
    """
    group_size = 8
    if mask_mod is not None:
        trainer_mask_mod = mask_mod
    elif sliding_window is not None:
        trainer_mask_mod = SlidingWindowMaskMod.Config(window_size=sliding_window)
    else:
        raise ValueError("cannot pass in both mask_mod and sliding_window ")
    model_spec = model_registry(
        "0.6B",
        attn_backend="flex",
        converters=[
            FlexConverter.Config(
                batch_invariant=batch_invariant,
                mask_mod=trainer_mask_mod,
            )
        ],
    )
    config = RLTrainer.Config(
        model_spec=model_spec,
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        num_steps=10,
        num_groups_per_rollout_batch=5,
        num_validation_samples=20,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=AlphabetSortRollouter.Config(),
        group_size=group_size,
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        batcher=Batcher.Config(
            batch=BatchConfig(local_batch_size=2, global_batch_size=8, seq_len=2048),
        ),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(dtype="bfloat16"),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=ParallelismConfig(
                tensor_parallel_degree=2,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=0.95,
                max_tokens=100,
            ),
            sliding_window=sliding_window,
        ),
    )

    if batch_invariant:
        block_size = model_spec.model.layers[0].attention.inner_attention.block_size
        config.batcher = dataclasses.replace(
            config.batcher, per_sample_pad_multiple=block_size
        )
        config.trainer = dataclasses.replace(
            config.trainer,
            debug=_BATCH_INVARIANT_DEBUG,
            parallelism=dataclasses.replace(
                config.trainer.parallelism, enable_sequence_parallel=False
            ),
        )
        config.generator = dataclasses.replace(
            config.generator, debug=_BATCH_INVARIANT_DEBUG
        )
    return config


def rl_grpo_qwen3_0_6b_flex() -> RLTrainer.Config:
    """GRPO training config for Qwen3-0.6B with flex attention (4 GPUs: 2 gen + 2 train)."""
    return _rl_grpo_qwen3_0_6b_flex()


def rl_grpo_qwen3_0_6b_flex_batch_invariant() -> RLTrainer.Config:
    """``rl_grpo_qwen3_0_6b_flex`` + batch invariance, for bitwise-identical
    trainer/generator numerics (4 GPUs: 2 gen + 2 train)."""
    return _rl_grpo_qwen3_0_6b_flex(batch_invariant=True)


def rl_grpo_qwen3_0_6b_flex_custom_mask_mod() -> RLTrainer.Config:
    """``rl_grpo_qwen3_0_6b_flex`` + a custom mask_mod (sliding window)"""
    return _rl_grpo_qwen3_0_6b_flex(
        mask_mod=SlidingWindowMaskMod.Config(window_size=64)
    )


def rl_grpo_qwen3_0_6b_flex_batch_invariant_custom_mask_mod() -> RLTrainer.Config:
    """``rl_grpo_qwen3_0_6b_flex_batch_invariant`` + a custom mask_mod (sliding window)"""
    return _rl_grpo_qwen3_0_6b_flex(
        batch_invariant=True, mask_mod=SlidingWindowMaskMod.Config(window_size=64)
    )


def rl_grpo_qwen3_0_6b_flex_sliding_window() -> RLTrainer.Config:
    """``rl_grpo_qwen3_0_6b_flex`` + a causal sliding window via vLLM's native
    ``per_layer_sliding_window`` spec (SlidingWindowSpec + KV-block pruning).
    """
    return _rl_grpo_qwen3_0_6b_flex(sliding_window=64)


def rl_grpo_qwen3_0_6b_flex_batch_invariant_sliding_window() -> RLTrainer.Config:
    """``rl_grpo_qwen3_0_6b_flex_batch_invariant`` + a causal sliding window via
    vLLM's native ``per_layer_sliding_window`` spec (KV-block pruning
    """
    return _rl_grpo_qwen3_0_6b_flex(batch_invariant=True, sliding_window=64)


def rl_grpo_qwen3_1_7b() -> RLTrainer.Config:
    """GRPO training config for Qwen3-1.7B (6 GPUs: 4 gen + 2 train)."""
    group_size = 8
    return RLTrainer.Config(
        model_spec=model_registry("1.7B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-1.7B",
        num_steps=10,
        num_groups_per_rollout_batch=5,
        num_validation_samples=20,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=AlphabetSortRollouter.Config(),
        group_size=group_size,
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        batcher=Batcher.Config(
            batch=BatchConfig(local_batch_size=2, global_batch_size=8, seq_len=2048),
        ),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=4,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=0.95,
                max_tokens=700,
            ),
        ),
    )


def rl_grpo_qwen3_14b() -> RLTrainer.Config:
    """GRPO training config for Qwen3-14B (16 GPUs: 8 gen + 8 train)."""
    group_size = 8
    return RLTrainer.Config(
        model_spec=model_registry("14B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-14B",
        num_steps=10,
        num_groups_per_rollout_batch=5,
        num_validation_samples=20,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=AlphabetSortRollouter.Config(),
        group_size=group_size,
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        batcher=Batcher.Config(
            batch=BatchConfig(local_batch_size=2, global_batch_size=8, seq_len=2048),
        ),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=1e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(dtype="bfloat16"),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=8,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=ParallelismConfig(
                tensor_parallel_degree=8,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=0.95,
                max_tokens=700,
            ),
        ),
    )


def rl_grpo_qwen3_0_6b_varlen_batch_invariant() -> RLTrainer.Config:
    """On-policy GRPO config for Qwen3-0.6B (4 GPUs: 2 gen + 2 train).

    Enables deterministic + batch-invariant mode for true on-policy RL training.
    """
    batch_invariant_config = DebugConfig(batch_invariant=True, deterministic=True)
    group_size = 8
    return RLTrainer.Config(
        model_spec=model_registry("0.6B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        num_steps=10,
        num_groups_per_rollout_batch=5,
        num_validation_samples=20,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=AlphabetSortRollouter.Config(),
        group_size=group_size,
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        batcher=Batcher.Config(
            batch=BatchConfig(local_batch_size=2, global_batch_size=8, seq_len=2048),
        ),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            # bfloat16 is needed for trainer to align with generator dtype
            # TODO: replace bfloat16 enablement with FSDP2+TP2
            training=TrainingConfig(dtype="bfloat16"),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            debug=batch_invariant_config,
            loss=GRPOLoss.Config(),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=ParallelismConfig(
                tensor_parallel_degree=2,
                data_parallel_replicate_degree=1,
                enable_sequence_parallel=False,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=0.95,
                max_tokens=700,
            ),
            debug=batch_invariant_config,
        ),
    )
