# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verified training recipes for the alphabet-sort example.

Each function returns a complete ``Controller.Config``, discoverable by
``ConfigLoader`` via
``--module torchtitan_recipes.rl.alphabet_sort --config rl_grpo_*``.
"""

import dataclasses

from renderers import GptOssRendererConfig, Qwen3RendererConfig

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.components.renderer import from_renderers
from torchtitan.config import DebugConfig, OverrideConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import (
    BatchInvariantFlexConverter,
    ModelConfigConverter,
    MXFP8LinearConverter,
)
from torchtitan.distributed.activation_remat import FullActivationRematConfig
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.gpt_oss import build_model_config as build_gpt_oss_model_config
from torchtitan.models.qwen3 import build_model_config
from torchtitan.models.qwen3_5 import build_model_config as build_qwen3_5_model_config
from torchtitan.rl.controller.controller import (
    AsyncLoopConfig,
    Controller,
    ValidationConfig,
)
from torchtitan.rl.examples.alphabet_sort.data import AlphabetSortDataset
from torchtitan.rl.examples.alphabet_sort.env import AlphabetSortEnv
from torchtitan.rl.examples.alphabet_sort.rubric import RewardAlphabetSort
from torchtitan.rl.generator.generator import Generator
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.types import SamplingConfig
from torchtitan.rl.generator.vllm.config import VLLMCudaGraphConfig
from torchtitan.rl.generator.vllm.engine import VLLMEngine
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.rollout.data import IterableRLDataLoader
from torchtitan.rl.rollout.rollouter import Rollouter, RolloutWorker
from torchtitan.rl.rollout.rubric import Rubric
from torchtitan.rl.trainer.losses import GRPOLoss
from torchtitan.rl.trainer.trainer import Trainer

from torchtitan_recipes.rl.model_configs import build_rl_model_configs

_BATCH_INVARIANT_DEBUG = DebugConfig(batch_invariant=True, deterministic=True)

# TODO: Enable CUDA graphs for RL trainers after eager/graph numerics parity is
# verified.


def _alphabet_sort_rollouter_config() -> Rollouter.Config:
    return Rollouter.Config(
        training_dataloader=IterableRLDataLoader.Config(
            dataset=AlphabetSortDataset.Config(seed=42)
        ),
        validation_dataset=AlphabetSortDataset.Config(seed=99),
        worker=RolloutWorker.Config(
            rubric=Rubric.Config(reward_fns=[RewardAlphabetSort.Config(weight=1.0)]),
            message_env=AlphabetSortEnv.Config(),
        ),
    )


def _qwen3_rl_model_configs(
    flavor: str,
    *,
    seq_len: int,
    attn_backend: str,
    trainer_parallelism: ParallelismConfig,
    generator_parallelism: InferenceParallelismConfig,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> tuple[Decoder.Config, Decoder.Config]:
    """Qwen3 ``(trainer, generator)`` model configs for RL, with fp32 lm_head logits."""
    return build_rl_model_configs(
        build_model_config,
        flavor,
        trainer_parallelism=trainer_parallelism,
        generator_parallelism=generator_parallelism,
        seq_len=seq_len,
        attn_backend=attn_backend,
        converters=converters,
    )


def rl_grpo_qwen3_0_6b_varlen(*, seq_len: int = 2048) -> Controller.Config:
    """GRPO training config for Qwen3-0.6B (6 GPUs: 4 gen + 2 train)."""
    num_samples_per_prompt = 8
    trainer_parallelism = ParallelismConfig(
        data_parallel_shard_degree=1,
        tensor_parallel_degree=2,
    )
    generator_parallelism = InferenceParallelismConfig(
        data_parallel_degree=1,
        tensor_parallel_degree=4,
    )
    trainer_model, generator_model = _qwen3_rl_model_configs(
        "0.6B",
        seq_len=seq_len,
        attn_backend="varlen",
        trainer_parallelism=trainer_parallelism,
        generator_parallelism=generator_parallelism,
    )
    return Controller.Config(
        hf_assets_path="./assets/hf/Qwen3-0.6B",
        async_loop=AsyncLoopConfig(
            num_training_steps=10,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(steps=20),
        ),
        rollouter=_alphabet_sort_rollouter_config(),
        renderer=from_renderers(Qwen3RendererConfig(enable_thinking=False)),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=Trainer.Config(
            model=trainer_model,
            optim=Optim.Config(
                optimizer=OptimizersContainer.Config(
                    optimizers=[AdamW.Config(pattern=r".*", lr=2e-6)]
                ),
                lr_scheduler=LRSchedulersContainer.Config(
                    warmup_steps=2,
                    decay_type="linear",
                ),
            ),
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=2 * seq_len,
                max_context_length=seq_len,
            ),
            parallelism=trainer_parallelism,
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=GRPOLoss.Config(
                    global_vocab_size=decoder_vocab_size(trainer_model)
                ),
            ),
        ),
        generator=Generator.Config(
            model=generator_model,
            model_dtype="bfloat16",
            parallelism=generator_parallelism,
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=0.8,
                max_tokens=700,
            ),
        ),
    )


def rl_grpo_qwen3_0_6b_varlen_no_compile(*, seq_len: int = 2048) -> Controller.Config:
    config = rl_grpo_qwen3_0_6b_varlen(seq_len=seq_len)
    for model_config in (config.trainer.model, config.generator.model):
        model_config.local_compile.regions = []
    return config


def rl_grpo_qwen3_0_6b_flex() -> Controller.Config:
    """GRPO training config for Qwen3-0.6B with flex attention (4 GPUs: 2 gen + 2 train)."""
    num_samples_per_prompt = 8
    seq_len = 2048
    trainer_parallelism = ParallelismConfig(
        data_parallel_shard_degree=1,
        tensor_parallel_degree=2,
    )
    generator_parallelism = InferenceParallelismConfig(
        data_parallel_degree=1,
        tensor_parallel_degree=2,
    )
    trainer_model, generator_model = _qwen3_rl_model_configs(
        "0.6B",
        seq_len=seq_len,
        attn_backend="flex",
        trainer_parallelism=trainer_parallelism,
        generator_parallelism=generator_parallelism,
    )
    return Controller.Config(
        hf_assets_path="./assets/hf/Qwen3-0.6B",
        async_loop=AsyncLoopConfig(
            num_training_steps=10,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(steps=20),
        ),
        rollouter=_alphabet_sort_rollouter_config(),
        renderer=from_renderers(Qwen3RendererConfig(enable_thinking=False)),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=Trainer.Config(
            model=trainer_model,
            optim=Optim.Config(
                optimizer=OptimizersContainer.Config(
                    optimizers=[AdamW.Config(pattern=r".*", lr=2e-6)]
                ),
                lr_scheduler=LRSchedulersContainer.Config(
                    warmup_steps=2,
                    decay_type="linear",
                ),
            ),
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=2 * seq_len,
                max_context_length=seq_len,
                dtype="bfloat16",
            ),
            parallelism=trainer_parallelism,
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=GRPOLoss.Config(
                    global_vocab_size=decoder_vocab_size(trainer_model)
                ),
            ),
        ),
        generator=Generator.Config(
            model=generator_model,
            model_dtype="bfloat16",
            parallelism=generator_parallelism,
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=0.8,
                max_tokens=100,
            ),
        ),
    )


def rl_grpo_qwen3_0_6b_varlen_mxfp8() -> Controller.Config:
    """Qwen3-0.6B GRPO with FSDP-managed MXFP8 inference weights."""
    config = rl_grpo_qwen3_0_6b_varlen()
    # TODO: Allow LMHeadCastConverter and QuantizationConverter to
    # co-exist, since they target different layers
    converter = MXFP8LinearConverter.Config(fqns=["layers."]).build()
    config.trainer.model = converter.convert(config.trainer.model)
    config.generator.model = converter.convert(config.generator.model)
    return config


def rl_grpo_qwen3_0_6b_flex_batch_invariant() -> Controller.Config:
    """GRPO training config for Qwen3-0.6B with flex attention and batch invariance
    for bitwise-identical numerics between trainer and generator (4 GPUs: 2 gen + 2 train).

    Trainer keeps fp32 master weights; FSDP mixed precision
    (mixed_precision_param="bfloat16", the default) casts them to bf16 for the
    forward (even at data_parallel_shard_degree=1), matching the bf16 generator.
    """
    config = rl_grpo_qwen3_0_6b_flex()
    # Batch invariance requires strict on-policy: the generator must run the
    # latest weights before generating so trainer/generator logprobs stay
    # bitwise-identical (bit_wise/logprob_diff == 0) every step, not just step 1.
    config.async_loop.target_offpolicy_steps = 0
    config.trainer = dataclasses.replace(
        config.trainer,
        debug=_BATCH_INVARIANT_DEBUG,
        # fp32 master weights; FSDP mixed precision casts to bf16 for the forward.
        training=dataclasses.replace(config.trainer.training, dtype="float32"),
        parallelism=dataclasses.replace(
            config.trainer.parallelism, enable_sequence_parallel=False
        ),
    )
    config.generator = dataclasses.replace(
        config.generator,
        debug=_BATCH_INVARIANT_DEBUG,
        reset_kv_cache_on_model_state_dict_load=True,
    )
    config.trainer.model, config.generator.model = _qwen3_rl_model_configs(
        "0.6B",
        seq_len=config.trainer.training.max_context_length,
        attn_backend="flex",
        trainer_parallelism=config.trainer.parallelism,
        generator_parallelism=config.generator.parallelism,
        converters=[BatchInvariantFlexConverter.Config()],
    )
    # Local compile regions do not support batch-invariant mode.
    for model_config in (config.trainer.model, config.generator.model):
        model_config.local_compile.regions = []
    block_size = config.trainer.model.layers[0].attention.inner_attention.block_size
    config.async_loop.batcher = dataclasses.replace(
        config.async_loop.batcher, per_sample_pad_multiple=block_size
    )
    return config


def rl_grpo_gpt_oss_20b_varlen() -> Controller.Config:
    """GRPO training config for GPT-OSS-20B with varlen attention.

    GPT-OSS uses alternating attention: even layers apply a sliding window, odd
    layers use full causal attention. Metadata carries the per-layer window.
    """
    num_samples_per_prompt = 8
    seq_len = 2048
    trainer_parallelism = ParallelismConfig(
        data_parallel_shard_degree=1,
        tensor_parallel_degree=2,
        expert_parallel_degree=2,
    )
    generator_parallelism = InferenceParallelismConfig(
        data_parallel_degree=1,
        tensor_parallel_degree=4,
        expert_parallel_degree=4,
    )
    trainer_model, generator_model = build_rl_model_configs(
        build_gpt_oss_model_config,
        "20b",
        seq_len=seq_len,
        attn_backend="varlen",
        trainer_parallelism=trainer_parallelism,
        generator_parallelism=generator_parallelism,
    )
    return Controller.Config(
        hf_assets_path="./assets/hf/gpt-oss-20b",
        async_loop=AsyncLoopConfig(
            num_training_steps=10,
            num_prompts_per_train_step=5,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(steps=20),
        ),
        rollouter=_alphabet_sort_rollouter_config(),
        renderer=from_renderers(GptOssRendererConfig(reasoning_effort="low")),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=Trainer.Config(
            model=trainer_model,
            optim=Optim.Config(
                optimizer=OptimizersContainer.Config(
                    optimizers=[AdamW.Config(pattern=r".*", lr=2e-6)]
                ),
                lr_scheduler=LRSchedulersContainer.Config(
                    warmup_steps=2,
                    decay_type="linear",
                ),
            ),
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=2 * seq_len,
                max_context_length=seq_len,
            ),
            parallelism=trainer_parallelism,
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=GRPOLoss.Config(
                    global_vocab_size=decoder_vocab_size(trainer_model)
                ),
            ),
        ),
        generator=Generator.Config(
            model=generator_model,
            model_dtype="bfloat16",
            parallelism=generator_parallelism,
            # TODO: use FULL CUDA graphs once a CUDA-graph-compatible
            # distributed MoE token dispatcher is enabled for GPT-OSS.
            engine=VLLMEngine.Config(cuda_graph=VLLMCudaGraphConfig(mode="NONE")),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=0.8,
                max_tokens=700,
            ),
        ),
    )


def rl_grpo_qwen3_1_7b() -> Controller.Config:
    """GRPO training config for Qwen3-1.7B (6 GPUs: 4 gen + 2 train)."""
    num_samples_per_prompt = 8
    seq_len = 2048
    trainer_parallelism = ParallelismConfig(
        data_parallel_shard_degree=1,
        tensor_parallel_degree=2,
    )
    generator_parallelism = InferenceParallelismConfig(
        data_parallel_degree=1,
        tensor_parallel_degree=4,
    )
    trainer_model, generator_model = _qwen3_rl_model_configs(
        "1.7B",
        seq_len=seq_len,
        attn_backend="varlen",
        trainer_parallelism=trainer_parallelism,
        generator_parallelism=generator_parallelism,
    )
    return Controller.Config(
        hf_assets_path="./assets/hf/Qwen3-1.7B",
        async_loop=AsyncLoopConfig(
            num_training_steps=10,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(steps=20),
        ),
        rollouter=_alphabet_sort_rollouter_config(),
        renderer=from_renderers(Qwen3RendererConfig(enable_thinking=False)),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=Trainer.Config(
            model=trainer_model,
            optim=Optim.Config(
                optimizer=OptimizersContainer.Config(
                    optimizers=[AdamW.Config(pattern=r".*", lr=2e-6)]
                ),
                lr_scheduler=LRSchedulersContainer.Config(
                    warmup_steps=2,
                    decay_type="linear",
                ),
            ),
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=2 * seq_len,
                max_context_length=seq_len,
            ),
            parallelism=trainer_parallelism,
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=GRPOLoss.Config(
                    global_vocab_size=decoder_vocab_size(trainer_model)
                ),
            ),
        ),
        generator=Generator.Config(
            model=generator_model,
            model_dtype="bfloat16",
            parallelism=generator_parallelism,
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=0.8,
                max_tokens=700,
            ),
        ),
    )


def rl_grpo_qwen3_14b() -> Controller.Config:
    """GRPO training config for Qwen3-14B (16 GPUs: 8 gen + 8 train)."""
    num_samples_per_prompt = 8
    seq_len = 2048
    trainer_parallelism = ParallelismConfig(
        data_parallel_shard_degree=1,
        tensor_parallel_degree=8,
    )
    generator_parallelism = InferenceParallelismConfig(
        data_parallel_degree=1,
        tensor_parallel_degree=8,
    )
    trainer_model, generator_model = _qwen3_rl_model_configs(
        "14B",
        seq_len=seq_len,
        attn_backend="varlen",
        trainer_parallelism=trainer_parallelism,
        generator_parallelism=generator_parallelism,
    )
    return Controller.Config(
        hf_assets_path="./assets/hf/Qwen3-14B",
        async_loop=AsyncLoopConfig(
            num_training_steps=10,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(steps=20),
        ),
        rollouter=_alphabet_sort_rollouter_config(),
        renderer=from_renderers(Qwen3RendererConfig(enable_thinking=False)),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=Trainer.Config(
            model=trainer_model,
            optim=Optim.Config(
                optimizer=OptimizersContainer.Config(
                    optimizers=[AdamW.Config(pattern=r".*", lr=1e-6)]
                ),
                lr_scheduler=LRSchedulersContainer.Config(
                    warmup_steps=2,
                    decay_type="linear",
                ),
            ),
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=2 * seq_len,
                max_context_length=seq_len,
                dtype="bfloat16",
            ),
            parallelism=trainer_parallelism,
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=GRPOLoss.Config(
                    global_vocab_size=decoder_vocab_size(trainer_model)
                ),
            ),
        ),
        generator=Generator.Config(
            model=generator_model,
            model_dtype="bfloat16",
            parallelism=generator_parallelism,
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=0.8,
                max_tokens=700,
            ),
        ),
    )


def rl_grpo_qwen3_30b_a3b_varlen() -> Controller.Config:
    """GRPO training config for Qwen3-30B-A3B MoE (8 GPUs: 4 gen + 4 train).

    Trainer and generator uses TP=2 for dense layers and EP=4 for MoE experts.

    Note: Qwen3-30B-A3B has 4 KV heads, so TP degree cannot exceed 4.
    """
    num_samples_per_prompt = 8
    seq_len = 2048
    trainer_parallelism = ParallelismConfig(
        data_parallel_shard_degree=2,
        data_parallel_replicate_degree=1,
        tensor_parallel_degree=2,
        expert_parallel_degree=4,
    )
    generator_parallelism = InferenceParallelismConfig(
        data_parallel_degree=2,
        tensor_parallel_degree=2,
        expert_parallel_degree=4,
    )
    trainer_model, generator_model = build_rl_model_configs(
        build_model_config,
        "30B-A3B",
        seq_len=seq_len,
        attn_backend="varlen",
        trainer_parallelism=trainer_parallelism,
        generator_parallelism=generator_parallelism,
    )
    return Controller.Config(
        hf_assets_path="./assets/hf/Qwen3-30B-A3B",
        async_loop=AsyncLoopConfig(
            num_training_steps=10,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(steps=20),
        ),
        rollouter=_alphabet_sort_rollouter_config(),
        renderer=from_renderers(Qwen3RendererConfig(enable_thinking=False)),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=Trainer.Config(
            model=trainer_model,
            optim=Optim.Config(
                optimizer=OptimizersContainer.Config(
                    optimizers=[AdamW.Config(pattern=r".*", lr=1e-6)]
                ),
                lr_scheduler=LRSchedulersContainer.Config(
                    warmup_steps=2,
                    decay_type="linear",
                ),
            ),
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=2 * seq_len,
                max_context_length=seq_len,
                dtype="bfloat16",
            ),
            parallelism=trainer_parallelism,
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=GRPOLoss.Config(
                    global_vocab_size=decoder_vocab_size(trainer_model)
                ),
            ),
        ),
        generator=Generator.Config(
            model=generator_model,
            model_dtype="bfloat16",
            engine=VLLMEngine.Config(cuda_graph=VLLMCudaGraphConfig(mode="NONE")),
            parallelism=generator_parallelism,
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=0.8,
                max_tokens=700,
            ),
        ),
    )


def rl_grpo_qwen3_30b_a3b_varlen_perf() -> Controller.Config:
    """Qwen3-30B-A3B GRPO with throughput overrides (8 GPUs: 4 gen + 4 train).

    Same model/parallelism/data as ``rl_grpo_qwen3_30b_a3b_varlen``, but applies
    opt-in overrides to both the trainer and generator models:

    * ``fused_swiglu`` fuses the dense SwiGLU activation; the sibling grouped
      experts override also fuses its gate/up projections
      into a single weight (one GEMM; fused SiLU-and-mul Triton kernel).

    The override is CUDA-only. Checkpoints stay interchangeable with the
    non-fused 30B config.
    """
    config = rl_grpo_qwen3_30b_a3b_varlen()
    config.override = OverrideConfig(
        imports=["torchtitan_recipes.overrides.fused_swiglu.fused_swiglu"]
    )
    return config


def rl_grpo_qwen3_0_6b_varlen_batch_invariant(
    *, seq_len: int = 2048
) -> Controller.Config:
    """On-policy GRPO config for Qwen3-0.6B (8 GPUs: trainer TP=2 + 3 generators TP=2).

    Enables deterministic + batch-invariant mode for true on-policy RL training.

    Trainer keeps fp32 master weights; FSDP mixed precision
    (mixed_precision_param="bfloat16", the default) casts them to bf16 for the
    forward (the cast happens even at data_parallel_shard_degree=1, where FSDP
    wraps the model purely as a mixed-precision boundary), so the trainer
    forward is bitwise identical to the bf16 generator.
    """
    batch_invariant_config = DebugConfig(batch_invariant=True, deterministic=True)
    num_samples_per_prompt = 8
    trainer_parallelism = ParallelismConfig(
        data_parallel_shard_degree=1,
        tensor_parallel_degree=2,
        enable_sequence_parallel=False,
    )
    generator_parallelism = InferenceParallelismConfig(
        data_parallel_degree=1,
        tensor_parallel_degree=2,
    )
    trainer_model, generator_model = _qwen3_rl_model_configs(
        "0.6B",
        seq_len=seq_len,
        attn_backend="varlen",
        trainer_parallelism=trainer_parallelism,
        generator_parallelism=generator_parallelism,
    )
    # Local compile regions do not support batch-invariant mode.
    for model_config in (trainer_model, generator_model):
        model_config.local_compile.regions = []
    return Controller.Config(
        hf_assets_path="./assets/hf/Qwen3-0.6B",
        num_generators=3,
        async_loop=AsyncLoopConfig(
            num_training_steps=10,
            # Batch invariance: strict on-policy so trainer/generator logprobs
            # stay bitwise-identical every step.
            target_offpolicy_steps=0,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(steps=20),
        ),
        rollouter=_alphabet_sort_rollouter_config(),
        renderer=from_renderers(Qwen3RendererConfig(enable_thinking=False)),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=Trainer.Config(
            model=trainer_model,
            optim=Optim.Config(
                optimizer=OptimizersContainer.Config(
                    optimizers=[AdamW.Config(pattern=r".*", lr=2e-6)]
                ),
                lr_scheduler=LRSchedulersContainer.Config(
                    warmup_steps=2,
                    decay_type="linear",
                ),
            ),
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=2 * seq_len,
                max_context_length=seq_len,
            ),
            parallelism=trainer_parallelism,
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            debug=batch_invariant_config,
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=GRPOLoss.Config(
                    global_vocab_size=decoder_vocab_size(trainer_model)
                ),
            ),
        ),
        generator=Generator.Config(
            model=generator_model,
            model_dtype="bfloat16",
            parallelism=generator_parallelism,
            checkpointer=None,
            sampling=SamplingConfig(
                # T=1 so the RL loss golden doesn't depend on the T != 1 TODO in SamplingConfig.
                temperature=1.0,
                max_tokens=700,
            ),
            debug=batch_invariant_config,
            reset_kv_cache_on_model_state_dict_load=True,
        ),
    )


def _qwen3_5_rl_model_configs(
    flavor: str,
    *,
    seq_len: int,
    attn_backend: str = "varlen",
    trainer_parallelism: ParallelismConfig,
    generator_parallelism: InferenceParallelismConfig,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> tuple[Decoder.Config, Decoder.Config]:
    """Qwen3.5 ``(trainer, generator)`` model configs for RL, with fp32 lm_head logits."""
    return build_rl_model_configs(
        build_qwen3_5_model_config,
        flavor,
        trainer_parallelism=trainer_parallelism,
        generator_parallelism=generator_parallelism,
        seq_len=seq_len,
        attn_backend=attn_backend,
        converters=converters,
    )


def rl_grpo_qwen3_5_9b_varlen() -> Controller.Config:
    """Qwen3.5-9B GRPO with trainer and generator TP=2 (6 GPUs)."""
    num_samples_per_prompt = 8
    seq_len = 2048
    trainer_parallelism = ParallelismConfig(
        data_parallel_shard_degree=2,
        tensor_parallel_degree=2,
        enable_sequence_parallel=True,
    )
    generator_parallelism = InferenceParallelismConfig(
        data_parallel_degree=1,
        tensor_parallel_degree=2,
    )
    trainer_model, generator_model = _qwen3_5_rl_model_configs(
        "9B",
        seq_len=seq_len,
        attn_backend="varlen",
        trainer_parallelism=trainer_parallelism,
        generator_parallelism=generator_parallelism,
    )
    return Controller.Config(
        hf_assets_path="./assets/hf/Qwen3.5-9B",
        async_loop=AsyncLoopConfig(
            num_training_steps=10,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(steps=20),
        ),
        rollouter=_alphabet_sort_rollouter_config(),
        renderer=from_renderers(Qwen3RendererConfig(enable_thinking=False)),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=Trainer.Config(
            model=trainer_model,
            optim=Optim.Config(
                optimizer=OptimizersContainer.Config(
                    optimizers=[AdamW.Config(pattern=r".*", lr=1e-6)]
                ),
                lr_scheduler=LRSchedulersContainer.Config(
                    warmup_steps=0,
                    min_lr_factor=1.0,
                ),
            ),
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=seq_len,
                max_context_length=seq_len,
                dtype="bfloat16",
            ),
            parallelism=trainer_parallelism,
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=GRPOLoss.Config(
                    global_vocab_size=decoder_vocab_size(trainer_model)
                ),
            ),
        ),
        generator=Generator.Config(
            model=generator_model,
            model_dtype="bfloat16",
            engine=VLLMEngine.Config(cuda_graph=VLLMCudaGraphConfig(mode="FULL")),
            parallelism=generator_parallelism,
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=0.8,
                max_tokens=700,
            ),
        ),
    )


def rl_grpo_qwen3_5_9b_varlen_batch_invariant() -> Controller.Config:
    """On-policy, batch-invariant Qwen3.5-9B GRPO with matching TP=2."""
    config = rl_grpo_qwen3_5_9b_varlen()
    config.async_loop = dataclasses.replace(config.async_loop, target_offpolicy_steps=0)
    config.trainer = dataclasses.replace(
        config.trainer,
        debug=_BATCH_INVARIANT_DEBUG,
        # Matching TP and disabling SP keep trainer/generator reduction order equal.
        parallelism=dataclasses.replace(
            config.trainer.parallelism,
            data_parallel_shard_degree=1,
            enable_sequence_parallel=False,
        ),
    )
    config.generator = dataclasses.replace(
        config.generator,
        debug=_BATCH_INVARIANT_DEBUG,
        reset_kv_cache_on_model_state_dict_load=True,
    )
    config.trainer.model, config.generator.model = _qwen3_5_rl_model_configs(
        "9B",
        seq_len=config.trainer.training.max_context_length,
        trainer_parallelism=config.trainer.parallelism,
        generator_parallelism=config.generator.parallelism,
    )
    # Local compile regions do not support batch-invariant mode.
    for model_config in (config.trainer.model, config.generator.model):
        model_config.local_compile.regions = []
    return config


def rl_grpo_qwen3_6_27b_varlen_perf() -> Controller.Config:
    """Qwen3.6-27B GRPO performance config.

    Qwen3.6-27B uses the Qwen3.5-compatible dense Gated DeltaNet model flavor.
    The 8-GPU layout assigns TP2 x FSDP2 to training and TP4 to generation.
    """
    seq_len = 65536
    config = rl_grpo_qwen3_5_9b_varlen()
    config.hf_assets_path = "./assets/hf/Qwen3.6-27B"
    trainer_parallelism = dataclasses.replace(
        config.trainer.parallelism,
        data_parallel_shard_degree=2,
        tensor_parallel_degree=2,
    )
    generator_parallelism = dataclasses.replace(
        config.generator.parallelism,
        data_parallel_degree=1,
        tensor_parallel_degree=4,
    )
    trainer_model, generator_model = _qwen3_5_rl_model_configs(
        "27B",
        seq_len=seq_len,
        trainer_parallelism=trainer_parallelism,
        generator_parallelism=generator_parallelism,
    )
    loss_config = config.trainer.loss
    assert isinstance(loss_config, ChunkedLossWrapper.Config)
    assert isinstance(loss_config.loss_fn, GRPOLoss.Config)
    config.trainer = dataclasses.replace(
        config.trainer,
        model=trainer_model,
        loss=dataclasses.replace(
            loss_config,
            loss_fn=dataclasses.replace(
                loss_config.loss_fn,
                global_vocab_size=decoder_vocab_size(trainer_model),
            ),
        ),
        optim=dataclasses.replace(
            config.trainer.optim,
            optimizer=dataclasses.replace(
                config.trainer.optim.optimizer,
                optimizers=[
                    dataclasses.replace(
                        optimizer,
                        moment_dtype="bfloat16",
                    )
                    for optimizer in config.trainer.optim.optimizer.optimizers
                ],
            ),
        ),
        activation_remat=FullActivationRematConfig(),
        parallelism=trainer_parallelism,
    )
    config.generator = dataclasses.replace(
        config.generator, model=generator_model, parallelism=generator_parallelism
    )
    return config
