# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config entry points for the alphabet-sort example.

Each function returns a complete ``Controller.Config``, discoverable by
``ConfigManager`` via
``--module alphabet_sort --config rl_grpo_qwen3_*``.
"""

import dataclasses

from renderers import GptOssRendererConfig, Qwen3RendererConfig

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.components.renderer import from_renderers
from torchtitan.config import (
    CompileConfig,
    DebugConfig,
    OverrideConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.config.transform import (
    BatchInvariantFlexConverter,
    LMHeadCastConverter,
    ModelConfigConverter,
    MXFP8LinearConverter,
)
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.gpt_oss import model_registry as gpt_oss_model_registry
from torchtitan.models.qwen3 import model_registry
from torchtitan.models.qwen3_5 import model_registry as qwen3_5_model_registry
from torchtitan.rl.components.training_sample_builder import TrainingSampleBuilder
from torchtitan.rl.controller import AsyncLoopConfig, Controller, ValidationConfig
from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig
from torchtitan.rl.distributed.routing.inter_generator import InterGeneratorRouter
from torchtitan.rl.distributed.routing.strategies import (
    LeastLoadedRoutingStrategy,
    StickySessionRoutingStrategy,
)
from torchtitan.rl.examples.alphabet_sort.data import AlphabetSortDataset
from torchtitan.rl.examples.alphabet_sort.env import AlphabetSortEnv
from torchtitan.rl.examples.alphabet_sort.rubric import RewardAlphabetSort
from torchtitan.rl.generator import SamplingConfig, VLLMCudaGraphConfig, VLLMGenerator
from torchtitan.rl.losses import GRPOLoss
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.rollout.rollouter import Rollouter, RolloutWorker
from torchtitan.rl.rubric import Rubric
from torchtitan.rl.trainer import Trainer

_BATCH_INVARIANT_DEBUG = DebugConfig(batch_invariant=True, deterministic=True)

# TODO: Enable CUDA graphs for RL trainers after eager/graph numerics parity is
# verified.


def _alphabet_sort_rollouter_config() -> Rollouter.Config:
    return Rollouter.Config(
        train_dataset=AlphabetSortDataset.Config(seed=42),
        validation_dataset=AlphabetSortDataset.Config(seed=99),
        worker=RolloutWorker.Config(
            rubric=Rubric.Config(reward_fns=[RewardAlphabetSort.Config(weight=1.0)]),
            message_env=AlphabetSortEnv.Config(),
        ),
    )


def _qwen3_rl_model_registry(
    flavor: str,
    *,
    seq_len: int,
    attn_backend: str,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> Decoder.Config:
    """``qwen3.model_registry`` for RL, with the lm_head fp32 cast always on.

    RL logprob / KL math needs the lm_head logits in fp32, so every RL config
    runs ``LMHeadCastConverter`` on top of whatever converters it passes.
    """
    converters = list(converters or [])
    converters.append(LMHeadCastConverter.Config())
    spec = model_registry(
        flavor, seq_len=seq_len, attn_backend=attn_backend, converters=converters
    )
    return spec


def rl_grpo_qwen3_0_6b_varlen() -> Controller.Config:
    """GRPO training config for Qwen3-0.6B (6 GPUs: 4 gen + 2 train)."""
    num_samples_per_prompt = 8
    seq_len = 2048
    model_config = _qwen3_rl_model_registry(
        "0.6B", seq_len=seq_len, attn_backend="varlen"
    )
    return Controller.Config(
        model=model_config,
        hf_assets_path="torchtitan/rl/example_checkpoint/Qwen3-0.6B",
        async_loop=AsyncLoopConfig(
            num_training_steps=10,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(num_samples=20),
        ),
        compile=CompileConfig(backend="aot_eager"),
        rollouter=_alphabet_sort_rollouter_config(),
        renderer=from_renderers(Qwen3RendererConfig(enable_thinking=False)),
        generator_router=InterGeneratorRouter.Config(
            strategy=StickySessionRoutingStrategy.Config(
                fallback_strategy=LeastLoadedRoutingStrategy.Config()
            )
        ),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=Trainer.Config(
            optimizer=default_adamw(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=2 * seq_len,
                max_context_length=seq_len,
            ),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
            ),
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=GRPOLoss.Config(
                    global_vocab_size=decoder_vocab_size(model_config)
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=4,
            ),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=0.95,
                max_tokens=700,
            ),
        ),
    )


def rl_grpo_qwen3_0_6b_varlen_no_compile() -> Controller.Config:
    config = rl_grpo_qwen3_0_6b_varlen()
    config.compile = None
    return config


def rl_grpo_qwen3_0_6b_varlen_checkpoint_test() -> Controller.Config:
    config = rl_grpo_qwen3_0_6b_varlen()
    assert config.trainer.checkpointer is not None
    config.trainer.checkpointer.interval = 2
    config.trainer.lr_scheduler.total_steps = 4
    return config


def rl_grpo_qwen3_0_6b_flex() -> Controller.Config:
    """GRPO training config for Qwen3-0.6B with flex attention (4 GPUs: 2 gen + 2 train)."""
    num_samples_per_prompt = 8
    seq_len = 2048
    model_config = _qwen3_rl_model_registry(
        "0.6B", seq_len=seq_len, attn_backend="flex"
    )
    return Controller.Config(
        model=model_config,
        hf_assets_path="torchtitan/rl/example_checkpoint/Qwen3-0.6B",
        async_loop=AsyncLoopConfig(
            num_training_steps=10,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(num_samples=20),
        ),
        compile=CompileConfig(backend="aot_eager"),
        rollouter=_alphabet_sort_rollouter_config(),
        renderer=from_renderers(Qwen3RendererConfig(enable_thinking=False)),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=Trainer.Config(
            optimizer=default_adamw(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=2 * seq_len,
                max_context_length=seq_len,
                dtype="bfloat16",
            ),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
            ),
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=GRPOLoss.Config(
                    global_vocab_size=decoder_vocab_size(model_config)
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=2,
            ),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=0.95,
                max_tokens=100,
            ),
        ),
    )


def rl_grpo_qwen3_0_6b_varlen_mxfp8() -> Controller.Config:
    """Qwen3-0.6B GRPO with FSDP-managed MXFP8 inference weights."""
    config = rl_grpo_qwen3_0_6b_varlen()
    config.compile = None
    # TODO: Allow LMHeadCastConverter and QuantizationConverter to
    # co-exist, since they target different layers
    config.model = (
        MXFP8LinearConverter.Config(fqns=["layers."]).build().convert(config.model)
    )
    return config


def rl_grpo_qwen3_0_6b_flex_batch_invariant() -> Controller.Config:
    """GRPO training config for Qwen3-0.6B with flex attention and batch invariance
    for bitwise-identical numerics between trainer and generator (4 GPUs: 2 gen + 2 train).

    Trainer keeps fp32 master weights; FSDP mixed precision
    (mixed_precision_param="bfloat16", the default) casts them to bf16 for the
    forward (even at data_parallel_shard_degree=1), matching the bf16 generator.
    """
    config = rl_grpo_qwen3_0_6b_flex()
    config.model = _qwen3_rl_model_registry(
        "0.6B",
        seq_len=config.trainer.training.max_context_length,
        attn_backend="flex",
        converters=[BatchInvariantFlexConverter.Config()],
    )
    block_size = config.model.layers[0].attention.inner_attention.block_size
    config.async_loop.batcher = dataclasses.replace(
        config.async_loop.batcher, per_sample_pad_multiple=block_size
    )
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
        config.generator, debug=_BATCH_INVARIANT_DEBUG
    )
    return config


def rl_grpo_gpt_oss_20b_varlen() -> Controller.Config:
    """GRPO training config for GPT-OSS-20B with varlen attention.

    GPT-OSS uses alternating attention: even layers apply a sliding window, odd
    layers use full causal attention; the per-layer window is baked into each
    ``VarlenInnerAttention.window_size``.
    """
    num_samples_per_prompt = 8
    seq_len = 2048
    model_config = gpt_oss_model_registry("20b", seq_len=seq_len, attn_backend="varlen")
    return Controller.Config(
        model=model_config,
        hf_assets_path="torchtitan/rl/example_checkpoint/gpt-oss-20b",
        async_loop=AsyncLoopConfig(
            num_training_steps=10,
            num_prompts_per_train_step=5,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(num_samples=20),
        ),
        compile=CompileConfig(backend="aot_eager"),
        rollouter=_alphabet_sort_rollouter_config(),
        renderer=from_renderers(GptOssRendererConfig(reasoning_effort="low")),
        generator_router=InterGeneratorRouter.Config(
            strategy=StickySessionRoutingStrategy.Config(
                fallback_strategy=LeastLoadedRoutingStrategy.Config()
            )
        ),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=Trainer.Config(
            optimizer=default_adamw(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=2 * seq_len,
                max_context_length=seq_len,
            ),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
            ),
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=GRPOLoss.Config(
                    global_vocab_size=decoder_vocab_size(model_config)
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=4,
            ),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=0.95,
                max_tokens=700,
            ),
        ),
    )


def rl_grpo_gpt_oss_debug_varlen() -> Controller.Config:
    """Small GPT-OSS debug config (random init) to exercise the full RL loop."""
    num_samples_per_prompt = 8
    seq_len = 2048
    model_config = gpt_oss_model_registry(
        "debugmodel", seq_len=seq_len, attn_backend="varlen"
    )
    return Controller.Config(
        model=model_config,
        hf_assets_path="tests/assets/tokenizer",
        async_loop=AsyncLoopConfig(
            num_training_steps=3,
            num_prompts_per_train_step=5,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(num_samples=20),
            training_sample_builder=TrainingSampleBuilder.Config(
                drop_zero_std_reward_groups=False,
            ),
        ),
        compile=CompileConfig(backend="aot_eager"),
        rollouter=_alphabet_sort_rollouter_config(),
        # Debug tokenizer (vocab 2048, matches debugmodel); the gpt_oss renderer
        # needs gpt-oss special tokens absent here, so use the qwen3 renderer
        # like the other debug configs.
        renderer=from_renderers(Qwen3RendererConfig(enable_thinking=False)),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=Trainer.Config(
            optimizer=default_adamw(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=2 * seq_len,
                max_context_length=seq_len,
            ),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
            ),
            checkpointer=None,
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=GRPOLoss.Config(
                    global_vocab_size=decoder_vocab_size(model_config)
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=4,
            ),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=0.95,
                max_tokens=50,
            ),
        ),
    )


def rl_grpo_gpt_oss_debug_varlen_no_compile() -> Controller.Config:
    config = rl_grpo_gpt_oss_debug_varlen()
    config.compile = None
    return config


def rl_grpo_gpt_oss_debug_varlen_batch_invariant() -> Controller.Config:
    """Small GPT-OSS debug config in deterministic + batch-invariant mode.

    Trainer keeps fp32 master weights; FSDP mixed precision
    (mixed_precision_param="bfloat16", the default) casts them to bf16 for the
    forward (even at data_parallel_shard_degree=1), matching the bf16 generator.
    """
    batch_invariant_config = DebugConfig(batch_invariant=True, deterministic=True)
    num_samples_per_prompt = 8
    seq_len = 2048
    model_config = gpt_oss_model_registry(
        "debugmodel", seq_len=seq_len, attn_backend="varlen"
    )
    return Controller.Config(
        model=model_config,
        hf_assets_path="tests/assets/tokenizer",
        async_loop=AsyncLoopConfig(
            num_training_steps=3,
            # Batch invariance: strict on-policy so trainer/generator logprobs
            # stay bitwise-identical every step.
            target_offpolicy_steps=0,
            num_prompts_per_train_step=5,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(num_samples=20),
            training_sample_builder=TrainingSampleBuilder.Config(
                drop_zero_std_reward_groups=False,
            ),
        ),
        compile=CompileConfig(backend="aot_eager"),
        rollouter=_alphabet_sort_rollouter_config(),
        # Debug tokenizer (vocab 2048, matches debugmodel); the gpt_oss renderer
        # needs gpt-oss special tokens absent here, so use the qwen3 renderer
        # like the other debug configs.
        renderer=from_renderers(Qwen3RendererConfig(enable_thinking=False)),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=Trainer.Config(
            optimizer=default_adamw(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            # fp32 master weights; FSDP mixed precision casts to bf16 for the
            # forward (mixed_precision_param="bfloat16" is the default).
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=2 * seq_len,
                max_context_length=seq_len,
            ),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
                enable_sequence_parallel=False,
            ),
            checkpointer=None,
            debug=batch_invariant_config,
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=GRPOLoss.Config(
                    global_vocab_size=decoder_vocab_size(model_config)
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                # Must match the trainer's TP for bitwise parity: a different TP
                # degree changes reduction order / sharding in the parallel
                # matmuls and attention, which batch-invariant ops do not undo.
                tensor_parallel_degree=2,
            ),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=0.95,
                max_tokens=50,
            ),
            debug=batch_invariant_config,
        ),
    )


def rl_grpo_qwen3_1_7b() -> Controller.Config:
    """GRPO training config for Qwen3-1.7B (6 GPUs: 4 gen + 2 train)."""
    num_samples_per_prompt = 8
    seq_len = 2048
    model_config = _qwen3_rl_model_registry(
        "1.7B", seq_len=seq_len, attn_backend="varlen"
    )
    return Controller.Config(
        model=model_config,
        hf_assets_path="torchtitan/rl/example_checkpoint/Qwen3-1.7B",
        async_loop=AsyncLoopConfig(
            num_training_steps=10,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(num_samples=20),
        ),
        compile=CompileConfig(backend="aot_eager"),
        rollouter=_alphabet_sort_rollouter_config(),
        renderer=from_renderers(Qwen3RendererConfig(enable_thinking=False)),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=Trainer.Config(
            optimizer=default_adamw(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=2 * seq_len,
                max_context_length=seq_len,
            ),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
            ),
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=GRPOLoss.Config(
                    global_vocab_size=decoder_vocab_size(model_config)
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=4,
            ),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=0.95,
                max_tokens=700,
            ),
        ),
    )


def rl_grpo_qwen3_14b() -> Controller.Config:
    """GRPO training config for Qwen3-14B (16 GPUs: 8 gen + 8 train)."""
    num_samples_per_prompt = 8
    seq_len = 2048
    model_config = _qwen3_rl_model_registry(
        "14B", seq_len=seq_len, attn_backend="varlen"
    )
    return Controller.Config(
        model=model_config,
        hf_assets_path="torchtitan/rl/example_checkpoint/Qwen3-14B",
        async_loop=AsyncLoopConfig(
            num_training_steps=10,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(num_samples=20),
        ),
        compile=CompileConfig(backend="aot_eager"),
        rollouter=_alphabet_sort_rollouter_config(),
        renderer=from_renderers(Qwen3RendererConfig(enable_thinking=False)),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=Trainer.Config(
            optimizer=default_adamw(lr=1e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=2 * seq_len,
                max_context_length=seq_len,
                dtype="bfloat16",
            ),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=8,
            ),
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=GRPOLoss.Config(
                    global_vocab_size=decoder_vocab_size(model_config)
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=8,
            ),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=0.95,
                max_tokens=700,
            ),
        ),
    )


def rl_grpo_qwen3_moe_debug_varlen() -> Controller.Config:
    """Debug MoE config with EP+TP on generator (8 GPUs: 4 gen + 4 train).

    Trainer uses data_parallel_shard_degree=2 as FSDP degree and TP=2.
    Generator uses data_parallel_degree=2 (vLLM pure DP), with TP=2.
    MoE layers use EP=4.
    """
    num_samples_per_prompt = 8
    seq_len = 2048
    model_config = model_registry(
        "debugmodel_moe", seq_len=seq_len, attn_backend="varlen"
    )
    return Controller.Config(
        model=model_config,
        hf_assets_path="tests/assets/tokenizer",
        async_loop=AsyncLoopConfig(
            num_training_steps=5,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(num_samples=20),
            training_sample_builder=TrainingSampleBuilder.Config(
                drop_zero_std_reward_groups=False,
            ),
        ),
        # MoE EP all-to-all path issues unpinned D2H copies that block
        # torch.compile and CUDA graph capture; disable both.
        compile=None,
        rollouter=_alphabet_sort_rollouter_config(),
        renderer=from_renderers(Qwen3RendererConfig(enable_thinking=False)),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=Trainer.Config(
            optimizer=default_adamw(lr=8e-4),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=2 * seq_len,
                max_context_length=seq_len,
            ),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=2,
                tensor_parallel_degree=2,
                data_parallel_replicate_degree=1,
                expert_parallel_degree=4,
            ),
            checkpointer=None,
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=GRPOLoss.Config(
                    global_vocab_size=decoder_vocab_size(model_config)
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            # Disable torch.compile + CUDA graph capture: the EP all-to-all
            # path issues an unpinned D2H copy of split sizes that the
            # piecewise/full graph capture rejects.
            cuda_graph=VLLMCudaGraphConfig(mode="NONE"),
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=2,
                tensor_parallel_degree=2,
                expert_parallel_degree=4,
            ),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=0.95,
                max_tokens=50,
            ),
        ),
    )


def rl_grpo_qwen3_moe_debug_deepep() -> Controller.Config:
    """Debug MoE config on the DeepEP v2 backend with a CUDA-graph-capturable generator
    (8 GPUs: 4 gen + 4 train).

    Same EP/TP/DP layout as ``rl_grpo_qwen3_moe_debug_varlen`` (trainer FSDP=2/TP=2/EP=4,
    generator DP=2/TP=2/EP=4), but the MoE uses the DeepEP v2 comm backend. Unlike the
    standard all-to-all -- whose unpinned D2H split-size copy blocks CUDA graph capture, so
    that config disables it -- DeepEP v2's inference dispatch is a static, host-sync-free
    EXPAND layout, so this generator enables CUDA graph capture.

    Per-role config from one shared model config: the trainer uses it as-is (compact,
    host-synced, backward-able DeepEP path), while the generator applies per-actor
    overrides (``generator.override``) to its own copy (``fused_swiglu`` +
    ``deepep_override`` with ``cuda_graph_compatible=True``) to switch its dispatchers to the
    CUDA-graph-compatible EXPAND layout. The overrides touch only the generator's spec, so the
    trainer and weight sync are unaffected.
    """
    config = rl_grpo_qwen3_moe_debug_varlen()
    config.model = model_registry(
        "debugmodel_moe",
        seq_len=config.trainer.training.max_context_length,
        attn_backend="varlen",
        moe_comm_backend="deepep",
    )
    loss_config = config.trainer.loss
    assert isinstance(loss_config, ChunkedLossWrapper.Config)
    assert isinstance(loss_config.loss_fn, GRPOLoss.Config)
    config.trainer = dataclasses.replace(
        config.trainer,
        loss=dataclasses.replace(
            loss_config,
            loss_fn=dataclasses.replace(
                loss_config.loss_fn,
                global_vocab_size=decoder_vocab_size(config.model),
            ),
        ),
    )
    # Generator-only overrides -> CUDA-graph-compatible DeepEP EXPAND dispatch; trainer keeps compact.
    config.generator.override = OverrideConfig(
        imports=[
            "torchtitan.overrides.fused_swiglu.fused_swiglu",
            (
                "torchtitan.overrides.moe_token_dispatcher.deepep_override",
                {"cuda_graph_compatible": True},
            ),
        ]
    )
    config.generator.cuda_graph = VLLMCudaGraphConfig(mode="FULL")
    # vLLM's per-step token budget. The wrapper derives DeepEP's per-rank buffer capacity
    # from this scheduler limit, CUDA graph capture sizes, CP, and SP.
    config.generator.max_num_batched_tokens = 2048
    return config


def rl_grpo_qwen3_moe_debug_varlen_batch_invariant() -> Controller.Config:
    """Batch-invariant MoE EP config for bitwise parity testing (8 GPUs).

    Trainer uses data_parallel_shard_degree=2 as FSDP degree and TP=2.
    Generator uses data_parallel_degree=2 (vLLM pure DP), with TP=2.
    MoE layers use EP=4.

    Parity: trainer FSDP2 TP2 EP4 matches generator DP2 TP2 EP4 bitwise
    (verified ``bit_wise/logprob_diff/max == 0``). The trainer holds fp32 master
    weights; FSDP mixed precision (``training.mixed_precision_param ==
    "bfloat16"``, the default) all-gathers the full params in bf16 before the
    forward, so the forward is numerically identical to the generator's
    replicated bf16 dense DP.

    """
    num_samples_per_prompt = 8
    seq_len = 2048
    model_config = model_registry(
        "debugmodel_moe",
        seq_len=seq_len,
        attn_backend="varlen",
        moe_comm_backend="standard",
    )
    return Controller.Config(
        model=model_config,
        hf_assets_path="tests/assets/tokenizer",
        async_loop=AsyncLoopConfig(
            num_training_steps=10,
            # Batch invariance: strict on-policy so trainer/generator logprobs
            # stay bitwise-identical every step.
            target_offpolicy_steps=0,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(num_samples=20),
            training_sample_builder=TrainingSampleBuilder.Config(
                drop_zero_std_reward_groups=False,
            ),
        ),
        # MoE EP all-to-all path issues unpinned D2H copies that block
        # torch.compile and CUDA graph capture; disable both.
        compile=None,
        rollouter=_alphabet_sort_rollouter_config(),
        renderer=from_renderers(Qwen3RendererConfig(enable_thinking=False)),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=Trainer.Config(
            optimizer=default_adamw(lr=8e-4),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            # fp32 master weights; FSDP mixed precision casts to bf16 for the
            # forward (mixed_precision_param="bfloat16" is the default).
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=2 * seq_len,
                max_context_length=seq_len,
            ),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=2,
                tensor_parallel_degree=2,
                data_parallel_replicate_degree=1,
                expert_parallel_degree=4,
                enable_sequence_parallel=False,
            ),
            checkpointer=None,
            debug=_BATCH_INVARIANT_DEBUG,
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=GRPOLoss.Config(
                    global_vocab_size=decoder_vocab_size(model_config)
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            cuda_graph=VLLMCudaGraphConfig(mode="NONE"),
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=2,
                tensor_parallel_degree=2,
                expert_parallel_degree=4,
            ),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=0.95,
                max_tokens=50,
            ),
            debug=_BATCH_INVARIANT_DEBUG,
        ),
    )


def rl_grpo_qwen3_30b_a3b_varlen() -> Controller.Config:
    """GRPO training config for Qwen3-30B-A3B MoE (8 GPUs: 4 gen + 4 train).

    Trainer and generator uses TP=2 for dense layers and EP=4 for MoE experts.

    Note: Qwen3-30B-A3B has 4 KV heads, so TP degree cannot exceed 4.
    """
    num_samples_per_prompt = 8
    seq_len = 2048
    model_config = model_registry("30B-A3B", seq_len=seq_len, attn_backend="varlen")
    return Controller.Config(
        model=model_config,
        hf_assets_path="torchtitan/rl/example_checkpoint/Qwen3-30B-A3B",
        async_loop=AsyncLoopConfig(
            num_training_steps=10,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(num_samples=20),
        ),
        compile=None,
        rollouter=_alphabet_sort_rollouter_config(),
        renderer=from_renderers(Qwen3RendererConfig(enable_thinking=False)),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=Trainer.Config(
            optimizer=default_adamw(lr=1e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=2 * seq_len,
                max_context_length=seq_len,
                dtype="bfloat16",
            ),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=2,
                data_parallel_replicate_degree=1,
                tensor_parallel_degree=2,
                expert_parallel_degree=4,
            ),
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=GRPOLoss.Config(
                    global_vocab_size=decoder_vocab_size(model_config)
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            cuda_graph=VLLMCudaGraphConfig(mode="NONE"),
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=2,
                tensor_parallel_degree=2,
                expert_parallel_degree=4,
            ),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=0.95,
                max_tokens=700,
            ),
        ),
    )


def rl_grpo_qwen3_30b_a3b_varlen_perf() -> Controller.Config:
    """Qwen3-30B-A3B GRPO with throughput overrides (8 GPUs: 4 gen + 4 train).

    Same model/parallelism/data as ``rl_grpo_qwen3_30b_a3b_varlen``, but applies
    opt-in overrides (per-actor) to both the trainer and generator:

    * ``fused_swiglu`` fuses the dense SwiGLU activation; the sibling grouped
      experts override also fuses its gate/up projections
      into a single weight (one GEMM; fused SiLU-and-mul Triton kernel).
    * ``helion_rope`` applies cos/sin RoPE with a fused Helion kernel (qwen3 uses
      ``CosSinRoPE``, which the override targets).

    Both are CUDA-only; ``helion_rope`` additionally needs the optional ``helion``
    package. Checkpoints stay interchangeable with the non-fused/stock-RoPE 30B
    config.
    """
    config = rl_grpo_qwen3_30b_a3b_varlen()
    # Applied after each actor's update_from_config and before build; separate
    # OverrideConfig instances keep the trainer and generator overrides
    # independent (they run in different actors).
    perf_imports = [
        "torchtitan.overrides.fused_swiglu.fused_swiglu",
        "torchtitan.overrides.helion_rope.helion_cos_sin_rope",
    ]
    config.trainer = dataclasses.replace(
        config.trainer, override=OverrideConfig(imports=list(perf_imports))
    )
    config.generator = dataclasses.replace(
        config.generator,
        override=OverrideConfig(imports=list(perf_imports)),
    )
    return config


def rl_grpo_qwen3_0_6b_varlen_batch_invariant() -> Controller.Config:
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
    seq_len = 2048
    model_config = _qwen3_rl_model_registry(
        "0.6B", seq_len=seq_len, attn_backend="varlen"
    )
    return Controller.Config(
        model=model_config,
        hf_assets_path="torchtitan/rl/example_checkpoint/Qwen3-0.6B",
        num_generators=3,
        async_loop=AsyncLoopConfig(
            num_training_steps=10,
            # Batch invariance: strict on-policy so trainer/generator logprobs
            # stay bitwise-identical every step.
            target_offpolicy_steps=0,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(num_samples=20),
        ),
        compile=CompileConfig(backend="aot_eager"),
        rollouter=_alphabet_sort_rollouter_config(),
        renderer=from_renderers(Qwen3RendererConfig(enable_thinking=False)),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=Trainer.Config(
            optimizer=default_adamw(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=2 * seq_len,
                max_context_length=seq_len,
            ),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
                enable_sequence_parallel=False,
            ),
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            debug=batch_invariant_config,
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=GRPOLoss.Config(
                    global_vocab_size=decoder_vocab_size(model_config)
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=2,
            ),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=0.95,
                max_tokens=700,
            ),
            debug=batch_invariant_config,
        ),
    )


def _qwen3_5_rl_model_registry(
    flavor: str,
    *,
    seq_len: int,
    attn_backend: str = "varlen",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> Decoder.Config:
    """``qwen3_5.model_registry`` for RL, with the lm_head fp32 cast always on.

    RL logprob / KL math needs the lm_head logits in fp32, so every RL config
    runs ``LMHeadCastConverter`` on top of whatever converters it passes.
    """
    converters = list(converters or [])
    converters.append(LMHeadCastConverter.Config())
    return qwen3_5_model_registry(
        flavor, seq_len=seq_len, attn_backend=attn_backend, converters=converters
    )


def rl_grpo_qwen3_5_9b_varlen() -> Controller.Config:
    """Qwen3.5-9B GRPO with trainer and generator TP=2 (6 GPUs)."""
    num_samples_per_prompt = 8
    seq_len = 2048
    model_config = _qwen3_5_rl_model_registry(
        "9B", seq_len=seq_len, attn_backend="varlen"
    )
    return Controller.Config(
        model=model_config,
        hf_assets_path="torchtitan/rl/example_checkpoint/Qwen3.5-9B",
        async_loop=AsyncLoopConfig(
            num_training_steps=10,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(num_samples=20),
        ),
        compile=None,
        rollouter=_alphabet_sort_rollouter_config(),
        renderer=from_renderers(Qwen3RendererConfig(enable_thinking=False)),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=Trainer.Config(
            optimizer=default_adamw(lr=1e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=0,
                min_lr_factor=1.0,
            ),
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=seq_len,
                max_context_length=seq_len,
                dtype="bfloat16",
            ),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=2,
                tensor_parallel_degree=2,
            ),
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=GRPOLoss.Config(
                    global_vocab_size=decoder_vocab_size(model_config)
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            cuda_graph=VLLMCudaGraphConfig(mode="FULL"),
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=2,
            ),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=0.95,
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
        config.generator, debug=_BATCH_INVARIANT_DEBUG
    )
    return config


def rl_grpo_qwen3_5_debug_varlen() -> Controller.Config:
    """Random-init Qwen3.5 GRPO config for CI."""
    num_samples_per_prompt = 8
    seq_len = 2048
    model_config = _qwen3_5_rl_model_registry(
        "debugmodel", seq_len=seq_len, attn_backend="varlen"
    )
    return Controller.Config(
        model=model_config,
        hf_assets_path="tests/assets/tokenizer",
        async_loop=AsyncLoopConfig(
            num_training_steps=5,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(num_samples=20),
            training_sample_builder=TrainingSampleBuilder.Config(
                drop_zero_std_reward_groups=False,
            ),
        ),
        compile=None,
        rollouter=_alphabet_sort_rollouter_config(),
        renderer=from_renderers(Qwen3RendererConfig(enable_thinking=False)),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=Trainer.Config(
            optimizer=default_adamw(lr=1e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=0,
                min_lr_factor=1.0,
            ),
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=seq_len,
                max_context_length=seq_len,
                dtype="bfloat16",
            ),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=2,
                tensor_parallel_degree=2,
            ),
            checkpointer=None,  # random-init weights
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=GRPOLoss.Config(
                    global_vocab_size=decoder_vocab_size(model_config)
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            cuda_graph=VLLMCudaGraphConfig(mode="FULL"),
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=2,
            ),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=0.95,
                max_tokens=256,
            ),
        ),
    )


def rl_grpo_qwen3_5_debug_varlen_batch_invariant() -> Controller.Config:
    """On-policy, batch-invariant Qwen3.5 GRPO config for CI."""
    config = rl_grpo_qwen3_5_debug_varlen()
    config.async_loop = dataclasses.replace(config.async_loop, target_offpolicy_steps=0)
    config.trainer = dataclasses.replace(
        config.trainer,
        debug=_BATCH_INVARIANT_DEBUG,
        parallelism=dataclasses.replace(
            config.trainer.parallelism,
            enable_sequence_parallel=False,
        ),
    )
    config.generator = dataclasses.replace(
        config.generator, debug=_BATCH_INVARIANT_DEBUG
    )
    return config


def rl_grpo_qwen3_6_27b_varlen_perf() -> Controller.Config:
    """Qwen3.6-27B GRPO with fused OffsetRMSNorm on trainer and generator.

    Qwen3.6-27B uses the Qwen3.5-compatible dense Gated DeltaNet model flavor.
    The 8-GPU layout assigns TP2 x FSDP2 to training and TP4 to generation.
    """
    seq_len = 65536
    config = rl_grpo_qwen3_5_9b_varlen()
    config.model = _qwen3_5_rl_model_registry(
        "27B", seq_len=seq_len, attn_backend="varlen"
    )
    config.hf_assets_path = "torchtitan/rl/example_checkpoint/Qwen3.6-27B"
    perf_imports = ["torchtitan.overrides.offset_rmsnorm.triton_offset_rmsnorm"]
    loss_config = config.trainer.loss
    assert isinstance(loss_config, ChunkedLossWrapper.Config)
    assert isinstance(loss_config.loss_fn, GRPOLoss.Config)
    config.trainer = dataclasses.replace(
        config.trainer,
        loss=dataclasses.replace(
            loss_config,
            loss_fn=dataclasses.replace(
                loss_config.loss_fn,
                global_vocab_size=decoder_vocab_size(config.model),
            ),
        ),
        optimizer=dataclasses.replace(
            config.trainer.optimizer,
            implementation="fused_opt_states_bf16",
        ),
        activation_checkpoint=FullAC.Config(),
        parallelism=dataclasses.replace(
            config.trainer.parallelism,
            data_parallel_shard_degree=2,
            tensor_parallel_degree=2,
        ),
        override=OverrideConfig(imports=list(perf_imports)),
    )
    config.generator = dataclasses.replace(
        config.generator,
        parallelism=dataclasses.replace(
            config.generator.parallelism,
            data_parallel_degree=1,
            tensor_parallel_degree=4,
        ),
        override=OverrideConfig(imports=list(perf_imports)),
    )
    return config


def rl_grpo_qwen3_6_27b_varlen_perf_mxfp8() -> Controller.Config:
    """Qwen3.6-27B performance config with MXFP8 linear layers."""
    config = rl_grpo_qwen3_6_27b_varlen_perf()
    # TODO: Allow LMHeadCastConverter and QuantizationConverter to
    # co-exist, since they target different layers
    config.model = (
        MXFP8LinearConverter.Config(
            # in_proj_a/b have 48-wide outputs, which are not divisible by the
            # MXFP8 block size.
            fqns=[
                "attention.",
                "feed_forward.",
                "delta_net.in_proj_q",
                "delta_net.in_proj_k",
                "delta_net.in_proj_v",
                "delta_net.in_proj_z",
                "delta_net.out_proj",
            ]
        )
        .build()
        .convert(config.model)
    )
    return config
