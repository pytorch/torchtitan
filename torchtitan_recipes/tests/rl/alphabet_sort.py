# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Debug-model and CI recipes for the alphabet-sort example.

These configurations exercise RL code paths in CI with small debug models or
test-only overrides. They are not recommendations for production training;
verified recipes live in ``torchtitan_recipes.rl.alphabet_sort``.
"""

import dataclasses
import os

from renderers import Qwen3RendererConfig

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
    apply_transforms,
    LMHeadFP32OutputConverter,
    ModelConfigConverter,
    ModelConfigTransformContext,
    TokenDispatcherTransform,
)
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.common.token_dispatcher import DeepEPTokenDispatcher
from torchtitan.models.gpt_oss import build_model_config as build_gpt_oss_model_config
from torchtitan.models.kimi_k3 import build_model_config as build_kimi_k3_model_config
from torchtitan.models.qwen3 import build_model_config
from torchtitan.rl.components.batcher import Batcher
from torchtitan.rl.components.training_sample_builder import TrainingSampleBuilder
from torchtitan.rl.controller import AsyncLoopConfig, Controller, ValidationConfig
from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator import SamplingConfig, VLLMCudaGraphConfig, VLLMGenerator
from torchtitan.rl.losses import GRPOLoss
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.trainer import Trainer

from torchtitan_recipes.rl.alphabet_sort import (
    _alphabet_sort_rollouter_config,
    _BATCH_INVARIANT_DEBUG,
    _build_qwen3_5_rl_model_config,
    rl_grpo_qwen3_0_6b_varlen,
    rl_grpo_qwen3_0_6b_varlen_batch_invariant,
    rl_grpo_qwen3_0_6b_varlen_no_compile,
)


def rl_grpo_qwen3_0_6b_varlen_checkpoint_test(
    *, seq_len: int = 2048
) -> Controller.Config:
    config = rl_grpo_qwen3_0_6b_varlen(seq_len=seq_len)
    assert config.trainer.checkpointer is not None
    config.trainer.checkpointer.interval = 2
    config.trainer.optim.lr_scheduler.total_steps = 4
    return config


def rl_grpo_gpt_oss_debug_varlen(*, seq_len: int = 2048) -> Controller.Config:
    """Small GPT-OSS debug config (random init) to exercise the full RL loop."""
    num_samples_per_prompt = 8
    model_config = build_gpt_oss_model_config(
        "debugmodel",
        seq_len=seq_len,
        attn_backend="varlen",
        converters=[LMHeadFP32OutputConverter.Config()],
    )
    return Controller.Config(
        model=model_config,
        hf_assets_path="tests/assets/tokenizer",
        async_loop=AsyncLoopConfig(
            num_training_steps=3,
            num_prompts_per_train_step=5,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(steps=20),
            training_sample_builder=TrainingSampleBuilder.Config(
                drop_zero_std_reward_groups=False,
            ),
        ),
        rollouter=_alphabet_sort_rollouter_config(),
        # Debug tokenizer (vocab 2048, matches debugmodel); the gpt_oss renderer
        # needs gpt-oss special tokens absent here, so use the qwen3 renderer
        # like the other debug configs.
        renderer=from_renderers(Qwen3RendererConfig(enable_thinking=False)),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=Trainer.Config(
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
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
                expert_parallel_degree=2,
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
                max_tokens=50,
            ),
        ),
    )


def rl_grpo_gpt_oss_debug_varlen_no_compile(
    *, seq_len: int = 2048
) -> Controller.Config:
    config = rl_grpo_gpt_oss_debug_varlen(seq_len=seq_len)
    assert config.model is not None
    config.model.local_compile_regions = []
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
    model_config = build_gpt_oss_model_config(
        "debugmodel",
        seq_len=seq_len,
        attn_backend="varlen",
        converters=[LMHeadFP32OutputConverter.Config()],
    )
    # Local compile regions do not support batch-invariant mode.
    model_config.local_compile_regions = []
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
            validation=ValidationConfig(steps=20),
            training_sample_builder=TrainingSampleBuilder.Config(
                drop_zero_std_reward_groups=False,
            ),
        ),
        rollouter=_alphabet_sort_rollouter_config(),
        # Debug tokenizer (vocab 2048, matches debugmodel); the gpt_oss renderer
        # needs gpt-oss special tokens absent here, so use the qwen3 renderer
        # like the other debug configs.
        renderer=from_renderers(Qwen3RendererConfig(enable_thinking=False)),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=Trainer.Config(
            optim=Optim.Config(
                optimizer=OptimizersContainer.Config(
                    optimizers=[AdamW.Config(pattern=r".*", lr=2e-6)]
                ),
                lr_scheduler=LRSchedulersContainer.Config(
                    warmup_steps=2,
                    decay_type="linear",
                ),
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
                expert_parallel_degree=2,
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
                expert_parallel_degree=2,
            ),
            # TODO: use FULL CUDA graphs once a CUDA-graph-compatible
            # distributed MoE token dispatcher is enabled for GPT-OSS.
            cuda_graph=VLLMCudaGraphConfig(mode="NONE"),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=0.8,
                max_tokens=50,
            ),
            debug=batch_invariant_config,
            reset_kv_cache_on_weight_sync=True,
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
    model_config = build_model_config(
        "debugmodel_moe",
        seq_len=seq_len,
        attn_backend="varlen",
        converters=[LMHeadFP32OutputConverter.Config()],
    )
    return Controller.Config(
        model=model_config,
        hf_assets_path="tests/assets/tokenizer",
        async_loop=AsyncLoopConfig(
            num_training_steps=5,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(steps=20),
            training_sample_builder=TrainingSampleBuilder.Config(
                drop_zero_std_reward_groups=False,
            ),
        ),
        rollouter=_alphabet_sort_rollouter_config(),
        renderer=from_renderers(Qwen3RendererConfig(enable_thinking=False)),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=Trainer.Config(
            optim=Optim.Config(
                optimizer=OptimizersContainer.Config(
                    optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)]
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
    config.model = build_model_config(
        "debugmodel_moe",
        seq_len=config.trainer.training.max_context_length,
        attn_backend="varlen",
        converters=[LMHeadFP32OutputConverter.Config()],
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
            "torchtitan_recipes.overrides.fused_swiglu.fused_swiglu",
            (
                "torchtitan_recipes.overrides.moe_token_dispatcher.deepep_override",
                {"cuda_graph_compatible": True},
            ),
        ]
    )
    config.generator.cuda_graph = VLLMCudaGraphConfig(mode="FULL")
    # vLLM's per-step token budget. The wrapper derives DeepEP's per-rank buffer capacity
    # from this scheduler limit, CUDA graph capture sizes, CP, and SP.
    config.generator.max_num_batched_tokens = 2048
    return apply_transforms(
        config,
        [TokenDispatcherTransform(dispatcher=DeepEPTokenDispatcher)],
        context=ModelConfigTransformContext(
            training=config.trainer.training,
            parallelism=config.trainer.parallelism,
        ),
    )


def rl_grpo_qwen3_moe_debug_varlen_batch_invariant(
    *, seq_len: int = 2048
) -> Controller.Config:
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
    model_config = build_model_config(
        "debugmodel_moe",
        seq_len=seq_len,
        attn_backend="varlen",
        converters=[LMHeadFP32OutputConverter.Config()],
    )
    # Local compile regions do not support batch-invariant mode.
    model_config.local_compile_regions = []
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
            validation=ValidationConfig(steps=20),
            training_sample_builder=TrainingSampleBuilder.Config(
                drop_zero_std_reward_groups=False,
            ),
        ),
        rollouter=_alphabet_sort_rollouter_config(),
        renderer=from_renderers(Qwen3RendererConfig(enable_thinking=False)),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=Trainer.Config(
            optim=Optim.Config(
                optimizer=OptimizersContainer.Config(
                    optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)]
                ),
                lr_scheduler=LRSchedulersContainer.Config(
                    warmup_steps=2,
                    decay_type="linear",
                ),
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
                max_tokens=50,
            ),
            debug=_BATCH_INVARIANT_DEBUG,
            reset_kv_cache_on_weight_sync=True,
        ),
    )


def rl_grpo_qwen3_5_debug_varlen(*, seq_len: int = 2048) -> Controller.Config:
    """Random-init Qwen3.5 GRPO config for CI."""
    num_samples_per_prompt = 8
    model_config = _build_qwen3_5_rl_model_config(
        "debugmodel", seq_len=seq_len, attn_backend="varlen"
    )
    return Controller.Config(
        model=model_config,
        hf_assets_path="tests/assets/tokenizer",
        async_loop=AsyncLoopConfig(
            num_training_steps=5,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(steps=20),
            training_sample_builder=TrainingSampleBuilder.Config(
                drop_zero_std_reward_groups=False,
            ),
        ),
        rollouter=_alphabet_sort_rollouter_config(),
        renderer=from_renderers(Qwen3RendererConfig(enable_thinking=False)),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=Trainer.Config(
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
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=2,
                tensor_parallel_degree=2,
                enable_sequence_parallel=True,
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
            # vLLM models FULL-graph capture warmup as one sequence, so keep
            # its synthetic sequence within the GDN model's context length.
            max_num_batched_tokens=seq_len,
            cuda_graph=VLLMCudaGraphConfig(mode="FULL"),
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=2,
            ),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=0.8,
                max_tokens=256,
            ),
        ),
    )


def rl_grpo_qwen3_5_debug_varlen_batch_invariant(
    *, seq_len: int = 2048
) -> Controller.Config:
    """On-policy, batch-invariant Qwen3.5 GRPO config for CI."""
    config = rl_grpo_qwen3_5_debug_varlen(seq_len=seq_len)
    assert config.model is not None
    # Local compile regions do not support batch-invariant mode.
    config.model.local_compile_regions = []
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
        config.generator,
        debug=_BATCH_INVARIANT_DEBUG,
        reset_kv_cache_on_weight_sync=True,
    )
    return config


def _build_kimi_k3_rl_model_config(
    flavor: str,
    *,
    seq_len: int,
    attn_backend: str = "varlen",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> Decoder.Config:
    """``kimi_k3.build_model_config`` for RL, with the lm_head fp32 cast always on."""
    converters = list(converters or [])
    converters.append(LMHeadFP32OutputConverter.Config())
    return build_kimi_k3_model_config(
        flavor,
        seq_len=seq_len,
        attn_backend=attn_backend,
        converters=converters,
    )


def rl_grpo_kimi_k3_debug_varlen(*, seq_len: int = 2048) -> Controller.Config:
    """Random-init Kimi K3 GRPO config for Blackwell integration testing."""
    model_config = _build_kimi_k3_rl_model_config(
        "debugmodel", seq_len=seq_len, attn_backend="varlen"
    )
    model_config.local_compile_regions = []
    return Controller.Config(
        model=model_config,
        hf_assets_path="tests/assets/tokenizer",
        async_loop=AsyncLoopConfig(
            num_training_steps=5,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=8,
            batcher=Batcher.Config(max_num_documents=32),
            validation=ValidationConfig(steps=20),
            training_sample_builder=TrainingSampleBuilder.Config(
                drop_zero_std_reward_groups=False,
            ),
        ),
        rollouter=_alphabet_sort_rollouter_config(),
        renderer=from_renderers(Qwen3RendererConfig(enable_thinking=False)),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=Trainer.Config(
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
            # Kimi K3 currently supports FSDP data parallelism but not TP.
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=2,
                tensor_parallel_degree=1,
                fsdp_defer_gradient_reduction=True,
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
            cuda_graph=VLLMCudaGraphConfig(mode="FULL_DECODE_ONLY"),
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=1,
            ),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=0.8,
                max_tokens=256,
            ),
        ),
    )


def _set_kimi_k3_batch_invariant(config: Controller.Config) -> Controller.Config:
    """Enable the shared trainer/generator batch-invariant execution mode."""
    config.async_loop = dataclasses.replace(
        config.async_loop,
        target_offpolicy_steps=0,
        windowed_fifo_batches=1,
    )
    config.trainer = dataclasses.replace(
        config.trainer,
        debug=_BATCH_INVARIANT_DEBUG,
        parallelism=dataclasses.replace(
            config.trainer.parallelism,
            enable_sequence_parallel=False,
        ),
    )
    config.generator = dataclasses.replace(
        config.generator,
        debug=_BATCH_INVARIANT_DEBUG,
        reset_kv_cache_on_weight_sync=True,
    )
    return config


def rl_grpo_kimi_k3_debug_varlen_batch_invariant(
    *, seq_len: int = 2048
) -> Controller.Config:
    """On-policy, batch-invariant random-weight Kimi K3 GRPO config."""
    return _set_kimi_k3_batch_invariant(rl_grpo_kimi_k3_debug_varlen(seq_len=seq_len))


def _configure_ci(
    config: Controller.Config,
    *,
    steps: int,
    num_tokens_per_microbatch: int,
    max_generated_tokens: int,
    use_hf_assets_from_env: bool,
) -> Controller.Config:
    config.async_loop.num_training_steps = steps
    config.async_loop.num_samples_per_prompt = 2
    config.async_loop.training_sample_builder.drop_zero_std_reward_groups = False
    config.trainer.training.num_tokens_per_microbatch_per_dp_rank = (
        num_tokens_per_microbatch
    )
    config.generator.sampling.max_tokens = max_generated_tokens
    config.generator.gpu_memory_limit = 0.8
    config.metrics.enable_wandb = False
    config.rollouter.worker.token_env.max_rollout_tokens = (
        config.model.max_context_length
    )
    if use_hf_assets_from_env:
        assets_path = os.environ.get("TORCHTITAN_TEST_HF_ASSETS_PATH")
        if assets_path:
            config.hf_assets_path = assets_path
    return config


def rl_grpo_fsdp2_gen_tp2_no_compile() -> Controller.Config:
    config = rl_grpo_qwen3_0_6b_varlen_no_compile(seq_len=1024)
    config.num_generators = 3
    config.trainer.parallelism.data_parallel_shard_degree = 2
    config.trainer.parallelism.tensor_parallel_degree = 1
    config.generator.parallelism.tensor_parallel_degree = 2
    return _configure_ci(
        config,
        steps=5,
        num_tokens_per_microbatch=2048,
        max_generated_tokens=256,
        use_hf_assets_from_env=True,
    )


def rl_grpo_fsdp2_gen_tp2_compile() -> Controller.Config:
    config = rl_grpo_qwen3_0_6b_varlen(seq_len=1024)
    config.num_generators = 3
    config.trainer.parallelism.data_parallel_shard_degree = 2
    config.trainer.parallelism.tensor_parallel_degree = 1
    config.generator.parallelism.tensor_parallel_degree = 2
    return _configure_ci(
        config,
        steps=5,
        num_tokens_per_microbatch=2048,
        max_generated_tokens=256,
        use_hf_assets_from_env=True,
    )


def rl_grpo_moe_debug_tp4_ep4() -> Controller.Config:
    config = rl_grpo_gpt_oss_debug_varlen_no_compile(seq_len=1024)
    config.trainer.parallelism.data_parallel_shard_degree = 1
    config.trainer.parallelism.tensor_parallel_degree = 4
    config.trainer.parallelism.expert_parallel_degree = 4
    config.generator.parallelism.data_parallel_degree = 1
    config.generator.parallelism.tensor_parallel_degree = 4
    config.generator.parallelism.expert_parallel_degree = 4
    # TODO: use FULL CUDA graphs once GPT-OSS CI enables a
    # CUDA-graph-compatible distributed MoE token dispatcher.
    config.generator.cuda_graph.mode = "NONE"
    return _configure_ci(
        config,
        steps=5,
        num_tokens_per_microbatch=2048,
        max_generated_tokens=256,
        use_hf_assets_from_env=False,
    )


def rl_grpo_checkpoint_save() -> Controller.Config:
    config = rl_grpo_qwen3_0_6b_varlen_checkpoint_test(seq_len=1024)
    config.num_generators = 2
    config.trainer.parallelism.data_parallel_shard_degree = 2
    config.trainer.parallelism.tensor_parallel_degree = 1
    config.generator.parallelism.tensor_parallel_degree = 2
    return _configure_ci(
        config,
        steps=2,
        num_tokens_per_microbatch=2048,
        max_generated_tokens=256,
        use_hf_assets_from_env=True,
    )


def rl_grpo_checkpoint_resume() -> Controller.Config:
    config = rl_grpo_qwen3_0_6b_varlen_checkpoint_test(seq_len=1024)
    config.num_generators = 1
    config.trainer.parallelism.data_parallel_shard_degree = 1
    config.trainer.parallelism.tensor_parallel_degree = 2
    config.generator.parallelism.tensor_parallel_degree = 4
    return _configure_ci(
        config,
        steps=4,
        num_tokens_per_microbatch=2048,
        max_generated_tokens=256,
        use_hf_assets_from_env=True,
    )


def rl_grpo_0_6b_tp4_batch_invariant() -> Controller.Config:
    config = rl_grpo_qwen3_0_6b_varlen_batch_invariant(seq_len=1024)
    config.num_generators = 1
    config.trainer.parallelism.tensor_parallel_degree = 4
    config.generator.parallelism.tensor_parallel_degree = 4
    return _configure_ci(
        config,
        steps=3,
        num_tokens_per_microbatch=2048,
        max_generated_tokens=128,
        use_hf_assets_from_env=True,
    )


def rl_grpo_moe_debug_tp4_ep4_batch_invariant() -> Controller.Config:
    config = rl_grpo_qwen3_moe_debug_varlen_batch_invariant(seq_len=1024)
    return _configure_ci(
        config,
        steps=5,
        num_tokens_per_microbatch=2048,
        max_generated_tokens=256,
        use_hf_assets_from_env=False,
    )


def rl_grpo_qwen3_5_debug_tp2_batch_invariant() -> Controller.Config:
    config = rl_grpo_qwen3_5_debug_varlen_batch_invariant(seq_len=1024)
    config.num_generators = 2
    config.trainer.parallelism.data_parallel_shard_degree = 2
    config.trainer.parallelism.tensor_parallel_degree = 2
    config.generator.parallelism.tensor_parallel_degree = 2
    return _configure_ci(
        config,
        steps=3,
        num_tokens_per_microbatch=1024,
        max_generated_tokens=128,
        use_hf_assets_from_env=False,
    )


def rl_grpo_kimi_k3_debug_batch_invariant() -> Controller.Config:
    config = rl_grpo_kimi_k3_debug_varlen_batch_invariant(seq_len=1024)
    config.num_generators = 2
    config.trainer.parallelism.data_parallel_shard_degree = 2
    config.trainer.parallelism.tensor_parallel_degree = 1
    config.generator.parallelism.tensor_parallel_degree = 1
    return _configure_ci(
        config,
        steps=3,
        num_tokens_per_microbatch=1024,
        max_generated_tokens=128,
        use_hf_assets_from_env=False,
    )


def rl_grpo_moe_debug_dist_moe_tp2_ep4() -> Controller.Config:
    """Dist-MoE on both RL roles, with generator CUDA graphs and uneven DP load.

    Trainer FSDP=2 x TP=2 with EP=4 (4 GPUs); one generator with vLLM DP=2 x TP=2
    and EP=4 (4 GPUs). The two generator DP replicas schedule different numbers of
    requests, so the Dist-MoE ranks see different token counts, which exercises
    the padding that keeps Dist-MoE's equal-token contract. The scratch factor is
    the EP size, the topology maximum, because a rank that receives more rows than
    the plan holds faults instead of raising. Needs an SM100+ GPU and the optional
    ``dist_moe`` package.
    """
    from torchtitan_recipes.models.deepseek_v3 import _require_dist_moe

    _require_dist_moe()
    from torchtitan.config.transform.dist_moe import DistMoeTransform
    from torchtitan.distributed.activation_checkpoint import RegionAC
    from torchtitan.models.common.dist_moe import DistMoeRuntime

    config = rl_grpo_qwen3_moe_debug_varlen()
    expert_parallel_size = 4
    config.model = DistMoeTransform().transform(config.model)
    # Full activation checkpointing would replay dist_moe's forward in backward and
    # corrupt its activation planner; RegionAC with no saved regions does not.
    config.trainer.activation_checkpoint = RegionAC.Config(save_regions=[])
    config.trainer.dist_moe = DistMoeRuntime.Config(
        scratch_capacity_factor=float(expert_parallel_size)
    )
    config.generator.dist_moe_runtime = DistMoeRuntime.Config(
        scratch_capacity_factor=float(expert_parallel_size), inference=True
    )
    # The standard MoE dispatcher cannot be captured in a CUDA graph; Dist-MoE can.
    config.generator.cuda_graph.mode = "FULL_DECODE_ONLY"
    return _configure_ci(
        config,
        steps=3,
        num_tokens_per_microbatch=4096,
        max_generated_tokens=128,
        use_hf_assets_from_env=False,
    )
