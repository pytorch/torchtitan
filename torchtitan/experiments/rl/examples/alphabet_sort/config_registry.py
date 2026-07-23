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

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import (
    CompileConfig,
    DebugConfig,
    OverrideConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.experiments.rl.actors.generator import (
    SamplingConfig,
    VLLMCudagraphConfig,
    VLLMGenerator,
)
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.batch_invariance import BatchInvariantFlexConverter
from torchtitan.experiments.rl.components.batcher import BatchConfig, Batcher
from torchtitan.experiments.rl.components.training_sample_builder import (
    TrainingSampleBuilder,
)
from torchtitan.experiments.rl.controller import (
    AsyncLoopConfig,
    Controller,
    ValidationConfig,
)
from torchtitan.experiments.rl.examples.alphabet_sort import AlphabetSortRollouter
from torchtitan.experiments.rl.losses import GRPOLoss
from torchtitan.experiments.rl.models.cast_linear import LMHeadCastConverter
from torchtitan.experiments.rl.models.vllm_registry import InferenceParallelismConfig
from torchtitan.experiments.rl.observability.metrics import MetricsProcessor
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.routing.inter_generator_router import (
    InterGeneratorRouter,
)
from torchtitan.experiments.rl.routing.strategies import (
    LeastLoadedRoutingStrategy,
    StickySessionRoutingStrategy,
)
from torchtitan.models.gpt_oss import model_registry as gpt_oss_model_registry
from torchtitan.models.qwen3 import model_registry
from torchtitan.models.qwen3_5 import model_registry as qwen3_5_model_registry
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.model_spec import ModelSpec

_BATCH_INVARIANT_DEBUG = DebugConfig(batch_invariant=True, deterministic=True)


def _qwen3_rl_model_registry(
    flavor: str,
    *,
    attn_backend: str,
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    """``qwen3.model_registry`` for RL, with the lm_head fp32 cast always on.

    RL logprob / KL math needs the lm_head logits in fp32, so every RL config
    runs ``LMHeadCastConverter`` on top of whatever converters it passes.
    """
    converters = list(converters or [])
    converters.append(LMHeadCastConverter.Config())
    spec = model_registry(flavor, attn_backend=attn_backend, converters=converters)
    return spec


def rl_grpo_qwen3_0_6b_varlen() -> Controller.Config:
    """GRPO training config for Qwen3-0.6B (6 GPUs: 4 gen + 2 train)."""
    num_samples_per_prompt = 8
    return Controller.Config(
        model_spec=_qwen3_rl_model_registry("0.6B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        async_loop=AsyncLoopConfig(
            num_training_steps=10,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(num_samples=20),
            batcher=Batcher.Config(
                batch=BatchConfig(local_batch_size=2, seq_len=2048),
            ),
        ),
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=AlphabetSortRollouter.Config(),
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        generator_router=InterGeneratorRouter.Config(
            strategy=StickySessionRoutingStrategy.Config(
                fallback_strategy=LeastLoadedRoutingStrategy.Config()
            )
        ),
        metrics=MetricsProcessor.Config(enable_wandb=True),
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
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=ChunkedLossWrapper.Config(num_chunks=8, loss_fn=GRPOLoss.Config()),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=4,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=0.95,
                max_tokens=700,
            ),
        ),
    )


def rl_grpo_qwen3_0_6b_flex() -> Controller.Config:
    """GRPO training config for Qwen3-0.6B with flex attention (4 GPUs: 2 gen + 2 train)."""
    num_samples_per_prompt = 8
    return Controller.Config(
        model_spec=_qwen3_rl_model_registry("0.6B", attn_backend="flex"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        async_loop=AsyncLoopConfig(
            num_training_steps=10,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(num_samples=20),
            batcher=Batcher.Config(
                batch=BatchConfig(local_batch_size=2, seq_len=2048),
            ),
        ),
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=AlphabetSortRollouter.Config(),
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=True),
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
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=ChunkedLossWrapper.Config(num_chunks=8, loss_fn=GRPOLoss.Config()),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=2,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=0.95,
                max_tokens=100,
            ),
        ),
    )


def rl_grpo_qwen3_0_6b_flex_batch_invariant() -> Controller.Config:
    """GRPO training config for Qwen3-0.6B with flex attention and batch invariance
    for bitwise-identical numerics between trainer and generator (4 GPUs: 2 gen + 2 train).

    Trainer keeps fp32 master weights; FSDP mixed precision
    (mixed_precision_param="bfloat16", the default) casts them to bf16 for the
    forward (even at data_parallel_shard_degree=1), matching the bf16 generator.
    """
    config = rl_grpo_qwen3_0_6b_flex()
    config.model_spec = _qwen3_rl_model_registry(
        "0.6B",
        attn_backend="flex",
        converters=[BatchInvariantFlexConverter.Config()],
    )
    block_size = config.model_spec.model.layers[0].attention.inner_attention.block_size
    config.async_loop.batcher = dataclasses.replace(
        config.async_loop.batcher, per_sample_pad_multiple=block_size
    )
    # Batch invariance requires strict on-policy: the generator must run the
    # latest weights before generating so trainer/generator logprobs stay
    # bitwise-identical (bit_wise/logprob_diff == 0) every step, not just step 1.
    config.async_loop.max_offpolicy_steps = 0
    config.trainer = dataclasses.replace(
        config.trainer,
        debug=_BATCH_INVARIANT_DEBUG,
        # fp32 master weights; FSDP mixed precision casts to bf16 for the forward.
        training=TrainingConfig(),
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
    ``VarlenAttention.window_size``.
    """
    num_samples_per_prompt = 8
    return Controller.Config(
        model_spec=gpt_oss_model_registry("20b", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/gpt-oss-20b",
        async_loop=AsyncLoopConfig(
            num_training_steps=10,
            num_prompts_per_train_step=5,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(num_samples=20),
            batcher=Batcher.Config(
                batch=BatchConfig(local_batch_size=2, seq_len=2048),
            ),
        ),
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=AlphabetSortRollouter.Config(),
        renderer=RendererConfig(name="gpt_oss", enable_thinking=False),
        generator_router=InterGeneratorRouter.Config(
            strategy=StickySessionRoutingStrategy.Config(
                fallback_strategy=LeastLoadedRoutingStrategy.Config()
            )
        ),
        metrics=MetricsProcessor.Config(enable_wandb=True),
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
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=ChunkedLossWrapper.Config(num_chunks=8, loss_fn=GRPOLoss.Config()),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=4,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
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
    return Controller.Config(
        model_spec=gpt_oss_model_registry("debugmodel", attn_backend="varlen"),
        hf_assets_path="tests/assets/tokenizer",
        async_loop=AsyncLoopConfig(
            num_training_steps=3,
            num_prompts_per_train_step=5,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(num_samples=20),
            batcher=Batcher.Config(
                batch=BatchConfig(local_batch_size=2, seq_len=2048),
            ),
            training_sample_builder=TrainingSampleBuilder.Config(
                drop_zero_std_reward_groups=False,
            ),
        ),
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=AlphabetSortRollouter.Config(),
        # Debug tokenizer (vocab 2048, matches debugmodel); the gpt_oss renderer
        # needs gpt-oss special tokens absent here, so use the qwen3 renderer
        # like the other debug configs.
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=True),
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
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            loss=ChunkedLossWrapper.Config(num_chunks=8, loss_fn=GRPOLoss.Config()),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=4,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=0.8,
                top_p=0.95,
                max_tokens=50,
            ),
        ),
    )


def rl_grpo_gpt_oss_debug_varlen_batch_invariant() -> Controller.Config:
    """Small GPT-OSS debug config in deterministic + batch-invariant mode.

    Trainer keeps fp32 master weights; FSDP mixed precision
    (mixed_precision_param="bfloat16", the default) casts them to bf16 for the
    forward (even at data_parallel_shard_degree=1), matching the bf16 generator.
    """
    batch_invariant_config = DebugConfig(batch_invariant=True, deterministic=True)
    num_samples_per_prompt = 8
    return Controller.Config(
        model_spec=gpt_oss_model_registry("debugmodel", attn_backend="varlen"),
        hf_assets_path="tests/assets/tokenizer",
        async_loop=AsyncLoopConfig(
            num_training_steps=3,
            # Batch invariance: strict on-policy so trainer/generator logprobs
            # stay bitwise-identical every step.
            max_offpolicy_steps=0,
            num_prompts_per_train_step=5,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(num_samples=20),
            batcher=Batcher.Config(
                batch=BatchConfig(local_batch_size=2, seq_len=2048),
            ),
            training_sample_builder=TrainingSampleBuilder.Config(
                drop_zero_std_reward_groups=False,
            ),
        ),
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=AlphabetSortRollouter.Config(),
        # Debug tokenizer (vocab 2048, matches debugmodel); the gpt_oss renderer
        # needs gpt-oss special tokens absent here, so use the qwen3 renderer
        # like the other debug configs.
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            # fp32 master weights; FSDP mixed precision casts to bf16 for the
            # forward (mixed_precision_param="bfloat16" is the default).
            training=TrainingConfig(),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
                enable_sequence_parallel=False,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            debug=batch_invariant_config,
            loss=ChunkedLossWrapper.Config(num_chunks=8, loss_fn=GRPOLoss.Config()),
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
            checkpoint=CheckpointManager.Config(enable=False),
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
    return Controller.Config(
        model_spec=_qwen3_rl_model_registry("1.7B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-1.7B",
        async_loop=AsyncLoopConfig(
            num_training_steps=10,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(num_samples=20),
            batcher=Batcher.Config(
                batch=BatchConfig(local_batch_size=2, seq_len=2048),
            ),
        ),
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=AlphabetSortRollouter.Config(),
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=True),
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
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=ChunkedLossWrapper.Config(num_chunks=8, loss_fn=GRPOLoss.Config()),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=4,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
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
    return Controller.Config(
        model_spec=_qwen3_rl_model_registry("14B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-14B",
        async_loop=AsyncLoopConfig(
            num_training_steps=10,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(num_samples=20),
            batcher=Batcher.Config(
                batch=BatchConfig(local_batch_size=2, seq_len=2048),
            ),
        ),
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=AlphabetSortRollouter.Config(),
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=True),
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
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=ChunkedLossWrapper.Config(num_chunks=8, loss_fn=GRPOLoss.Config()),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=8,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
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
    return Controller.Config(
        model_spec=model_registry("debugmodel_moe", attn_backend="varlen"),
        hf_assets_path="tests/assets/tokenizer",
        async_loop=AsyncLoopConfig(
            num_training_steps=5,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(num_samples=20),
            batcher=Batcher.Config(
                batch=BatchConfig(local_batch_size=2, seq_len=2048),
            ),
            training_sample_builder=TrainingSampleBuilder.Config(
                drop_zero_std_reward_groups=False,
            ),
        ),
        # MoE EP all-to-all path issues unpinned D2H copies that block
        # torch.compile and CUDA graph capture; disable both.
        compile=CompileConfig(enable=False),
        rollouter=AlphabetSortRollouter.Config(),
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=8e-4),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=2,
                tensor_parallel_degree=2,
                data_parallel_replicate_degree=1,
                expert_parallel_degree=4,
            ),
            checkpoint=CheckpointManager.Config(
                enable=False,
                interval=10,
                last_save_model_only=False,
            ),
            loss=ChunkedLossWrapper.Config(num_chunks=8, loss_fn=GRPOLoss.Config()),
        ),
        generator=VLLMGenerator.Config(
            # Disable torch.compile + CUDA graph capture: the EP all-to-all
            # path issues an unpinned D2H copy of split sizes that the
            # piecewise/full graph capture rejects.
            cudagraph=VLLMCudagraphConfig(enable=False),
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=2,
                tensor_parallel_degree=2,
                expert_parallel_degree=4,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=0.95,
                max_tokens=50,
            ),
        ),
    )


def rl_grpo_qwen3_moe_debug_deepep() -> Controller.Config:
    """Debug MoE config on the DeepEP v2 backend with a cudagraph-capturable generator
    (8 GPUs: 4 gen + 4 train).

    Same EP/TP/DP layout as ``rl_grpo_qwen3_moe_debug_varlen`` (trainer FSDP=2/TP=2/EP=4,
    generator DP=2/TP=2/EP=4), but the MoE uses the DeepEP v2 comm backend. Unlike the
    standard all-to-all -- whose unpinned D2H split-size copy blocks CUDA graph capture, so
    that config disables it -- DeepEP v2's inference dispatch is a static, host-sync-free
    EXPAND layout, so this generator enables CUDA graph capture.

    Per-role config from ONE shared model_spec: the trainer uses it as-is (compact,
    host-synced, backward-able DeepEP path), while the generator applies per-actor
    overrides (``generator.override``) to its own copy (``fused_swiglu`` +
    ``deepep_override`` with ``cudagraphable=True``) to switch its dispatchers to the
    cudagraph-able EXPAND layout. The overrides touch only the generator's spec, so the
    trainer and weight sync are unaffected.
    """
    config = rl_grpo_qwen3_moe_debug_varlen()
    config.model_spec = model_registry(
        "debugmodel_moe", attn_backend="varlen", moe_comm_backend="deepep"
    )
    # Generator-only overrides -> cudagraph-able DeepEP EXPAND dispatch; trainer keeps compact.
    # FULL_AND_PIECEWISE: decode captured FULL (incl. the expand MoE), prefill breakable.
    config.generator.override = OverrideConfig(
        imports=[
            "torchtitan.overrides.fused_swiglu.fused_swiglu",
            "torchtitan.overrides.fused_swiglu.fused_grouped_experts",
            (
                "torchtitan.overrides.moe_token_dispatcher.deepep_override",
                {"cudagraphable": True},
            ),
        ]
    )
    config.generator.cudagraph = VLLMCudagraphConfig(
        enable=True, mode="FULL_AND_PIECEWISE"
    )
    # Two inference knobs to set per workload (no golden default; here EP=4):
    #  * max_num_batched_tokens: vLLM's per-step token budget. We expose the knob (default
    #    None -> vLLM's own default of 2048). Decide it from your input/rollout sequence
    #    length -- it is effectively the longest input sequence length the engine batches
    #    (vLLM's 2048 default is just a stand-in for knowing that).
    #  * num_max_tokens_per_rank: per-rank EXPAND-dispatch capacity, REQUIRED by the
    #    deepep_override. For a dropless model (highest memory) set it to
    #    longest_sequence_length // sp == max_num_batched_tokens // sp; lower it gradually to
    #    save memory (trading off dropped tokens).
    config.generator.max_num_batched_tokens = 2048
    num_max_tokens_per_rank = (
        config.generator.max_num_batched_tokens
        // config.generator.parallelism.expert_parallel_degree
    )
    for block in config.model_spec.model.layers:
        moe = getattr(block, "moe", None)
        if moe is None:
            continue
        moe.routed_experts.token_dispatcher.num_max_tokens_per_rank = (
            num_max_tokens_per_rank
        )
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
    return Controller.Config(
        model_spec=model_registry(
            "debugmodel_moe", attn_backend="varlen", moe_comm_backend="standard"
        ),
        hf_assets_path="tests/assets/tokenizer",
        async_loop=AsyncLoopConfig(
            num_training_steps=10,
            # Batch invariance: strict on-policy so trainer/generator logprobs
            # stay bitwise-identical every step.
            max_offpolicy_steps=0,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(num_samples=20),
            batcher=Batcher.Config(
                batch=BatchConfig(local_batch_size=2, seq_len=2048),
            ),
            training_sample_builder=TrainingSampleBuilder.Config(
                drop_zero_std_reward_groups=False,
            ),
        ),
        # MoE EP all-to-all path issues unpinned D2H copies that block
        # torch.compile and CUDA graph capture; disable both.
        compile=CompileConfig(enable=False),
        rollouter=AlphabetSortRollouter.Config(),
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=8e-4),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            # fp32 master weights; FSDP mixed precision casts to bf16 for the
            # forward (mixed_precision_param="bfloat16" is the default).
            training=TrainingConfig(),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=2,
                tensor_parallel_degree=2,
                data_parallel_replicate_degree=1,
                expert_parallel_degree=4,
                enable_sequence_parallel=False,
            ),
            checkpoint=CheckpointManager.Config(
                enable=False,
                interval=10,
                last_save_model_only=False,
            ),
            debug=_BATCH_INVARIANT_DEBUG,
            loss=ChunkedLossWrapper.Config(num_chunks=8, loss_fn=GRPOLoss.Config()),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            cudagraph=VLLMCudagraphConfig(enable=False),
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=2,
                tensor_parallel_degree=2,
                expert_parallel_degree=4,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
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
    return Controller.Config(
        model_spec=model_registry("30B-A3B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-30B-A3B",
        async_loop=AsyncLoopConfig(
            num_training_steps=10,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(num_samples=20),
            batcher=Batcher.Config(
                batch=BatchConfig(local_batch_size=2, seq_len=2048),
            ),
        ),
        compile=CompileConfig(enable=False),
        rollouter=AlphabetSortRollouter.Config(),
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=1e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(dtype="bfloat16"),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=2,
                data_parallel_replicate_degree=1,
                tensor_parallel_degree=2,
                expert_parallel_degree=4,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=ChunkedLossWrapper.Config(num_chunks=8, loss_fn=GRPOLoss.Config()),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            cudagraph=VLLMCudagraphConfig(enable=False),
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=2,
                tensor_parallel_degree=2,
                expert_parallel_degree=4,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
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

    * ``fused_swiglu`` fuses the dense and grouped-experts gate+up projections
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
        "torchtitan.overrides.fused_swiglu.fused_grouped_experts",
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
    return Controller.Config(
        model_spec=_qwen3_rl_model_registry("0.6B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-0.6B",
        num_generators=3,
        async_loop=AsyncLoopConfig(
            num_training_steps=10,
            # Batch invariance: strict on-policy so trainer/generator logprobs
            # stay bitwise-identical every step.
            max_offpolicy_steps=0,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(num_samples=20),
            batcher=Batcher.Config(
                batch=BatchConfig(local_batch_size=2, seq_len=2048),
            ),
        ),
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=AlphabetSortRollouter.Config(),
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=2e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=2,
                enable_sequence_parallel=False,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            debug=batch_invariant_config,
            loss=ChunkedLossWrapper.Config(num_chunks=8, loss_fn=GRPOLoss.Config()),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=2,
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


def _qwen3_5_rl_model_registry(
    flavor: str,
    *,
    attn_backend: str = "varlen",
    converters: list[ModelConfigConverter.Config] | None = None,
) -> ModelSpec:
    """``qwen3_5.model_registry`` for RL, with the lm_head fp32 cast always on.

    RL logprob / KL math needs the lm_head logits in fp32, so every RL config
    runs ``LMHeadCastConverter`` on top of whatever converters it passes.
    """
    converters = list(converters or [])
    converters.append(LMHeadCastConverter.Config())
    return qwen3_5_model_registry(
        flavor, attn_backend=attn_backend, converters=converters
    )


def rl_grpo_qwen3_5_9b_varlen() -> Controller.Config:
    """Qwen3.5-9B GRPO with trainer and generator TP=2 (6 GPUs)."""
    group_size = 8
    return Controller.Config(
        model_spec=_qwen3_5_rl_model_registry("9B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3.5-9B",
        async_loop=AsyncLoopConfig(
            num_training_steps=10,
            num_groups_per_train_step=8,
            group_size=group_size,
            validation=ValidationConfig(num_samples=20),
            batcher=Batcher.Config(
                batch=BatchConfig(local_batch_size=1, seq_len=2048),
            ),
        ),
        compile=CompileConfig(enable=False),
        rollouter=AlphabetSortRollouter.Config(),
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=1e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(dtype="bfloat16"),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=2,
                tensor_parallel_degree=2,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=ChunkedLossWrapper.Config(num_chunks=8, loss_fn=GRPOLoss.Config()),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            # GDN decode supports full capture; prefill breaks into eager pieces.
            cudagraph=VLLMCudagraphConfig(enable=True, mode="FULL_AND_PIECEWISE"),
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=2,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
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
    config.async_loop = dataclasses.replace(config.async_loop, max_offpolicy_steps=0)
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
    group_size = 8
    return Controller.Config(
        model_spec=_qwen3_5_rl_model_registry("debugmodel", attn_backend="varlen"),
        hf_assets_path="tests/assets/tokenizer",
        async_loop=AsyncLoopConfig(
            num_training_steps=5,
            num_groups_per_train_step=8,
            group_size=group_size,
            validation=ValidationConfig(num_samples=20),
            batcher=Batcher.Config(
                batch=BatchConfig(local_batch_size=1, seq_len=2048),
            ),
        ),
        compile=CompileConfig(enable=False),
        rollouter=AlphabetSortRollouter.Config(),
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=1e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_type="linear",
            ),
            training=TrainingConfig(dtype="bfloat16"),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=2,
                tensor_parallel_degree=2,
            ),
            checkpoint=CheckpointManager.Config(enable=False),  # random-init weights
            loss=ChunkedLossWrapper.Config(num_chunks=8, loss_fn=GRPOLoss.Config()),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            cudagraph=VLLMCudagraphConfig(enable=True, mode="FULL_AND_PIECEWISE"),
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=2,
            ),
            checkpoint=CheckpointManager.Config(enable=False),
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
    config.async_loop = dataclasses.replace(config.async_loop, max_offpolicy_steps=0)
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
