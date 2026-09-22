# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config entry points for the Search-R1 example.

These set the full Search-R1 recipe entirely from the example's config — the core
defaults are unchanged, so every other config keeps vanilla GRPO. ``ConfigManager``
discovers these directly from the example module::

    --module search_r1 \\
        --config rl_grpo_qwen3_1_7b_search_r1
"""

from __future__ import annotations

import dataclasses

from renderers import Qwen3RendererConfig

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.components.renderer import from_renderers
from torchtitan.config import (
    CompileConfig,
    OverrideConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.muse_glimmer import model_registry as muse_glimmer_model_registry
from torchtitan.models.qwen3 import model_registry
from torchtitan.rl.controller import AsyncLoopConfig, Controller, ValidationConfig
from torchtitan.rl.distributed.parallelism import InferenceParallelismConfig
from torchtitan.rl.examples.search_r1.data import SearchR1Dataset
from torchtitan.rl.examples.search_r1.env import SearchR1Env
from torchtitan.rl.examples.search_r1.rubric import RewardExactMatch
from torchtitan.rl.generator import SamplingConfig, VLLMCudaGraphConfig, VLLMGenerator
from torchtitan.rl.losses import DAPOLoss
from torchtitan.rl.model.muse_glimmer.renderer import MuseGlimmerRendererConfig
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.rollout.advantage import AdvantageEstimator
from torchtitan.rl.rollout.environment import TokenEnv
from torchtitan.rl.rollout.rollouter import Rollouter, RolloutWorker
from torchtitan.rl.rubric import Rubric
from torchtitan.rl.trainer import Trainer

# TODO: Enable CUDA graphs for RL trainers after eager/graph numerics parity is
# verified.


def _search_r1_rollouter_config() -> Rollouter.Config:
    return Rollouter.Config(
        train_dataset=SearchR1Dataset.Config(filename="train.parquet", seed=42),
        validation_dataset=SearchR1Dataset.Config(
            filename="test.parquet",
            seed=99,
            data_source="nq",
            shuffle=False,
        ),
        worker=RolloutWorker.Config(
            rubric=Rubric.Config(
                reward_fns=[RewardExactMatch.Config(weight=1.0)],
                truncation_reward=0.0,
            ),
            message_env=SearchR1Env.Config(),
            token_env=TokenEnv.Config(
                max_rollout_tokens=3072,
                max_num_turns=4,
            ),
            advantage=AdvantageEstimator.Config(should_std_normalize=True),
        ),
    )


def rl_grpo_qwen3_1_7b_search_r1() -> Controller.Config:
    """GRPO Search-R1 (multi-turn retrieval QA) for Qwen3-1.7B.

    Runs on 8 GPUs: 4 generator (TP=4) + 1 trainer (TP=1), with a dense retrieval
    server on the spare GPUs. Requires a running retrieval server and the QA parquet
    data; see ``README.md``.
    """
    seq_len = 4096
    model_config = model_registry("1.7B", seq_len=seq_len, attn_backend="varlen")
    return Controller.Config(
        model=model_config,
        hf_assets_path="torchtitan/rl/example_checkpoint/Qwen3-1.7B",
        async_loop=AsyncLoopConfig(
            num_training_steps=500,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=8,
            validation=ValidationConfig(num_samples=500),
        ),
        compile=CompileConfig(backend="aot_eager"),
        rollouter=_search_r1_rollouter_config(),
        renderer=from_renderers(Qwen3RendererConfig(enable_thinking=False)),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=Trainer.Config(
            optimizer=default_adamw(lr=1e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2, decay_type="linear", min_lr_factor=1.0
            ),
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=seq_len,
                max_context_length=seq_len,
            ),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=1,
            ),
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,  # first run loads HF; restarts resume from DCP
                # Mid-run checkpoints so a preempted run resumes; full last save
                # (not model-only) keeps it resumable; keep_latest_k caps disk.
                interval=50,
                last_save_model_only=False,
                keep_latest_k=3,
            ),
            # DAPO-style clip-higher (asymmetric clip); no KL / reference model.
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=DAPOLoss.Config(
                    ratio_clip_low=0.2,
                    ratio_clip_high=0.28,
                    global_vocab_size=decoder_vocab_size(model_config),
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=4,
            ),
            cuda_graph=VLLMCudaGraphConfig(mode="FULL"),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=512,
            ),
        ),
    )


def rl_grpo_qwen3_8b_search_r1() -> Controller.Config:
    """GRPO Search-R1 for Qwen3-8B — same recipe as the 1.7B config.

    Only the model and GPU split differ. 8 GPUs: 2 generator (TP=2) + 4 trainer
    (TP=4) + retriever on the spare GPUs. The fp32 trainer needs TP=4 to avoid OOM.
    """
    # TODO: use mixed precision (fp32 master + bf16 compute) via FSDP + activation
    # checkpointing, which is more memory-efficient and could keep the split generator-heavy.
    config = rl_grpo_qwen3_1_7b_search_r1()
    config.model = model_registry(
        "8B",
        seq_len=config.trainer.training.max_context_length,
        attn_backend="varlen",
    )
    config.hf_assets_path = "torchtitan/rl/example_checkpoint/Qwen3-8B"
    loss_config = config.trainer.loss
    assert isinstance(loss_config, ChunkedLossWrapper.Config)
    assert isinstance(loss_config.loss_fn, DAPOLoss.Config)
    config.trainer = dataclasses.replace(
        config.trainer,
        loss=dataclasses.replace(
            loss_config,
            loss_fn=dataclasses.replace(
                loss_config.loss_fn,
                global_vocab_size=decoder_vocab_size(config.model),
            ),
        ),
        parallelism=dataclasses.replace(
            config.trainer.parallelism, tensor_parallel_degree=4
        ),
    )
    # 0.6 (vs the 0.9 default) reserves room for the weight-sync memory spike, which
    # OOMs the 8B generator otherwise.
    # TODO(@meetv18): the spike is likely GPU-Direct weight transfer being on by default;
    # make the transfer device configurable (CPU default) so this cap can be raised.
    config.generator = dataclasses.replace(
        config.generator,
        gpu_memory_limit=0.6,
        parallelism=dataclasses.replace(
            config.generator.parallelism, tensor_parallel_degree=2
        ),
    )
    return config


def rl_grpo_qwen3_30b_a3b_deepep_search_r1_perf() -> Controller.Config:
    """GRPO Search-R1 for Qwen3-30B-A3B MoE with a DeepEP v2 CUDA graph generator.

    DeepEP v2 runs multi-node on H100 (NVLink intra-node + IB/RoCE inter-node), so unlike
    a HybridEP generator (whose all-to-all is intra-node only) this generator may span
    nodes. Qwen3-30B-A3B has 4 KV heads, so the generator TP must be <=4. The trainer
    keeps the compact (host-synced, backward-able) DeepEP path; the generator applies the
    ``deepep_override`` to switch its dispatchers to the CUDA-graph-compatible EXPAND
    layout. Applies the same ``fused_swiglu`` + ``helion_rope`` perf overrides (CUDA-only)
    as ``rl_grpo_qwen3_30b_a3b_varlen_perf``.
    """
    seq_len = 4096
    model_config = model_registry(
        "30B-A3B",
        seq_len=seq_len,
        attn_backend="varlen",
        moe_comm_backend="deepep",
    )

    # Same opt-in throughput overrides as rl_grpo_qwen3_30b_a3b_varlen_perf, applied
    # independently to the trainer and generator actors.
    perf_imports = [
        "torchtitan.overrides.fused_swiglu.fused_swiglu",
        "torchtitan.overrides.helion_rope.helion_cos_sin_rope",
    ]

    config = Controller.Config(
        model=model_config,
        hf_assets_path="torchtitan/rl/example_checkpoint/Qwen3-30B-A3B",
        num_generators=2,  # TODO: TBD -- number of generator proc meshes to spawn
        async_loop=AsyncLoopConfig(
            num_training_steps=500,
            num_prompts_per_train_step=32,  # TODO: TBD
            num_samples_per_prompt=8,  # TODO: TBD
            validation=ValidationConfig(num_samples=500),
        ),
        compile=None,
        rollouter=_search_r1_rollouter_config(),
        renderer=from_renderers(Qwen3RendererConfig(enable_thinking=False)),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=Trainer.Config(
            optimizer=default_adamw(lr=1e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2, decay_type="linear", min_lr_factor=1.0
            ),
            # TODO: Tune the trainer token budget and maximum context length.
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=seq_len,
                max_context_length=seq_len,
            ),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=8,  # TODO: TBD
                tensor_parallel_degree=1,
                expert_parallel_degree=8,
            ),
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=50,
                last_save_model_only=False,
                keep_latest_k=3,
            ),
            loss=DAPOLoss.Config(
                ratio_clip_low=0.2,
                ratio_clip_high=0.28,
                global_vocab_size=decoder_vocab_size(model_config),
            ),
            override=OverrideConfig(imports=list(perf_imports)),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(  # single node generator
                data_parallel_degree=1,
                tensor_parallel_degree=4,
                expert_parallel_degree=4,
            ),
            cuda_graph=VLLMCudaGraphConfig(mode="FULL"),
            checkpointer=None,
            sampling=SamplingConfig(temperature=1.0, top_p=1.0, max_tokens=512),
            # Generator-only: DeepEP CUDA graph EXPAND dispatch on top of the perf overrides.
            override=OverrideConfig(
                imports=[
                    *perf_imports,
                    (
                        "torchtitan.overrides.moe_token_dispatcher.deepep_override",
                        {"cuda_graph_compatible": True},
                    ),
                ]
            ),
        ),
    )
    # vLLM's per-step token budget. The wrapper derives DeepEP's per-rank buffer capacity
    # from this scheduler limit, CUDA graph capture sizes, CP, and SP.
    config.generator.max_num_batched_tokens = 2048  # TODO: TBD
    return config


def rl_grpo_muse_glimmer_30b_search_r1() -> Controller.Config:
    """GRPO/DAPO Search-R1 for Muse Glimmer 30B.

    8 GPUs: 6 trainer (FSDP=3 x TP=2) + 2 generator (TP=2), with a dense retrieval
    server on spare capacity. Requires a running retrieval server and the QA parquet
    data; see ``README.md``.

    Two constraints are specific to this model:

    * **Generator TP <= 2.** Muse Glimmer has 2 KV heads, so attention cannot be
      tensor-split further. Scale the trainer with FSDP rather than TP.
    * **Full activation checkpointing is required.** Adam's m/v are allocated on the
      *first* ``optimizer.step()``, so per-GPU memory jumps by roughly 8 bytes/param
      between step 1 and step 2 (~37 GB/GPU here, sharded 6 ways). With the default
      ``SelectiveAC`` that jump OOMs at step 2; ``FullAC`` frees the activation
      headroom it needs.

    varlen attention is used for both roles so the trainer and the vLLM generator run
    one model config. The state-dict adapter handles the HF checkpoint's Q/K RoPE layout
    on load, and the renderer handles Muse Glimmer's harmony chat
    format and ATEM tool calls.
    """
    model_config = muse_glimmer_model_registry("30B", attn_backend="varlen")
    return Controller.Config(
        model=model_config,
        hf_assets_path="torchtitan/rl/example_checkpoint/Muse-Glimmer-30B",
        async_loop=AsyncLoopConfig(
            num_training_steps=500,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=8,
            validation=ValidationConfig(num_samples=500),
        ),
        compile=None,
        rollouter=_search_r1_rollouter_config(),
        renderer=MuseGlimmerRendererConfig(),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        trainer=Trainer.Config(
            optimizer=default_adamw(lr=1e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2, decay_type="linear", min_lr_factor=1.0
            ),
            training=TrainingConfig(
                disable_cuda_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=4096,
                max_context_length=4096,
            ),
            activation_checkpoint=FullAC.Config(),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=3,
                tensor_parallel_degree=2,
            ),
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,  # first run loads HF; restarts resume from DCP
                interval=50,
                last_save_model_only=False,
                keep_latest_k=3,
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=DAPOLoss.Config(
                    ratio_clip_low=0.2,
                    ratio_clip_high=0.28,
                    global_vocab_size=decoder_vocab_size(model_config),
                ),
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=2,  # <= 2 KV heads
            ),
            cuda_graph=VLLMCudaGraphConfig(mode="NONE"),
            checkpointer=None,
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=4096,
            ),
        ),
    )
