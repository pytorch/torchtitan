# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config entry points for the Search-R1 example.

These set the full Search-R1 recipe entirely from the example's config — the core
defaults are unchanged, so every other config keeps vanilla GRPO. ``ConfigManager``
discovers these directly from the example module::

    --module torchtitan.experiments.rl.examples.search_r1 \\
        --config rl_grpo_qwen3_1_7b_search_r1
"""

from __future__ import annotations

import dataclasses

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.experiments.rl.actors.generator import (
    SamplingConfig,
    VLLMCudagraphConfig,
    VLLMGenerator,
)
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.batcher import BatchConfig, Batcher
from torchtitan.experiments.rl.examples.search_r1.rollouter import SearchR1Rollouter
from torchtitan.experiments.rl.losses import DAPOLoss
from torchtitan.experiments.rl.observability.metrics import MetricsProcessor
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.rollout.advantage import GRPOAdvantage
from torchtitan.experiments.rl.trainer import RLTrainer
from torchtitan.models.qwen3 import model_registry


def rl_grpo_qwen3_1_7b_search_r1() -> RLTrainer.Config:
    """GRPO Search-R1 (multi-turn retrieval QA) for Qwen3-1.7B.

    Runs on 8 GPUs: 4 generator (TP=4) + 1 trainer (TP=1), with a dense retrieval
    server on the spare GPUs. Requires a running retrieval server and the QA parquet
    data; see ``README.md``.
    """
    return RLTrainer.Config(
        model_spec=model_registry("1.7B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-1.7B",
        num_steps=500,
        # 32 prompts x group 8 = 256 rollouts/step.
        num_groups_per_rollout_batch=32,
        group_size=8,
        # Validate on a fixed file-order held-out set every 5 steps.
        num_validation_samples=500,
        validation_freq=5,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        # Standard GRPO (advantage normalized by group reward std). Advantage is a
        # post-scoring step on the rollouter; the trainer just consumes it.
        rollouter=SearchR1Rollouter.Config(
            advantage=GRPOAdvantage.Config(should_std_normalize=True),
        ),
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        batcher=Batcher.Config(
            batch=BatchConfig(local_batch_size=1, global_batch_size=48, seq_len=4096),
        ),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=1e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2, decay_type="linear", min_lr_factor=1.0
            ),
            training=TrainingConfig(),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=1,
                disable_loss_parallel=True,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10000,  # only the initial HF load; no mid-run checkpoints
                last_save_model_only=True,
            ),
            # DAPO-style clip-higher (asymmetric clip); no KL / reference model.
            loss=DAPOLoss.Config(
                ratio_clip_low=0.2,
                ratio_clip_high=0.28,
            ),
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
            # TODO(#3668): re-enable cudagraph once the large-batch capture bug is
            # fixed. Full cudagraph capture at large batch corrupts generation on this
            # vLLM build (random tokens + NaN logprobs), so we run eager for now.
            # https://github.com/pytorch/torchtitan/issues/3668
            cudagraph=VLLMCudagraphConfig(enable=False),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=512,
            ),
        ),
    )


def rl_grpo_qwen3_8b_search_r1() -> RLTrainer.Config:
    """GRPO Search-R1 for Qwen3-8B — same recipe as the 1.7B config.

    Only the model and GPU split differ. 8 GPUs: 4 generator (TP=4) + 2 trainer
    (TP=2) + retriever on the spare GPUs. The trainer runs full bf16 (no fp32 master
    copy), so 8B fits on TP=2 and leaves more GPUs for generation — RL is
    generation-bound, so we keep n_generators >= n_trainers.
    """
    config = rl_grpo_qwen3_1_7b_search_r1()
    config.model_spec = model_registry("8B", attn_backend="varlen")
    config.hf_assets_path = "torchtitan/experiments/rl/example_checkpoint/Qwen3-8B"
    config.trainer = dataclasses.replace(
        config.trainer,
        training=TrainingConfig(dtype="bfloat16"),
        parallelism=dataclasses.replace(
            config.trainer.parallelism, tensor_parallel_degree=2
        ),
    )
    # Leave headroom for the trainer->generator weight-sync transient (a full 0.9
    # KV cache OOMs the generator GPUs during sync).
    config.generator = dataclasses.replace(
        config.generator,
        gpu_memory_limit=0.6,
        parallelism=dataclasses.replace(
            config.generator.parallelism, tensor_parallel_degree=4
        ),
    )
    return config
