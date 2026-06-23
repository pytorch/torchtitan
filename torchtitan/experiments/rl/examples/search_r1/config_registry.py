# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config entry points for the Search-R1 example.

These set the full Search-R1 recipe entirely from the example's config -- the core
defaults are unchanged, so every other config keeps vanilla GRPO. ``ConfigManager``
discovers these directly from the example module::

    --module search_r1 \\
        --config rl_grpo_qwen3_1_7b_search_r1
"""

from __future__ import annotations

import dataclasses

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.experiments.rl.actors.generator import (
    SamplingConfig,
    VLLMCudagraphConfig,
    VLLMGenerator,
)
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.batcher import BatchConfig, Batcher
from torchtitan.experiments.rl.examples.search_r1.rollouter import SearchR1Rollouter
from torchtitan.experiments.rl.losses import DAPOLoss
from torchtitan.experiments.rl.models.vllm_registry import InferenceParallelismConfig
from torchtitan.experiments.rl.observability.metrics import MetricsProcessor
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.rollout.advantage import AdvantageEstimator
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
        num_groups_per_rollout_batch=32,
        group_size=8,
        num_validation_samples=500,
        validation_freq=5,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=SearchR1Rollouter.Config(
            advantage=AdvantageEstimator.Config(should_std_normalize=True),
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
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                # First run with an empty checkpoint folder loads the HF weights;
                # subsequent restarts resume from the latest mid-run checkpoint.
                initial_load_in_hf=True,
                # Mid-run checkpoints every `interval` steps so a preempted long
                # run resumes instead of restarting from step 1. last_save full
                # (not model-only) so the final checkpoint is also resumable;
                # keep_latest_k bounds disk use for the larger models.
                interval=50,
                last_save_model_only=False,
                keep_latest_k=3,
            ),
            # DAPO-style clip-higher (asymmetric clip); no KL / reference model.
            loss=DAPOLoss.Config(
                ratio_clip_low=0.2,
                ratio_clip_high=0.28,
            ),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=4,
            ),
            # cudagraph on: decode-only graphs (FULL_DECODE_ONLY) are safe at this
            # config's large batch; plain full graphs corrupted here before. See #3668.
            cudagraph=VLLMCudagraphConfig(enable=True),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(
                temperature=1.0,
                top_p=1.0,
                max_tokens=512,
            ),
        ),
    )


def rl_grpo_qwen3_8b_search_r1() -> RLTrainer.Config:
    """GRPO Search-R1 for Qwen3-8B -- same recipe as the 1.7B config.

    Only the model and GPU split differ. 8 GPUs: 2 generator (TP=2) + 4 trainer
    (TP=4) + retriever on the spare GPUs. The fp32 trainer needs TP=4 to avoid OOM.
    """
    # TODO: use mixed precision (fp32 master + bf16 compute) via FSDP + activation
    # checkpointing, which is more memory-efficient and could keep the split generator-heavy.
    config = rl_grpo_qwen3_1_7b_search_r1()
    config.model_spec = model_registry("8B", attn_backend="varlen")
    config.hf_assets_path = "torchtitan/experiments/rl/example_checkpoint/Qwen3-8B"
    config.trainer = dataclasses.replace(
        config.trainer,
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


def rl_grpo_qwen3_32b_search_r1() -> RLTrainer.Config:
    """GRPO Search-R1 for Qwen3-32B (dense): single-host trainer + N generators.

    Trainer: single-host FSDP=8 (``data_parallel_shard_degree=8``, TP=1) in pure
    bf16 (no fp32 master, fp32 gradient reduce) so 32B fits one 80GB host, with full
    activation checkpointing. Weight sync uses direct GPU-to-GPU RDMA. Each generator
    is TP=8 (1 host) with decode-only CUDA graphs. With ``--num_generators 3``:
    1 controller + 1 trainer + 3 generator = 5 hosts.
    """
    config = rl_grpo_qwen3_1_7b_search_r1()
    config.model_spec = model_registry("32B", attn_backend="varlen")
    config.hf_assets_path = "torchtitan/experiments/rl/example_checkpoint/Qwen3-32B"
    # Trainer per-layer compile off; the rollout's vLLM compile (generator
    # cudagraph below) is separate and stays on.
    config.compile = dataclasses.replace(config.compile, enable=False)
    config.trainer = dataclasses.replace(
        config.trainer,
        # Pure bf16 (no fp32 master) halves per-rank memory so FSDP=8 fits;
        # gradients are still reduced in fp32.
        training=dataclasses.replace(
            config.trainer.training,
            dtype="bfloat16",
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
        ),
        # Full activation checkpointing to fit 32B's forward on one host.
        ac_config=FullAC.Config(),
        # Direct GPU-to-GPU RDMA weight sync. Works with a single-host trainer +
        # multiple generators; requires Monarch RDMA (set False for CPU-staged).
        weight_sync_direct_rdma=True,
        parallelism=dataclasses.replace(
            config.trainer.parallelism,
            data_parallel_shard_degree=8,
            tensor_parallel_degree=1,
        ),
    )
    config.generator = dataclasses.replace(
        config.generator,
        gpu_memory_limit=0.6,
        # Direct RDMA weight sync (must match the trainer).
        weight_sync_direct_rdma=True,
        # Decode-only CUDA graphs: capture pure-decode batches, run prefill eagerly.
        cudagraph=VLLMCudagraphConfig(enable=True, mode="FULL_DECODE_ONLY"),
        parallelism=dataclasses.replace(
            config.generator.parallelism, tensor_parallel_degree=8
        ),
    )
    # One proc mesh per generator; override with --num_generators.
    config.num_generators = 3
    return config


def rl_grpo_qwen3_32b_search_r1_fsdp16() -> RLTrainer.Config:
    """GRPO Search-R1 for Qwen3-32B (dense): 2-host trainer (FSDP=16) + N generators.

    Like ``rl_grpo_qwen3_32b_search_r1`` but shards the trainer 16-way across two
    hosts (``data_parallel_shard_degree=16``, TP=1). 16-way sharding leaves room for
    an fp32 master and the faster SelectiveAC. Uses CPU-staged weight sync (direct
    RDMA is single-host-trainer only). With ``--num_generators 3``: 1 controller +
    2 trainer + 3 generator = 6 hosts.
    """
    config = rl_grpo_qwen3_32b_search_r1()
    config.trainer = dataclasses.replace(
        config.trainer,
        parallelism=dataclasses.replace(
            config.trainer.parallelism,
            data_parallel_shard_degree=16,
            tensor_parallel_degree=1,
        ),
        ac_config=SelectiveAC.Config(),
        # 16-way sharding affords an fp32 master; weight sync stays CPU-staged
        # (direct RDMA is single-host-trainer only).
        training=dataclasses.replace(config.trainer.training, dtype="float32"),
        weight_sync_direct_rdma=False,
    )
    config.generator = dataclasses.replace(
        config.generator, weight_sync_direct_rdma=False
    )
    return config


def rl_grpo_qwen3_32b_search_r1_fsdp16_rdma() -> RLTrainer.Config:
    """EXPERIMENTAL: FSDP=16 (2-host trainer) + direct-RDMA weight sync.

    Same as ``rl_grpo_qwen3_32b_search_r1`` (bf16 + ``direct_rdma=True``) but with the
    trainer sharded 16-way across two hosts. Direct RDMA currently works only with a
    single-host trainer; across a multi-host trainer it hits a Monarch RDMA
    limitation, so this recipe does not train yet -- use the CPU-staged
    ``rl_grpo_qwen3_32b_search_r1_fsdp16`` for multi-host trainers.
    """
    # TODO: 2-host-trainer RDMA test case -- does NOT work yet. Direct RDMA across a
    # multi-host trainer hits a Monarch RDMA limitation; fix the multi-host RDMA path
    # so this trains like the single-host FSDP=8 RDMA recipe.
    config = rl_grpo_qwen3_32b_search_r1()
    config.trainer = dataclasses.replace(
        config.trainer,
        # 16-way shard; keeps the base's bf16 + direct_rdma=True, SelectiveAC fits.
        parallelism=dataclasses.replace(
            config.trainer.parallelism,
            data_parallel_shard_degree=16,
            tensor_parallel_degree=1,
        ),
        ac_config=SelectiveAC.Config(),
    )
    return config
