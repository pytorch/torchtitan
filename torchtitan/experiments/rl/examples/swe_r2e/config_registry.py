# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config entry points for the SWE R2E coding-agent (Claude Code) example.

``ConfigManager`` discovers these directly from the example module::

    python -m torchtitan.experiments.rl.train \
        --module swe_r2e --config rl_grpo_qwen3_8b_swe_r2e \
        --hf_assets_path <path/to/Qwen3-8B>

Each function returns a complete ``Controller.Config`` for the async RL loop.
``hf_assets_path`` defaults to ``example_checkpoint/Qwen3-<size>`` (the same
convention as the other RL examples); point it at your downloaded HF weights via
the CLI flag above or the launcher's ``HF_ASSETS_PATH``. The R2E JSONL path comes
from ``SWE_PROMPT_DATA`` (set by the launcher's ``PROMPT_DATA``).

Recipes:
  - 1.7B / 8B: single-host smokes (prove the sandbox -> Claude Code -> adapter ->
    grading -> GRPO step path end to end).
  - 32B: the scale target. ``_fsdp16`` / ``_fsdp24`` are the multi-host async
    runs the ``mast_rl`` launcher wraps (a 2- or 3-host FSDP trainer + N generator
    hosts), with rollout collection overlapping training (``max_offpolicy_steps``).
"""

from __future__ import annotations

import dataclasses
import os

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import default_adamw
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.experiments.rl.actors.generator import (
    SamplingConfig,
    VLLMCudagraphConfig,
    VLLMGenerator,
)
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.components.batcher import BatchConfig, Batcher
from torchtitan.experiments.rl.controller import (
    AsyncLoopConfig,
    Controller,
    ValidationConfig,
)
from torchtitan.experiments.rl.examples.swe_r2e.data import SWER2EDataset
from torchtitan.experiments.rl.examples.swe_r2e.rollouter import SWER2ERollouter
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
from torchtitan.models.qwen3 import model_registry
from torchtitan.protocols.model_spec import ModelSpec

# R2E JSONL path, supplied by the launcher (PROMPT_DATA -> SWE_PROMPT_DATA).
# Empty by default; SWER2EDataset raises a clear error if it is not set.
_DEFAULT_DATA = os.environ.get("SWE_PROMPT_DATA", "")
_CKPT_DIR = "torchtitan/experiments/rl/example_checkpoint"
# qwen3 specs default max_seq_len to 4096; _set_max_seq_len raises every layer's
# RoPE max_seq_len (== vLLM max_model_len) so a full coding-agent episode fits.
_SWE_MAX_MODEL_LEN = 24576
_SMOKE_SEQ_LEN = 24576


def _set_max_seq_len(model_spec: ModelSpec, max_seq_len: int) -> None:
    """Raise the model's context length by setting every layer's RoPE max_seq_len
    (``ModelSpec.model.max_seq_len`` is a read-only property derived from it)."""
    for layer in model_spec.model.layers:
        rope = getattr(getattr(layer, "attention", None), "rope", None)
        if rope is not None:
            rope.max_seq_len = max_seq_len


def _qwen3_rl_model_registry(flavor: str, *, attn_backend: str) -> ModelSpec:
    """``qwen3.model_registry`` with the lm_head fp32 cast always on.

    RL logprob / KL math needs the lm_head logits in fp32; ``LMHeadCastConverter``
    makes the (chunked) loss apply an fp32 lm_head on top of the bf16 model.
    """
    return model_registry(
        flavor, attn_backend=attn_backend, converters=[LMHeadCastConverter.Config()]
    )


def _swe_rollouter() -> SWER2ERollouter.Config:
    """Train/validation datasets for the coding-agent rollouter (rubric + env
    defaults live on the rollouter Config)."""
    return SWER2ERollouter.Config(
        train_dataset=SWER2EDataset.Config(data_path=_DEFAULT_DATA, seed=42),
        validation_dataset=SWER2EDataset.Config(
            data_path=_DEFAULT_DATA, seed=99, shuffle=False
        ),
    )


def rl_grpo_qwen3_1_7b_swe_r2e() -> Controller.Config:
    """Fast full-pipeline smoke: Qwen3-1.7B, 1 R2E task x 2 samples -> backward.

    2 GPUs: 1 trainer (TP=1) + 1 generator (TP=1). ``max_offpolicy_steps=0`` keeps
    the smoke fully on-policy (sync): one group of 2 sibling rollouts, one step.
    Pairs with ``run_swe_r2e_daytona.sh``. Proves the sandbox -> Claude Code ->
    adapter -> grading -> GRPO step path end to end.
    """
    return Controller.Config(
        model_spec=_qwen3_rl_model_registry("1.7B", attn_backend="varlen"),
        hf_assets_path=f"{_CKPT_DIR}/Qwen3-1.7B",
        compile=CompileConfig(enable=False),
        async_loop=AsyncLoopConfig(
            num_training_steps=1,
            num_groups_per_train_step=1,
            group_size=2,
            max_offpolicy_steps=0,
            validation=ValidationConfig(num_samples=0),
            batcher=Batcher.Config(
                batch=BatchConfig(local_batch_size=1, seq_len=_SMOKE_SEQ_LEN),
            ),
        ),
        rollouter=_swe_rollouter(),
        renderer=RendererConfig(name="qwen3"),
        generator_router=InterGeneratorRouter.Config(
            strategy=StickySessionRoutingStrategy.Config(
                fallback_strategy=LeastLoadedRoutingStrategy.Config()
            )
        ),
        metrics=MetricsProcessor.Config(enable_wandb=False),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=1e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=1, decay_type="linear", min_lr_factor=1.0
            ),
            training=TrainingConfig(),
            # FullAC: long SWE episodes make activation memory large; recompute
            # the forward to fit. The chunked loss (below) bounds the logits.
            ac_config=FullAC.Config(),
            parallelism=ParallelismConfig(
                data_parallel_shard_degree=1,
                tensor_parallel_degree=1,
            ),
            checkpoint=CheckpointManager.Config(
                enable=True,
                initial_load_in_hf=True,
                interval=10000,  # initial HF load only; no mid-run checkpoints
                last_save_model_only=True,
            ),
            # Chunked loss splits [B, L, V] logits over the sequence dim so the
            # fp32 lm_head + GRPO loss fit at long context (no full-vocab OOM).
            loss=ChunkedLossWrapper.Config(num_chunks=8, loss_fn=GRPOLoss.Config()),
        ),
        generator=VLLMGenerator.Config(
            model_dtype="bfloat16",
            parallelism=InferenceParallelismConfig(
                data_parallel_degree=1,
                tensor_parallel_degree=1,
            ),
            gpu_memory_limit=0.6,
            cudagraph=VLLMCudagraphConfig(enable=True, mode="FULL_DECODE_ONLY"),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(temperature=1.0, top_p=1.0, max_tokens=2048),
        ),
    )


def rl_grpo_qwen3_8b_swe_r2e() -> Controller.Config:
    """Target smoke: Qwen3-8B, 1 R2E task x 2 samples -> backward.

    6 GPUs: trainer TP=4 + generator TP=2. Same recipe as the 1.7B config; only
    the model and GPU split differ.
    """
    config = rl_grpo_qwen3_1_7b_swe_r2e()
    config.model_spec = _qwen3_rl_model_registry("8B", attn_backend="varlen")
    _set_max_seq_len(config.model_spec, _SWE_MAX_MODEL_LEN)
    config.hf_assets_path = f"{_CKPT_DIR}/Qwen3-8B"
    config.trainer = dataclasses.replace(
        config.trainer,
        parallelism=dataclasses.replace(
            config.trainer.parallelism, tensor_parallel_degree=4
        ),
    )
    config.generator = dataclasses.replace(
        config.generator,
        parallelism=dataclasses.replace(
            config.generator.parallelism, tensor_parallel_degree=2
        ),
    )
    return config


def rl_grpo_qwen3_32b_swe_r2e() -> Controller.Config:
    """Qwen3-32B (dense) SWE-R2E: single FSDP-8 trainer host + TP-8 generator host.

    The FSDP-8 baseline (2 MAST hosts: 1 trainer + 1 generator). Trainer FSDP-8
    (data_parallel_shard_degree=8, TP=1) with bf16 master + bf16 AdamW states
    (fused_opt_states_bf16) + FullAC to fit 32B in one 8x80GB host; the chunked
    loss bounds the long-context (24576) logits. Generator is dense TP=8 with
    decode-only CUDA graphs. ``group_size=8`` mirrors slime's n_samples_per_prompt.
    For more trainer headroom + overlapped collection use the ``_fsdp16`` recipe.
    """
    config = rl_grpo_qwen3_1_7b_swe_r2e()
    config.model_spec = _qwen3_rl_model_registry("32B", attn_backend="varlen")
    _set_max_seq_len(config.model_spec, _SWE_MAX_MODEL_LEN)
    config.hf_assets_path = f"{_CKPT_DIR}/Qwen3-32B"
    config.async_loop = dataclasses.replace(
        config.async_loop,
        num_training_steps=1,
        num_groups_per_train_step=8,
        group_size=8,
        max_offpolicy_steps=0,
    )
    # bf16 master + bf16 Adam states to fit 32B in the FSDP-8 trainer's 8 GPUs.
    bf16_adam = default_adamw(lr=1e-6)
    bf16_adam.implementation = "fused_opt_states_bf16"
    config.trainer = dataclasses.replace(
        config.trainer,
        optimizer=bf16_adam,
        training=dataclasses.replace(config.trainer.training, dtype="bfloat16"),
        parallelism=dataclasses.replace(
            config.trainer.parallelism,
            data_parallel_shard_degree=8,
            tensor_parallel_degree=1,
        ),
    )
    config.generator = dataclasses.replace(
        config.generator,
        # Coding-agent edits can be long; raise the per-turn generation cap.
        sampling=dataclasses.replace(config.generator.sampling, max_tokens=4096),
        parallelism=dataclasses.replace(
            config.generator.parallelism, tensor_parallel_degree=8
        ),
    )
    return config


def _scale_32b_multihost(
    config: Controller.Config,
    *,
    trainer_dp_shard: int,
    num_generators: int,
    num_training_steps: int,
    max_offpolicy_steps: int,
) -> Controller.Config:
    """Turn the FSDP-8 32B baseline into a multi-host async run with TRUE FSDP.

    Widens the trainer to ``data_parallel_shard_degree=trainer_dp_shard`` (16 or
    24, over trainer_dp_shard/8 hosts) on the default FUSED-QKV 32B spec. PR #3807
    made the fused wqkv init + checkpoint save/load hooks gather to Replicate before
    the per-kv-head reshape, so FSDP > n_kv_heads (=8) no longer crashes ("unflatten
    unevenly sharded"); before #3807 this needed non-fused QKV. The wider shard
    halves/thirds per-GPU param+optimizer memory vs FSDP-8, so we drop the
    bf16-master memory hack back to fp32 master + fp32 Adam (better numerics).
    FullAC + chunked loss stay. ``num_generators`` TP-8 generator hosts collect
    rollouts that overlap training by up to ``max_offpolicy_steps`` steps, so each
    train step is short instead of blocking on the slow Claude Code rollouts; the
    wider shard also speeds fwd/bwd.
    """
    config.async_loop = dataclasses.replace(
        config.async_loop,
        num_training_steps=num_training_steps,
        # 4 groups/step (down from the base 8): a step needs all 8 rollouts of
        # each of N groups graded, and grading the slow Claude rollouts dominates,
        # so a smaller per-step batch lands steps sooner. dp_degree=16 still gets
        # 4*group_size=32 rollouts to pack.
        num_groups_per_train_step=2,
        max_offpolicy_steps=max_offpolicy_steps,
    )
    config.num_generators = num_generators
    # fp32 master + fp32 Adam (default): the wider FSDP shard has the headroom.
    config.trainer = dataclasses.replace(
        config.trainer,
        optimizer=default_adamw(lr=1e-6),
        training=dataclasses.replace(config.trainer.training, dtype="float32"),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2, decay_type="linear", min_lr_factor=1.0
        ),
        parallelism=dataclasses.replace(
            config.trainer.parallelism,
            data_parallel_shard_degree=trainer_dp_shard,
            tensor_parallel_degree=1,
        ),
    )
    return config


def rl_grpo_qwen3_32b_swe_r2e_fsdp16() -> Controller.Config:
    """Qwen3-32B SWE-R2E, multi-host async: TRUE FSDP-16 trainer + 3 TP-8 gens.

    6 MAST hosts: 1 controller + 2 trainer (data_parallel_shard_degree=16, TP=1) +
    3 generator (TP=8). Default fused QKV; PR #3807 made the fused init + checkpoint
    hooks gather-to-Replicate before the per-kv-head reshape, so FSDP-16 (>
    n_kv_heads=8) shards cleanly. fp32 master + FullAC + chunked loss. Rollout
    collection overlaps training by up to ``max_offpolicy_steps=2``
    (the trainer trains step N's batch while the generators collect step N+1's), so
    each train step is short (fwd/bwd + CPU-staged weight sync) instead of blocking
    on the slow Claude Code rollouts. ``num_groups_per_train_step=8`` x
    ``group_size=8`` = 64 rollouts per step.
    """
    config = rl_grpo_qwen3_32b_swe_r2e()
    return _scale_32b_multihost(
        config,
        trainer_dp_shard=16,
        num_generators=5,
        num_training_steps=10,
        max_offpolicy_steps=1,
    )


def rl_grpo_qwen3_32b_swe_r2e_fsdp24() -> Controller.Config:
    """Qwen3-32B SWE-R2E, multi-host async: TRUE FSDP-24 trainer + 3 TP-8 gens.

    7 MAST hosts: 1 controller + 3 trainer (data_parallel_shard_degree=24, TP=1) +
    3 generator (TP=8). Same fused-QKV (PR #3807) + fp32-master + async overlap as
    ``_fsdp16`` with a wider trainer shard (more memory headroom / faster fwd-bwd).
    """
    config = rl_grpo_qwen3_32b_swe_r2e()
    return _scale_32b_multihost(
        config,
        trainer_dp_shard=24,
        num_generators=5,
        num_training_steps=10,
        max_offpolicy_steps=1,
    )
