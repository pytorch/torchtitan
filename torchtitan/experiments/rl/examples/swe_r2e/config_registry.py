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

``hf_assets_path`` defaults to ``example_checkpoint/Qwen3-<size>`` (the same
convention as the other RL examples); point it at your downloaded HF weights via
the CLI flag above or the launcher's ``HF_ASSETS_PATH``. The R2E JSONL path comes
from ``SWE_PROMPT_DATA`` (set by the launcher's ``PROMPT_DATA``).

Models: 1.7B (fast full-pipeline smoke) and 8B (the documented target) are
TorchTitan-registered dense models. 30B-A3B (MoE) is the scale-up path. NOTE the
slime recipe's Qwen3.6-35B-A3B is a different (Megatron-only) model not in
TorchTitan's qwen3 registry -- 30B-A3B is the closest TorchTitan MoE.
"""

from __future__ import annotations

import dataclasses
import os

from torchtitan.components.checkpoint import CheckpointManager
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
from torchtitan.experiments.rl.batcher import BatchConfig, Batcher
from torchtitan.experiments.rl.examples.swe_r2e.data import SWER2EDataset
from torchtitan.experiments.rl.examples.swe_r2e.rollouter import SWER2ERollouter
from torchtitan.experiments.rl.losses import DAPOLoss
from torchtitan.experiments.rl.models.vllm_registry import InferenceParallelismConfig
from torchtitan.experiments.rl.observability.metrics import MetricsProcessor
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.trainer import RLTrainer
from torchtitan.models.qwen3 import model_registry
from torchtitan.protocols.model_spec import ModelSpec

# R2E JSONL path, supplied by the launcher (PROMPT_DATA -> SWE_PROMPT_DATA).
# Empty by default; SWER2EDataset raises a clear error if it is not set.
_DEFAULT_DATA = os.environ.get("SWE_PROMPT_DATA", "")
_CKPT_DIR = "torchtitan/experiments/rl/example_checkpoint"
# A coding-agent harness (Claude Code) sends a large prompt: its system prompt +
# ~20 tool schemas + read file contents quickly reach 10k+ tokens. The TorchTitan
# qwen3 specs default max_seq_len to 4096 (search_r1's short QA budget), which
# truncates the agent's first prompt to an empty generation. Raise the model
# context (RoPE max_seq_len, == vLLM max_model_len) and size seq_len to match so a
# full multi-turn episode packs without being dropped by the Batcher.
_SWE_MAX_MODEL_LEN = 24576
_SMOKE_SEQ_LEN = 24576


def _set_max_seq_len(model_spec: ModelSpec, max_seq_len: int) -> None:
    """Raise the model's context length by setting every layer's RoPE max_seq_len
    (``ModelSpec.model.max_seq_len`` is a read-only property derived from it)."""
    for layer in model_spec.model.layers:
        rope = getattr(getattr(layer, "attention", None), "rope", None)
        if rope is not None:
            rope.max_seq_len = max_seq_len


def rl_grpo_qwen3_1_7b_swe_r2e() -> RLTrainer.Config:
    """Fast full-pipeline smoke: Qwen3-1.7B, 1 R2E task x 2 samples -> backward.

    2 GPUs: 1 trainer (TP=1) + 1 generator (TP=1). Pairs with
    ``run_swe_r2e_daytona.sh`` (Daytona sandbox, tight context). Proves the
    sandbox -> Claude Code -> adapter -> grading -> GRPO step path end to end.
    """
    model_spec = model_registry("1.7B", attn_backend="varlen")
    _set_max_seq_len(model_spec, _SWE_MAX_MODEL_LEN)
    return RLTrainer.Config(
        model_spec=model_spec,
        hf_assets_path=f"{_CKPT_DIR}/Qwen3-1.7B",
        num_steps=1,
        num_groups_per_rollout_batch=1,
        group_size=2,
        num_validation_samples=0,
        validation_freq=0,
        compile=CompileConfig(enable=False),
        rollouter=SWER2ERollouter.Config(
            train_dataset=SWER2EDataset.Config(data_path=_DEFAULT_DATA, seed=42),
            validation_dataset=SWER2EDataset.Config(
                data_path=_DEFAULT_DATA, seed=99, shuffle=False
            ),
        ),
        renderer=RendererConfig(name="qwen3"),
        metrics=MetricsProcessor.Config(enable_wandb=False),
        # global_batch_size=1 keeps the smoke to ONE rollout group (2 daytona
        # sandboxes + 2 eval sandboxes): num_tokens_target = global_batch_size *
        # seq_len, and the collection loop boots groups until that target is met.
        # group_size=2 still gives a GRPO pair (advantages computed per group
        # before batching); the batch trains on the packed episode(s) that fit.
        batcher=Batcher.Config(
            batch=BatchConfig(
                local_batch_size=1, global_batch_size=1, seq_len=_SMOKE_SEQ_LEN
            ),
        ),
        trainer=PolicyTrainer.Config(
            optimizer=default_adamw(lr=1e-6),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=1, decay_type="linear", min_lr_factor=1.0
            ),
            training=TrainingConfig(),
            # FullAC: long SWE episodes make the full-vocab fp32 logprob transient
            # large; recompute the forward to fit.
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
            loss=DAPOLoss.Config(ratio_clip_low=0.2, ratio_clip_high=0.28),
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


def rl_grpo_qwen3_8b_swe_r2e() -> RLTrainer.Config:
    """Target smoke: Qwen3-8B, 1 R2E task x 2 samples -> backward.

    6 GPUs: trainer TP=4 (fp32 master) + generator TP=2. Same recipe as the 1.7B
    config; only the model and GPU split differ. Pairs with ``run_swe_r2e_8b.sh``.
    """
    config = rl_grpo_qwen3_1_7b_swe_r2e()
    config.model_spec = model_registry("8B", attn_backend="varlen")
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


def rl_grpo_qwen3_30b_a3b_swe_r2e() -> RLTrainer.Config:
    """Scale-up (MoE): Qwen3-30B-A3B on a SINGLE 8-GPU host.

    TorchTitan RL uses separate (non-colocated) trainer/generator meshes, so 8 GPUs
    are split 4 + 4. Fitting a 30B model in the trainer's 4 GPUs needs aggressive
    memory savings (vs the search_r1 32B dense recipe which used a whole host for
    FSDP and ran multi-generator on MAST):
      - bf16 master weights (``dtype="bfloat16"``, no fp32 master) + ``Adam`` states
        in bf16 (``implementation="fused_opt_states_bf16"``) roughly halve trainer
        memory; FullAC recomputes activations.
      - trainer FSDP-4 (``data_parallel_shard_degree=4``, TP=1); EP via FSDP.
      - generator TP=4 + EP=4 (full expert parallelism across its 4 GPUs).
      - CPU-staged weight sync on both (``weight_sync_direct_rdma=False``).

    This is the closest single-host TorchTitan analogue of the slime Qwen3.6-35B-A3B
    recipe (a different, Megatron-only model). For more headroom / real reward, run
    multi-host on MAST (a whole host for the FSDP trainer + N generator hosts).
    Requires the Qwen3-30B-A3B HF weights on disk.
    """
    config = rl_grpo_qwen3_1_7b_swe_r2e()
    config.model_spec = model_registry("30B-A3B", attn_backend="varlen")
    _set_max_seq_len(config.model_spec, _SWE_MAX_MODEL_LEN)
    config.hf_assets_path = f"{_CKPT_DIR}/Qwen3-30B-A3B"
    # bf16 master + bf16 Adam states to fit 30B in the trainer's 4 GPUs.
    bf16_adam = default_adamw(lr=1e-6)
    bf16_adam.implementation = "fused_opt_states_bf16"
    config.trainer = dataclasses.replace(
        config.trainer,
        optimizer=bf16_adam,
        training=dataclasses.replace(config.trainer.training, dtype="bfloat16"),
        ac_config=FullAC.Config(),
        weight_sync_direct_rdma=False,
        parallelism=dataclasses.replace(
            config.trainer.parallelism,
            data_parallel_shard_degree=4,
            tensor_parallel_degree=1,
        ),
    )
    config.generator = dataclasses.replace(
        config.generator,
        gpu_memory_limit=0.5,
        weight_sync_direct_rdma=False,
        # MoE expert routing's dynamic shapes break vLLM CUDA graph capture
        # ("Cannot copy between CPU and CUDA tensors during CUDA graph capture";
        # see generator.py's MoE-cudagraph TODO) -> run the generator eager.
        cudagraph=VLLMCudagraphConfig(enable=False),
        parallelism=dataclasses.replace(
            config.generator.parallelism,
            tensor_parallel_degree=4,
            expert_parallel_degree=4,
        ),
    )
    return config
