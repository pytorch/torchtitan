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

Models: 1.7B (fast full-pipeline smoke) and 8B (the documented target) are the two
the run script wires; 14B and 32B (dense) and 30B-A3B (MoE) are larger scale-up
recipes. All are TorchTitan-registered; point ``hf_assets_path`` at the weights.
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
    config; only the model and GPU split differ. Pairs with
    ``run_swe_r2e_daytona.sh`` (CONFIG=rl_grpo_qwen3_8b_swe_r2e).
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
        parallelism=dataclasses.replace(
            config.trainer.parallelism,
            data_parallel_shard_degree=4,
            tensor_parallel_degree=1,
        ),
    )
    config.generator = dataclasses.replace(
        config.generator,
        gpu_memory_limit=0.5,
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


def rl_grpo_qwen3_32b_swe_r2e() -> RLTrainer.Config:
    """Qwen3-32B (dense) SWE-R2E: single-host FSDP-8 trainer + TP-8 generator.

    The scale target. Trainer: FSDP across 8 GPUs (data_parallel_shard_degree=8,
    TP=1) with bf16 master + bf16 AdamW states (fused_opt_states_bf16) + FullAC to
    fit the long-context backward on one 80GB host (the full-vocab fp32 logprob
    transient dominates memory at this seq_len). Generator is dense TP=8 with
    decode-only CUDA graphs. ``group_size=4`` samples per prompt mirrors slime's
    ``n_samples_per_prompt``. ``global_batch_size=8`` is the FSDP-8 trainer's
    one-row-per-rank floor (below it ``gradient_accumulation_steps`` rounds to 0 and
    no optimizer step runs).
    """
    config = rl_grpo_qwen3_1_7b_swe_r2e()
    config.model_spec = model_registry("32B", attn_backend="varlen")
    _set_max_seq_len(config.model_spec, _SWE_MAX_MODEL_LEN)
    config.hf_assets_path = f"{_CKPT_DIR}/Qwen3-32B"
    config.group_size = 4
    config.num_steps = 1
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
        # Coding-agent edits can be long; raise the per-turn generation cap. slime
        # uses 8192 over a 38k context -> ~4096 ratio-matched to our 20480 budget
        # (the adapter further caps each turn at budget - prompt_len).
        sampling=dataclasses.replace(config.generator.sampling, max_tokens=4096),
        parallelism=dataclasses.replace(
            config.generator.parallelism,
            tensor_parallel_degree=8,
        ),
    )
    # global_batch_size must be a multiple of local_batch_size * trainer_dp_degree
    # (1 * FSDP-8 = 8) or gradient_accumulation_steps rounds to 0 -> no optim step.
    config.batcher = Batcher.Config(
        batch=BatchConfig(
            local_batch_size=1, global_batch_size=8, seq_len=_SMOKE_SEQ_LEN
        ),
    )
    return config


# 14B context cap. Kept below Qwen3-14B's 40960 native max: seq_len sets the
# collection target (num_tokens_target = global_batch * seq_len), so an
# over-large seq_len makes the loop over-collect rollouts (each shorter than
# seq_len) to fill it. 32768 is ~1.3x the 32B run's 24576 with a saner target.
_QWEN3_14B_MAX_LEN = 32768


def rl_grpo_qwen3_14b_swe_r2e() -> RLTrainer.Config:
    """Qwen3-14B (dense) SWE-R2E: FSDP-8 mixed-precision trainer + TP-4 generator(s).

    14B leaves enough headroom (vs 32B) to keep fp32-master mixed precision AND a
    larger context. Trainer: FSDP-8 (dp_shard=8, TP=1) with the default
    TrainingConfig (dtype=float32 master + mixed_precision_param=bfloat16 compute +
    fp32 reduce) + FullAC fit seq_len=40960 -- 14B's native context, ~1.7x the 32B
    run's 24576 and closer to slime's long-context coding rollouts. Generator is
    dense TP=4 (14B fits 4 GPUs); run ``--num_generators N`` for more parallel
    rollout. Per-turn gen cap 8192 (slime's MAX_GEN_LEN). global_batch_size=8 is the
    FSDP-8 one-row-per-rank floor; group_size=8, num_steps defaults to the smoke
    (override on the CLI for a longer run).
    """
    config = rl_grpo_qwen3_1_7b_swe_r2e()
    config.model_spec = model_registry("14B", attn_backend="varlen")
    _set_max_seq_len(config.model_spec, _QWEN3_14B_MAX_LEN)
    config.hf_assets_path = f"{_CKPT_DIR}/Qwen3-14B"
    config.group_size = 8  # slime n_samples_per_prompt
    config.trainer = dataclasses.replace(
        config.trainer,
        parallelism=dataclasses.replace(
            config.trainer.parallelism,
            data_parallel_shard_degree=8,
            tensor_parallel_degree=1,
        ),
    )
    config.generator = dataclasses.replace(
        config.generator,
        gpu_memory_limit=0.8,
        sampling=dataclasses.replace(config.generator.sampling, max_tokens=8192),
        parallelism=dataclasses.replace(
            config.generator.parallelism,
            tensor_parallel_degree=4,
        ),
    )
    config.batcher = Batcher.Config(
        batch=BatchConfig(
            local_batch_size=1, global_batch_size=8, seq_len=_QWEN3_14B_MAX_LEN
        ),
    )
    return config
