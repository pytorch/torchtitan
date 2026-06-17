# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config entry points for the SWE (R2E-Gym) coding-agent example.

Run (single 8x H100 box; needs podman + the R2E-Gym images, see README)::

    python -m torchtitan.experiments.rl.train \\
        --module torchtitan.experiments.rl.examples.swe \\
        --config rl_swe_r2e_qwen3_8b --metrics.no-enable-wandb

``rl_swe_r2e_qwen3_1_7b`` is a cheaper variant for pipeline iteration.
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
from torchtitan.experiments.rl.examples.swe.rollouter import SweRollouter
from torchtitan.experiments.rl.losses import DAPOLoss
from torchtitan.experiments.rl.observability.metrics import MetricsProcessor
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.rollout.advantage import AdvantageEstimator
from torchtitan.experiments.rl.trainer import RLTrainer
from torchtitan.models.qwen3 import model_registry

# A SWE rollout is a long tool-calling transcript (issue + many bash observations).
# seq_len must exceed SweRollouter's token_env.max_rollout_tokens (14000) so a full
# rollout fits one training episode; raise both together for harder instances (at a
# memory cost).
_SEQ_LEN = 16384


def _set_context_length(model_spec, max_seq_len: int) -> None:
    """Raise the model's context length to ``max_seq_len``.

    ``Qwen3Model.Config.max_seq_len`` is a read-only property derived from each
    attention layer's RoPE; set the underlying RoPE field on every layer. This
    drives both the model's positional encoding and vLLM's ``max_model_len``
    (which defaults to 4096 -- too short for SWE transcripts)."""
    for layer in model_spec.model.layers:
        rope = getattr(getattr(layer, "attention", None), "rope", None)
        if rope is not None:
            rope.max_seq_len = max_seq_len


def rl_swe_r2e_qwen3_1_7b() -> RLTrainer.Config:
    """GRPO SWE coding-agent (R2E-Gym) for Qwen3-1.7B.

    8 GPUs: 4 generator (TP=4) + 1 trainer (TP=1); the spare GPUs are idle (the
    sandbox runs on CPU in podman). Needs podman and the per-instance R2E-Gym
    images cached locally; see README.
    """
    config = RLTrainer.Config(
        model_spec=model_registry("1.7B", attn_backend="varlen"),
        hf_assets_path="torchtitan/experiments/rl/example_checkpoint/Qwen3-1.7B",
        num_steps=50,
        num_groups_per_rollout_batch=2,
        group_size=4,
        # Validation also boots sandboxes; keep it tiny and infrequent for a smoke.
        num_validation_samples=2,
        validation_freq=1000,
        compile=CompileConfig(enable=True, backend="aot_eager"),
        rollouter=SweRollouter.Config(
            advantage=AdvantageEstimator.Config(should_std_normalize=True),
        ),
        renderer=RendererConfig(name="qwen3", enable_thinking=False),
        metrics=MetricsProcessor.Config(enable_wandb=True),
        batcher=Batcher.Config(
            batch=BatchConfig(
                local_batch_size=1, global_batch_size=8, seq_len=_SEQ_LEN
            ),
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
            loss=DAPOLoss.Config(ratio_clip_low=0.2, ratio_clip_high=0.28),
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
            # fixed; run eager for now.
            cudagraph=VLLMCudagraphConfig(enable=False),
            checkpoint=CheckpointManager.Config(enable=False),
            sampling=SamplingConfig(temperature=1.0, top_p=1.0, max_tokens=2048),
        ),
    )
    _set_context_length(config.model_spec, _SEQ_LEN)
    return config


def rl_swe_r2e_qwen3_8b() -> RLTrainer.Config:
    """GRPO SWE coding-agent (R2E-Gym) for Qwen3-8B -- same recipe as 1.7B.

    8 GPUs: 2 generator (TP=2) + 4 trainer (TP=4). The fp32 trainer needs TP=4 to
    fit; the generator memory cap leaves room for the weight-sync spike.
    """
    config = rl_swe_r2e_qwen3_1_7b()
    config.model_spec = model_registry("8B", attn_backend="varlen")
    _set_context_length(config.model_spec, _SEQ_LEN)
    config.hf_assets_path = "torchtitan/experiments/rl/example_checkpoint/Qwen3-8B"
    config.trainer = dataclasses.replace(
        config.trainer,
        parallelism=dataclasses.replace(
            config.trainer.parallelism, tensor_parallel_degree=4
        ),
    )
    config.generator = dataclasses.replace(
        config.generator,
        gpu_memory_limit=0.6,
        parallelism=dataclasses.replace(
            config.generator.parallelism, tensor_parallel_degree=2
        ),
    )
    return config
