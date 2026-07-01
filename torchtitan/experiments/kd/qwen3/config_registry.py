# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config registry for knowledge distillation (KD) / quantization-aware
distillation (QAD) experiments with Qwen3 models.

QAD distills from a bf16 teacher into an NVFP4 fake-quantized student. The
student's MoE experts and dense linears are fake-quantized via
:func:`~torchtitan.components.quantization.nvfp4_fake_quant.apply_nvfp4_fake_quant`
(wired as the ``post_model_init_fn``); the teacher stays bf16.
"""

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import default_adamw
from torchtitan.components.quantization.nvfp4_fake_quant import apply_nvfp4_fake_quant
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.experiments.kd.trainer import KDTrainer
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.models.qwen3 import model_registry


def qwen3_moe_debug() -> KDTrainer.Config:
    """Debug KD config with a tiny Qwen3 MoE model (no fake quant)."""
    return KDTrainer.Config(
        loss=CrossEntropyLoss.Config(),
        hf_assets_path="./tests/assets/tokenizer",
        model_spec=model_registry("debugmodel_moe"),
        optimizer=default_adamw(lr=3e-4),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=2),
        training=TrainingConfig(
            local_batch_size=4,
            seq_len=4096,
            steps=10,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4_test",
        ),
        metrics=MetricsProcessor.Config(log_freq=1),
        checkpoint=CheckpointManager.Config(
            interval=10,
            last_save_model_only=False,
        ),
        activation_checkpoint=SelectiveAC.Config(),
        parallelism=ParallelismConfig(
            expert_parallel_degree=1,
        ),
        # KD-specific settings
        temperature=2.0,
        alpha=0.5,
    )


def qwen3_moe_debug_qad() -> KDTrainer.Config:
    """Debug QAD (quantization-aware distillation) config with a tiny Qwen3 MoE
    model.

    Same as :func:`qwen3_moe_debug` but with NVFP4 fake quantization applied to
    the student's MoE experts AND dense linears (w4a4). The teacher stays bf16.
    Self-contained (tiny model, ``c4_test``, no external checkpoint) -- the
    quickest way to exercise the full QAD path end to end.
    """
    config = qwen3_moe_debug()
    config.post_model_init_fn = apply_nvfp4_fake_quant
    return config


def qwen3_30b_a3b_qad() -> KDTrainer.Config:
    """QAD config for Qwen3-30B-A3B.

    Distills from a bf16 teacher into an NVFP4 fake-quantized student. Both the
    student and the (frozen) teacher are initialized from the same HF checkpoint;
    the student's MoE experts and dense linears are NVFP4 fake-quantized.
    """
    return KDTrainer.Config(
        loss=CrossEntropyLoss.Config(),
        hf_assets_path="./assets/hf/Qwen3-30B-A3B",
        model_spec=model_registry("30B-A3B", attn_backend="flex"),
        optimizer=default_adamw(lr=5e-6, betas=(0.9, 0.999), weight_decay=0.0),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=100,
            decay_ratio=0.1,
            decay_type="cosine",
        ),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=2048,
            steps=100,
        ),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="c4",
        ),
        metrics=MetricsProcessor.Config(log_freq=10),
        checkpoint=CheckpointManager.Config(
            enable=True,
            initial_load_in_hf=True,
            last_save_in_hf=True,
        ),
        activation_checkpoint=SelectiveAC.Config(),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            expert_parallel_degree=1,
        ),
        # KD-specific settings
        temperature=2.0,
        alpha=0.5,
        # QAD: fake-quantize the student (experts + linears); teacher stays bf16
        post_model_init_fn=apply_nvfp4_fake_quant,
    )
