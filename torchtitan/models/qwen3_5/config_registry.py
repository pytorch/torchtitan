# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.data import GrainDataLoader
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optimizer import default_adamw, LRSchedulersContainer
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.hf_datasets.multimodal.mm_datasets import MM_DATASETS
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3_8.config_registry import (
    _multimodal_collator_config,
    qwen38_27b,
    qwen38_debugmodel,
    qwen38_debugmodel_moe,
    qwen38_debugmodel_varlen_attn,
)
from torchtitan.trainer import Trainer

from . import model_registry, QWEN3_5_SPECIAL_TOKENS


def _retarget_shared_recipe(
    config: Trainer.Config,
    *,
    flavor: str,
    moe_comm_backend: str | None = None,
    hf_assets_path: str | None = None,
) -> Trainer.Config:
    config.model_spec = model_registry(
        flavor,
        moe_comm_backend=moe_comm_backend,
    )
    config.tokenizer = MultiModalTokenizer.Config(**QWEN3_5_SPECIAL_TOKENS)
    if hf_assets_path is not None:
        config.hf_assets_path = hf_assets_path
    return config


def qwen35_debugmodel() -> Trainer.Config:
    return _retarget_shared_recipe(qwen38_debugmodel(), flavor="debugmodel")


def qwen35_debugmodel_varlen_attn() -> Trainer.Config:
    config = qwen38_debugmodel_varlen_attn()
    config.model_spec = model_registry("debugmodel", attn_backend="varlen")
    config.tokenizer = MultiModalTokenizer.Config(**QWEN3_5_SPECIAL_TOKENS)
    return config


def qwen35_debugmodel_moe() -> Trainer.Config:
    return _retarget_shared_recipe(
        qwen38_debugmodel_moe(),
        flavor="debugmodel_moe",
        moe_comm_backend="standard",
    )


def _release_config(
    flavor: str,
    *,
    lr: float,
    tensor_parallel_degree: int = 1,
    expert_parallel_degree: int = 1,
    activation_checkpoint: FullAC.Config | SelectiveAC.Config,
    last_save_model_only: bool | None = False,
) -> Trainer.Config:
    model_spec = model_registry(
        flavor,
        moe_comm_backend="standard" if expert_parallel_degree > 1 else None,
    )
    checkpoint = (
        CheckpointManager.Config(interval=500)
        if last_save_model_only is None
        else CheckpointManager.Config(
            interval=500,
            last_save_model_only=last_save_model_only,
        )
    )
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model_spec),
            ),
        ),
        hf_assets_path=f"./assets/hf/Qwen3.5-{flavor}",
        tokenizer=MultiModalTokenizer.Config(**QWEN3_5_SPECIAL_TOKENS),
        model_spec=model_spec,
        dataloader=GrainDataLoader.Config(
            dataset=MM_DATASETS["cc12m"],
            collator=_multimodal_collator_config(MM_DATASETS["cc12m"]),
            streaming_shuffle_buffer_size=128,
        ),
        optimizer=default_adamw(lr=lr),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=20),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=4 * 4096,
            max_context_length=4096,
            steps=1000,
            disable_cuda_graphs=expert_parallel_degree > 1,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=tensor_parallel_degree,
            expert_parallel_degree=expert_parallel_degree,
        ),
        checkpoint=checkpoint,
        activation_checkpoint=activation_checkpoint,
    )


def qwen35_0_8b() -> Trainer.Config:
    return _release_config(
        "0.8B",
        lr=5e-3,
        activation_checkpoint=SelectiveAC.Config(),
    )


def qwen35_2b() -> Trainer.Config:
    return _release_config(
        "2B",
        lr=5e-3,
        activation_checkpoint=SelectiveAC.Config(),
    )


def qwen35_4b() -> Trainer.Config:
    return _release_config(
        "4B",
        lr=5e-4,
        activation_checkpoint=FullAC.Config(),
        last_save_model_only=None,
    )


def qwen35_9b() -> Trainer.Config:
    return _release_config(
        "9B",
        lr=5e-4,
        tensor_parallel_degree=2,
        activation_checkpoint=FullAC.Config(),
    )


def qwen35_27b() -> Trainer.Config:
    return _retarget_shared_recipe(
        qwen38_27b(),
        flavor="27B",
        hf_assets_path="./assets/hf/Qwen3.5-27B",
    )


def qwen35_35b_a3b() -> Trainer.Config:
    return _release_config(
        "35B-A3B",
        lr=5e-4,
        tensor_parallel_degree=2,
        expert_parallel_degree=8,
        activation_checkpoint=FullAC.Config(),
    )


def qwen35_122b_a10b() -> Trainer.Config:
    return _release_config(
        "122B-A10B",
        lr=5e-4,
        tensor_parallel_degree=4,
        expert_parallel_degree=8,
        activation_checkpoint=FullAC.Config(),
    )


def qwen35_397b_a17b() -> Trainer.Config:
    return _release_config(
        "397B-A17B",
        lr=5e-4,
        tensor_parallel_degree=8,
        expert_parallel_degree=16,
        activation_checkpoint=FullAC.Config(),
    )
