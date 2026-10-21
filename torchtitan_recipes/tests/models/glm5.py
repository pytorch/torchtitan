# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GLM-5 configurations used by tests and local development."""

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import DefaultActivationRematConfig
from torchtitan.models.common.config_utils import (
    decoder_vocab_size,
    DEFAULT_DEBUG_MODEL_SEQ_LEN,
)
from torchtitan.models.glm5 import build_model_config, MTPLoss
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer


def glm5_debugmodel(
    seq_len: int = DEFAULT_DEBUG_MODEL_SEQ_LEN,
    *,
    parallelism: ParallelismConfig | None = None,
    flavor: str = "glm5_2_debugmodel",
    mtp_scale: float | None = None,
) -> Trainer.Config:
    """Return a deterministic small GLM-5 training recipe."""
    if parallelism is None:
        parallelism = ParallelismConfig()
    model_config = build_model_config(
        flavor,
        parallelism=parallelism,
        seq_len=seq_len,
    )
    num_mtp_layers = len(model_config.mtp_layers or ())
    if mtp_scale is not None and num_mtp_layers == 0:
        raise ValueError("mtp_scale may only be set for a GLM MTP flavor")
    if mtp_scale is not None and mtp_scale < 0:
        raise ValueError("mtp_scale must be non-negative")
    num_tokens = 8 * seq_len
    loss_config = (
        MTPLoss.Config(
            global_vocab_size=decoder_vocab_size(model_config),
            mtp_scale=0.3 if mtp_scale is None else mtp_scale,
        )
        if model_config.mtp_layers
        else CrossEntropyLoss.Config(
            global_vocab_size=decoder_vocab_size(model_config),
        )
    )
    return Trainer.Config(
        loss=ChunkedLossWrapper.Config(loss_fn=loss_config),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model=model_config,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
            num_mtp_layers=num_mtp_layers,
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)]
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_ratio=0.8,
                decay_type="linear",
                min_lr_factor=0.0,
            ),
        ),
        training=TrainingConfig(
            disable_cuda_graphs=True,
            num_tokens_per_microbatch_per_dp_rank=num_tokens,
            max_context_length=seq_len,
            steps=10,
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_remat=DefaultActivationRematConfig(),
    )


def glm5_2_scaled_debugmodel(
    seq_len: int = 4096,
    *,
    parallelism: ParallelismConfig | None = None,
) -> Trainer.Config:
    return glm5_debugmodel(
        seq_len,
        parallelism=parallelism,
        flavor="glm5_2_scaled_debugmodel",
    )


def glm5_2_debugmodel_mtp(
    seq_len: int = DEFAULT_DEBUG_MODEL_SEQ_LEN,
    *,
    parallelism: ParallelismConfig | None = None,
    mtp_scale: float = 0.3,
) -> Trainer.Config:
    return glm5_debugmodel(
        seq_len,
        parallelism=parallelism,
        flavor="glm5_2_debugmodel_mtp",
        mtp_scale=mtp_scale,
    )
