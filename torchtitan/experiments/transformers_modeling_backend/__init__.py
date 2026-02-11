# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.hf_datasets.text_datasets import build_text_dataloader
from torchtitan.protocols.train_spec import TrainSpec

from .infra.parallelize import parallelize_hf_transformers
from .infra.pipeline import pipeline_hf_transformers
from .model.model import HFTransformerModel

__all__ = [
    "HFTransformerModel",
]


@dataclass
class TitanDenseModelConfig:
    """Arguments for the base TorchTitan model."""

    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int | None = None
    vocab_size: int | None = None
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000
    max_seq_len: int = 2048
    depth_init: bool = True
    use_flex_attn: bool = False
    attn_mask_type: str = "causal"


flavors = {
    "debugmodel": HFTransformerModel.Config(
        titan_dense_config=TitanDenseModelConfig(
            dim=256,
            n_layers=2,
            n_heads=16,
            n_kv_heads=16,
        ),
    ),
    "full": HFTransformerModel.Config(
        titan_dense_config=TitanDenseModelConfig(),
    ),
}


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_configs=flavors,
        parallelize_fn=parallelize_hf_transformers,
        pipelining_fn=pipeline_hf_transformers,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
