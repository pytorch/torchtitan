# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.datasets.hf_datasets import build_hf_dataloader
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from .infra.parallelize import parallelize_motif
from .infra.pipeline import pipeline_motif
from .model.args import TransformerModelArgs
from .model.model import Transformer

__all__ = [
    "parallelize_motif",
    "pipeline_motif",
    "TransformerModelArgs",
    "Transformer",
    "motif_configs",
]


motif_configs = {
    "debugmodel": TransformerModelArgs(
        dim=256, n_layers=6, n_heads=16, rope_theta=500000, max_seq_len=4096
    ),
    "tiny": TransformerModelArgs(
        dim=2048, n_layers=32, n_heads=16, rope_theta=500000, max_seq_len=16384
    ),
    "10B": TransformerModelArgs(
        dim=4096,
        n_layers=40,
        n_heads=32,
        n_kv_heads=16,
        ffn_dim_multiplier=1,
        multiple_of=16384,
        rope_theta=500000,
        max_seq_len=16384,
    ),
}


register_train_spec(
    TrainSpec(
        name="motif",
        model_cls=Transformer,
        model_args=motif_configs,
        parallelize_fn=parallelize_motif,
        pipelining_fn=pipeline_motif,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_hf_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
)
