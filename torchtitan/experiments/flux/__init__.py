# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

from torchtitan.components.loss import cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.experiments.flux.dataset.flux_dataset import build_flux_dataloader
from torchtitan.experiments.flux.parallelize_flux import parallelize_flux
from torchtitan.protocols.train_spec import register_train_spec, TrainSpec

from .model.model import FluxModel, FluxModelArgs

__all__ = [
    "FluxModelArgs",
    "FluxModel",
    "flux_configs",
    "parallelize_flux",
]


flux_configs = {
    "flux-dev": FluxModelArgs(
        in_channels=64,
        out_channels=64,
        vec_in_dim=768,
        context_in_dim=512,
        hidden_size=3072,
        mlp_ratio=4.0,
        num_heads=24,
        depth=19,
        depth_single_blocks=38,
        axes_dim=(16, 56, 56),
        theta=10_000,
        qkv_bias=True,
        guidance_embed=True,
    )
}


register_train_spec(
    TrainSpec(
        name="flux",
        cls=FluxModel,
        config=flux_configs,
        parallelize_fn=parallelize_flux,
        pipelining_fn=None,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_flux_dataloader,
        build_tokenizer_fn=None,
        loss_fn=cross_entropy_loss,
    )
)
