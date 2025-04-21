# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.


from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.experiments.flux.dataset.flux_dataset import build_flux_dataloader
from torchtitan.experiments.flux.loss import build_mse_loss
from torchtitan.experiments.flux.model.autoencoder import AutoEncoderParams
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
        context_in_dim=4096,
        hidden_size=3072,
        mlp_ratio=4.0,
        num_heads=24,
        depth=19,
        depth_single_blocks=38,
        axes_dim=(16, 56, 56),
        theta=10_000,
        qkv_bias=True,
        autoencoder_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=(1, 2, 4, 4),
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-schnell": FluxModelArgs(
        in_channels=64,
        out_channels=64,
        vec_in_dim=768,
        context_in_dim=4096,
        hidden_size=3072,
        mlp_ratio=4.0,
        num_heads=24,
        depth=19,
        depth_single_blocks=38,
        axes_dim=(16, 56, 56),
        theta=10_000,
        qkv_bias=True,
        autoencoder_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=(1, 2, 4, 4),
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
    "flux-debug": FluxModelArgs(
        in_channels=64,
        out_channels=64,
        vec_in_dim=768,
        context_in_dim=4096,
        hidden_size=3072,
        mlp_ratio=4.0,
        num_heads=24,
        depth=2,
        depth_single_blocks=2,
        axes_dim=(16, 56, 56),
        theta=10_000,
        qkv_bias=True,
        autoencoder_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=(1, 2, 4, 4),
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
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
        build_loss_fn=build_mse_loss,
    )
)
