# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
from functools import partial

import torch.nn as nn

from torchtitan.components.loss import build_mse_loss
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.protocols.model_spec import ModelSpec

from .flux_datasets import FluxDataLoader
from .model.autoencoder import AutoEncoderParams
from .model.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    Modulation,
    QKNorm,
    SelfAttention,
    SingleStreamBlock,
)
from .model.model import FluxModel
from .model.state_dict_adapter import FluxStateDictAdapter
from .parallelize import parallelize_flux

__all__ = [
    "FluxModel",
    "FluxDataLoader",
    "flux_configs",
    "parallelize_flux",
]

# Flux DiT-style param_init constants:
# - Modulation weights: zero-init for stable training start
# - LastLayer output weights: zero-init for output stability
# - MLPEmbedder (time_in, vector_in): normal(std=0.02)
# - RMSNorm weights (QKNorm children): ones
# - Default: xavier_uniform for remaining weights
# - Default: zeros for all biases
_ZERO_LINEAR = {"weight": nn.init.zeros_, "bias": nn.init.zeros_}
_XAVIER_LINEAR = {"weight": nn.init.xavier_uniform_, "bias": nn.init.zeros_}
_NORMAL_02 = {"weight": partial(nn.init.normal_, std=0.02), "bias": nn.init.zeros_}
_NORM_INIT = {"weight": nn.init.ones_}


def expand_layer_configs(config) -> None:
    """Expand block templates into per-block configs via deepcopy.

    Mutates config in place.
    """
    config.double_blocks_expanded = [
        deepcopy(config.double_block_config) for _ in range(config.depth)
    ]
    config.single_blocks_expanded = [
        deepcopy(config.single_block_config) for _ in range(config.depth_single_blocks)
    ]


def _make_double_block_config(
    hidden_size: int,
    num_heads: int,
    mlp_ratio: float,
    qkv_bias: bool,
) -> DoubleStreamBlock.Config:
    return DoubleStreamBlock.Config(
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        img_mod=Modulation.Config(
            lin=Linear.Config(bias=True, param_init=_ZERO_LINEAR)
        ),
        txt_mod=Modulation.Config(
            lin=Linear.Config(bias=True, param_init=_ZERO_LINEAR)
        ),
        img_attn=SelfAttention.Config(
            qkv=Linear.Config(bias=qkv_bias, param_init=_XAVIER_LINEAR),
            proj=Linear.Config(bias=True, param_init=_XAVIER_LINEAR),
            norm=QKNorm.Config(
                query_norm=RMSNorm.Config(param_init=_NORM_INIT),
                key_norm=RMSNorm.Config(param_init=_NORM_INIT),
            ),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        ),
        txt_attn=SelfAttention.Config(
            qkv=Linear.Config(bias=qkv_bias, param_init=_XAVIER_LINEAR),
            proj=Linear.Config(bias=True, param_init=_XAVIER_LINEAR),
            norm=QKNorm.Config(
                query_norm=RMSNorm.Config(param_init=_NORM_INIT),
                key_norm=RMSNorm.Config(param_init=_NORM_INIT),
            ),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        ),
        img_mlp_in=Linear.Config(bias=True, param_init=_XAVIER_LINEAR),
        img_mlp_out=Linear.Config(bias=True, param_init=_XAVIER_LINEAR),
        txt_mlp_in=Linear.Config(bias=True, param_init=_XAVIER_LINEAR),
        txt_mlp_out=Linear.Config(bias=True, param_init=_XAVIER_LINEAR),
    )


def _make_single_block_config(
    hidden_size: int,
    num_heads: int,
    mlp_ratio: float,
) -> SingleStreamBlock.Config:
    return SingleStreamBlock.Config(
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        linear1=Linear.Config(bias=True, param_init=_XAVIER_LINEAR),
        linear2=Linear.Config(bias=True, param_init=_XAVIER_LINEAR),
        modulation=Modulation.Config(
            lin=Linear.Config(bias=True, param_init=_ZERO_LINEAR)
        ),
        norm=QKNorm.Config(
            query_norm=RMSNorm.Config(param_init=_NORM_INIT),
            key_norm=RMSNorm.Config(param_init=_NORM_INIT),
        ),
    )


def _flux_dev():
    hidden_size = 3072
    num_heads = 24
    mlp_ratio = 4.0
    qkv_bias = True
    vec_in_dim = 768
    return FluxModel.Config(
        in_channels=64,
        out_channels=64,
        vec_in_dim=vec_in_dim,
        context_in_dim=4096,
        hidden_size=hidden_size,
        mlp_ratio=mlp_ratio,
        num_heads=num_heads,
        depth=19,
        depth_single_blocks=38,
        axes_dim=(16, 56, 56),
        theta=10_000,
        qkv_bias=qkv_bias,
        img_in=Linear.Config(bias=True, param_init=_XAVIER_LINEAR),
        txt_in=Linear.Config(bias=True, param_init=_XAVIER_LINEAR),
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
        pe_config=EmbedND.Config(dim=128, theta=10_000, axes_dim=(16, 56, 56)),
        time_in_config=MLPEmbedder.Config(
            in_dim=256,
            hidden_dim=hidden_size,
            in_layer=Linear.Config(bias=True, param_init=_NORMAL_02),
            out_layer=Linear.Config(bias=True, param_init=_NORMAL_02),
        ),
        vector_in_config=MLPEmbedder.Config(
            in_dim=vec_in_dim,
            hidden_dim=hidden_size,
            in_layer=Linear.Config(bias=True, param_init=_NORMAL_02),
            out_layer=Linear.Config(bias=True, param_init=_NORMAL_02),
        ),
        double_block_config=_make_double_block_config(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
        ),
        single_block_config=_make_single_block_config(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        ),
        final_layer_config=LastLayer.Config(
            hidden_size=hidden_size,
            patch_size=1,
            out_channels=64,
            linear=Linear.Config(bias=True, param_init=_ZERO_LINEAR),
            adaln_linear=Linear.Config(bias=True, param_init=_ZERO_LINEAR),
        ),
    )


def _flux_schnell():
    hidden_size = 3072
    num_heads = 24
    mlp_ratio = 4.0
    qkv_bias = True
    vec_in_dim = 768
    return FluxModel.Config(
        in_channels=64,
        out_channels=64,
        vec_in_dim=vec_in_dim,
        context_in_dim=4096,
        hidden_size=hidden_size,
        mlp_ratio=mlp_ratio,
        num_heads=num_heads,
        depth=19,
        depth_single_blocks=38,
        axes_dim=(16, 56, 56),
        theta=10_000,
        qkv_bias=qkv_bias,
        img_in=Linear.Config(bias=True, param_init=_XAVIER_LINEAR),
        txt_in=Linear.Config(bias=True, param_init=_XAVIER_LINEAR),
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
        pe_config=EmbedND.Config(dim=128, theta=10_000, axes_dim=(16, 56, 56)),
        time_in_config=MLPEmbedder.Config(
            in_dim=256,
            hidden_dim=hidden_size,
            in_layer=Linear.Config(bias=True, param_init=_NORMAL_02),
            out_layer=Linear.Config(bias=True, param_init=_NORMAL_02),
        ),
        vector_in_config=MLPEmbedder.Config(
            in_dim=vec_in_dim,
            hidden_dim=hidden_size,
            in_layer=Linear.Config(bias=True, param_init=_NORMAL_02),
            out_layer=Linear.Config(bias=True, param_init=_NORMAL_02),
        ),
        double_block_config=_make_double_block_config(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
        ),
        single_block_config=_make_single_block_config(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        ),
        final_layer_config=LastLayer.Config(
            hidden_size=hidden_size,
            patch_size=1,
            out_channels=64,
            linear=Linear.Config(bias=True, param_init=_ZERO_LINEAR),
            adaln_linear=Linear.Config(bias=True, param_init=_ZERO_LINEAR),
        ),
    )


def _flux_debug():
    hidden_size = 1536
    num_heads = 12
    mlp_ratio = 4.0
    qkv_bias = True
    vec_in_dim = 768
    return FluxModel.Config(
        in_channels=64,
        out_channels=64,
        vec_in_dim=vec_in_dim,
        context_in_dim=4096,
        hidden_size=hidden_size,
        mlp_ratio=mlp_ratio,
        num_heads=num_heads,
        depth=2,
        depth_single_blocks=2,
        axes_dim=(16, 56, 56),
        theta=10_000,
        qkv_bias=qkv_bias,
        img_in=Linear.Config(bias=True, param_init=_XAVIER_LINEAR),
        txt_in=Linear.Config(bias=True, param_init=_XAVIER_LINEAR),
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
        pe_config=EmbedND.Config(dim=128, theta=10_000, axes_dim=(16, 56, 56)),
        time_in_config=MLPEmbedder.Config(
            in_dim=256,
            hidden_dim=hidden_size,
            in_layer=Linear.Config(bias=True, param_init=_NORMAL_02),
            out_layer=Linear.Config(bias=True, param_init=_NORMAL_02),
        ),
        vector_in_config=MLPEmbedder.Config(
            in_dim=vec_in_dim,
            hidden_dim=hidden_size,
            in_layer=Linear.Config(bias=True, param_init=_NORMAL_02),
            out_layer=Linear.Config(bias=True, param_init=_NORMAL_02),
        ),
        double_block_config=_make_double_block_config(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
        ),
        single_block_config=_make_single_block_config(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        ),
        final_layer_config=LastLayer.Config(
            hidden_size=hidden_size,
            patch_size=1,
            out_channels=64,
            linear=Linear.Config(bias=True, param_init=_ZERO_LINEAR),
            adaln_linear=Linear.Config(bias=True, param_init=_ZERO_LINEAR),
        ),
    )


flux_configs = {
    "flux-dev": _flux_dev,
    "flux-schnell": _flux_schnell,
    "flux-debug": _flux_debug,
}


def model_registry(flavor: str) -> ModelSpec:
    config = flux_configs[flavor]()
    expand_layer_configs(config)
    return ModelSpec(
        name="flux",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_flux,
        pipelining_fn=None,
        build_loss_fn=build_mse_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=FluxStateDictAdapter,
    )
