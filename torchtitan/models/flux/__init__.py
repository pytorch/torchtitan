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

_ZERO_LINEAR = {"weight": nn.init.zeros_, "bias": nn.init.zeros_}
_XAVIER_LINEAR = {"weight": nn.init.xavier_uniform_, "bias": nn.init.zeros_}
_NORMAL_02 = {"weight": partial(nn.init.normal_, std=0.02), "bias": nn.init.zeros_}
_NORM_INIT = {"weight": nn.init.ones_}


def _make_double_block_config(
    hidden_size: int,
    num_heads: int,
    mlp_ratio: float,
    qkv_bias: bool,
) -> DoubleStreamBlock.Config:
    hs = hidden_size
    head_dim = hs // num_heads
    mlp_hidden_dim = int(hs * mlp_ratio)
    return DoubleStreamBlock.Config(
        hidden_size=hs,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        img_mod=Modulation.Config(
            double=True,
            lin=Linear.Config(
                in_features=hs,
                out_features=6 * hs,
                bias=True,
                param_init=_ZERO_LINEAR,
            ),
        ),
        txt_mod=Modulation.Config(
            double=True,
            lin=Linear.Config(
                in_features=hs,
                out_features=6 * hs,
                bias=True,
                param_init=_ZERO_LINEAR,
            ),
        ),
        img_attn=SelfAttention.Config(
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv=Linear.Config(
                in_features=hs,
                out_features=hs * 3,
                bias=qkv_bias,
                param_init=_XAVIER_LINEAR,
            ),
            proj=Linear.Config(
                in_features=hs,
                out_features=hs,
                bias=True,
                param_init=_XAVIER_LINEAR,
            ),
            norm=QKNorm.Config(
                query_norm=RMSNorm.Config(
                    normalized_shape=head_dim,
                    param_init=_NORM_INIT,
                ),
                key_norm=RMSNorm.Config(
                    normalized_shape=head_dim,
                    param_init=_NORM_INIT,
                ),
            ),
        ),
        txt_attn=SelfAttention.Config(
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv=Linear.Config(
                in_features=hs,
                out_features=hs * 3,
                bias=qkv_bias,
                param_init=_XAVIER_LINEAR,
            ),
            proj=Linear.Config(
                in_features=hs,
                out_features=hs,
                bias=True,
                param_init=_XAVIER_LINEAR,
            ),
            norm=QKNorm.Config(
                query_norm=RMSNorm.Config(
                    normalized_shape=head_dim,
                    param_init=_NORM_INIT,
                ),
                key_norm=RMSNorm.Config(
                    normalized_shape=head_dim,
                    param_init=_NORM_INIT,
                ),
            ),
        ),
        img_mlp_in=Linear.Config(
            in_features=hs,
            out_features=mlp_hidden_dim,
            bias=True,
            param_init=_XAVIER_LINEAR,
        ),
        img_mlp_out=Linear.Config(
            in_features=mlp_hidden_dim,
            out_features=hs,
            bias=True,
            param_init=_XAVIER_LINEAR,
        ),
        txt_mlp_in=Linear.Config(
            in_features=hs,
            out_features=mlp_hidden_dim,
            bias=True,
            param_init=_XAVIER_LINEAR,
        ),
        txt_mlp_out=Linear.Config(
            in_features=mlp_hidden_dim,
            out_features=hs,
            bias=True,
            param_init=_XAVIER_LINEAR,
        ),
    )


def _make_single_block_config(
    hidden_size: int,
    num_heads: int,
    mlp_ratio: float,
) -> SingleStreamBlock.Config:
    hs = hidden_size
    head_dim = hs // num_heads
    mlp_hidden_dim = int(hs * mlp_ratio)
    return SingleStreamBlock.Config(
        hidden_size=hs,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        linear1=Linear.Config(
            in_features=hs,
            out_features=hs * 3 + mlp_hidden_dim,
            bias=True,
            param_init=_XAVIER_LINEAR,
        ),
        linear2=Linear.Config(
            in_features=hs + mlp_hidden_dim,
            out_features=hs,
            bias=True,
            param_init=_XAVIER_LINEAR,
        ),
        modulation=Modulation.Config(
            double=False,
            lin=Linear.Config(
                in_features=hs,
                out_features=3 * hs,
                bias=True,
                param_init=_ZERO_LINEAR,
            ),
        ),
        norm=QKNorm.Config(
            query_norm=RMSNorm.Config(
                normalized_shape=head_dim,
                param_init=_NORM_INIT,
            ),
            key_norm=RMSNorm.Config(
                normalized_shape=head_dim,
                param_init=_NORM_INIT,
            ),
        ),
    )


def _flux_dev() -> FluxModel.Config:
    hidden_size = 3072
    num_heads = 24
    mlp_ratio = 4.0
    qkv_bias = True
    vec_in_dim = 768
    in_channels = 64
    context_in_dim = 4096
    depth = 19
    depth_single_blocks = 38
    double_tmpl = _make_double_block_config(
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
    )
    single_tmpl = _make_single_block_config(
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
    )
    return FluxModel.Config(
        in_channels=in_channels,
        out_channels=64,
        vec_in_dim=vec_in_dim,
        context_in_dim=context_in_dim,
        hidden_size=hidden_size,
        mlp_ratio=mlp_ratio,
        num_heads=num_heads,
        depth=depth,
        depth_single_blocks=depth_single_blocks,
        axes_dim=(16, 56, 56),
        theta=10_000,
        qkv_bias=qkv_bias,
        img_in=Linear.Config(
            in_features=in_channels,
            out_features=hidden_size,
            bias=True,
            param_init=_XAVIER_LINEAR,
        ),
        txt_in=Linear.Config(
            in_features=context_in_dim,
            out_features=hidden_size,
            bias=True,
            param_init=_XAVIER_LINEAR,
        ),
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
            in_layer=Linear.Config(
                in_features=256,
                out_features=hidden_size,
                bias=True,
                param_init=_NORMAL_02,
            ),
            out_layer=Linear.Config(
                in_features=hidden_size,
                out_features=hidden_size,
                bias=True,
                param_init=_NORMAL_02,
            ),
        ),
        vector_in_config=MLPEmbedder.Config(
            in_dim=vec_in_dim,
            hidden_dim=hidden_size,
            in_layer=Linear.Config(
                in_features=vec_in_dim,
                out_features=hidden_size,
                bias=True,
                param_init=_NORMAL_02,
            ),
            out_layer=Linear.Config(
                in_features=hidden_size,
                out_features=hidden_size,
                bias=True,
                param_init=_NORMAL_02,
            ),
        ),
        double_blocks=[deepcopy(double_tmpl) for _ in range(depth)],
        single_blocks=[deepcopy(single_tmpl) for _ in range(depth_single_blocks)],
        final_layer_config=LastLayer.Config(
            hidden_size=hidden_size,
            patch_size=1,
            out_channels=64,
            linear=Linear.Config(
                in_features=hidden_size,
                out_features=1 * 1 * 64,
                bias=True,
                param_init=_ZERO_LINEAR,
            ),
            adaln_linear=Linear.Config(
                in_features=hidden_size,
                out_features=2 * hidden_size,
                bias=True,
                param_init=_ZERO_LINEAR,
            ),
        ),
    )


def _flux_schnell() -> FluxModel.Config:
    hidden_size = 3072
    num_heads = 24
    mlp_ratio = 4.0
    qkv_bias = True
    vec_in_dim = 768
    in_channels = 64
    context_in_dim = 4096
    depth = 19
    depth_single_blocks = 38
    double_tmpl = _make_double_block_config(
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
    )
    single_tmpl = _make_single_block_config(
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
    )
    return FluxModel.Config(
        in_channels=in_channels,
        out_channels=64,
        vec_in_dim=vec_in_dim,
        context_in_dim=context_in_dim,
        hidden_size=hidden_size,
        mlp_ratio=mlp_ratio,
        num_heads=num_heads,
        depth=depth,
        depth_single_blocks=depth_single_blocks,
        axes_dim=(16, 56, 56),
        theta=10_000,
        qkv_bias=qkv_bias,
        img_in=Linear.Config(
            in_features=in_channels,
            out_features=hidden_size,
            bias=True,
            param_init=_XAVIER_LINEAR,
        ),
        txt_in=Linear.Config(
            in_features=context_in_dim,
            out_features=hidden_size,
            bias=True,
            param_init=_XAVIER_LINEAR,
        ),
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
            in_layer=Linear.Config(
                in_features=256,
                out_features=hidden_size,
                bias=True,
                param_init=_NORMAL_02,
            ),
            out_layer=Linear.Config(
                in_features=hidden_size,
                out_features=hidden_size,
                bias=True,
                param_init=_NORMAL_02,
            ),
        ),
        vector_in_config=MLPEmbedder.Config(
            in_dim=vec_in_dim,
            hidden_dim=hidden_size,
            in_layer=Linear.Config(
                in_features=vec_in_dim,
                out_features=hidden_size,
                bias=True,
                param_init=_NORMAL_02,
            ),
            out_layer=Linear.Config(
                in_features=hidden_size,
                out_features=hidden_size,
                bias=True,
                param_init=_NORMAL_02,
            ),
        ),
        double_blocks=[deepcopy(double_tmpl) for _ in range(depth)],
        single_blocks=[deepcopy(single_tmpl) for _ in range(depth_single_blocks)],
        final_layer_config=LastLayer.Config(
            hidden_size=hidden_size,
            patch_size=1,
            out_channels=64,
            linear=Linear.Config(
                in_features=hidden_size,
                out_features=1 * 1 * 64,
                bias=True,
                param_init=_ZERO_LINEAR,
            ),
            adaln_linear=Linear.Config(
                in_features=hidden_size,
                out_features=2 * hidden_size,
                bias=True,
                param_init=_ZERO_LINEAR,
            ),
        ),
    )


def _flux_debug() -> FluxModel.Config:
    hidden_size = 1536
    num_heads = 12
    mlp_ratio = 4.0
    qkv_bias = True
    vec_in_dim = 768
    in_channels = 64
    context_in_dim = 4096
    depth = 2
    depth_single_blocks = 2
    double_tmpl = _make_double_block_config(
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
    )
    single_tmpl = _make_single_block_config(
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
    )
    return FluxModel.Config(
        in_channels=in_channels,
        out_channels=64,
        vec_in_dim=vec_in_dim,
        context_in_dim=context_in_dim,
        hidden_size=hidden_size,
        mlp_ratio=mlp_ratio,
        num_heads=num_heads,
        depth=depth,
        depth_single_blocks=depth_single_blocks,
        axes_dim=(16, 56, 56),
        theta=10_000,
        qkv_bias=qkv_bias,
        img_in=Linear.Config(
            in_features=in_channels,
            out_features=hidden_size,
            bias=True,
            param_init=_XAVIER_LINEAR,
        ),
        txt_in=Linear.Config(
            in_features=context_in_dim,
            out_features=hidden_size,
            bias=True,
            param_init=_XAVIER_LINEAR,
        ),
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
            in_layer=Linear.Config(
                in_features=256,
                out_features=hidden_size,
                bias=True,
                param_init=_NORMAL_02,
            ),
            out_layer=Linear.Config(
                in_features=hidden_size,
                out_features=hidden_size,
                bias=True,
                param_init=_NORMAL_02,
            ),
        ),
        vector_in_config=MLPEmbedder.Config(
            in_dim=vec_in_dim,
            hidden_dim=hidden_size,
            in_layer=Linear.Config(
                in_features=vec_in_dim,
                out_features=hidden_size,
                bias=True,
                param_init=_NORMAL_02,
            ),
            out_layer=Linear.Config(
                in_features=hidden_size,
                out_features=hidden_size,
                bias=True,
                param_init=_NORMAL_02,
            ),
        ),
        double_blocks=[deepcopy(double_tmpl) for _ in range(depth)],
        single_blocks=[deepcopy(single_tmpl) for _ in range(depth_single_blocks)],
        final_layer_config=LastLayer.Config(
            hidden_size=hidden_size,
            patch_size=1,
            out_channels=64,
            linear=Linear.Config(
                in_features=hidden_size,
                out_features=1 * 1 * 64,
                bias=True,
                param_init=_ZERO_LINEAR,
            ),
            adaln_linear=Linear.Config(
                in_features=hidden_size,
                out_features=2 * hidden_size,
                bias=True,
                param_init=_ZERO_LINEAR,
            ),
        ),
    )


flux_configs = {
    "flux-dev": _flux_dev,
    "flux-schnell": _flux_schnell,
    "flux-debug": _flux_debug,
}


def model_registry(flavor: str) -> ModelSpec:
    config = flux_configs[flavor]()
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
