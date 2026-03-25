# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
        img_mod=Modulation.Config(lin=Linear.Config(bias=True)),
        txt_mod=Modulation.Config(lin=Linear.Config(bias=True)),
        img_attn=SelfAttention.Config(
            qkv=Linear.Config(bias=qkv_bias),
            proj=Linear.Config(bias=True),
            norm=QKNorm.Config(
                query_norm=RMSNorm.Config(),
                key_norm=RMSNorm.Config(),
            ),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        ),
        txt_attn=SelfAttention.Config(
            qkv=Linear.Config(bias=qkv_bias),
            proj=Linear.Config(bias=True),
            norm=QKNorm.Config(
                query_norm=RMSNorm.Config(),
                key_norm=RMSNorm.Config(),
            ),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        ),
        img_mlp_in=Linear.Config(bias=True),
        img_mlp_out=Linear.Config(bias=True),
        txt_mlp_in=Linear.Config(bias=True),
        txt_mlp_out=Linear.Config(bias=True),
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
        linear1=Linear.Config(bias=True),
        linear2=Linear.Config(bias=True),
        modulation=Modulation.Config(lin=Linear.Config(bias=True)),
        norm=QKNorm.Config(
            query_norm=RMSNorm.Config(),
            key_norm=RMSNorm.Config(),
        ),
    )


flux_configs = {
    "flux-dev": FluxModel.Config(
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
        img_in=Linear.Config(bias=True),
        txt_in=Linear.Config(bias=True),
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
            hidden_dim=3072,
            in_layer=Linear.Config(bias=True),
            out_layer=Linear.Config(bias=True),
        ),
        vector_in_config=MLPEmbedder.Config(
            in_dim=768,
            hidden_dim=3072,
            in_layer=Linear.Config(bias=True),
            out_layer=Linear.Config(bias=True),
        ),
        double_block_config=_make_double_block_config(
            hidden_size=3072,
            num_heads=24,
            mlp_ratio=4.0,
            qkv_bias=True,
        ),
        single_block_config=_make_single_block_config(
            hidden_size=3072,
            num_heads=24,
            mlp_ratio=4.0,
        ),
        final_layer_config=LastLayer.Config(
            hidden_size=3072,
            patch_size=1,
            out_channels=64,
            linear=Linear.Config(bias=True),
            adaln_linear=Linear.Config(bias=True),
        ),
    ),
    "flux-schnell": FluxModel.Config(
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
        img_in=Linear.Config(bias=True),
        txt_in=Linear.Config(bias=True),
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
            hidden_dim=3072,
            in_layer=Linear.Config(bias=True),
            out_layer=Linear.Config(bias=True),
        ),
        vector_in_config=MLPEmbedder.Config(
            in_dim=768,
            hidden_dim=3072,
            in_layer=Linear.Config(bias=True),
            out_layer=Linear.Config(bias=True),
        ),
        double_block_config=_make_double_block_config(
            hidden_size=3072,
            num_heads=24,
            mlp_ratio=4.0,
            qkv_bias=True,
        ),
        single_block_config=_make_single_block_config(
            hidden_size=3072,
            num_heads=24,
            mlp_ratio=4.0,
        ),
        final_layer_config=LastLayer.Config(
            hidden_size=3072,
            patch_size=1,
            out_channels=64,
            linear=Linear.Config(bias=True),
            adaln_linear=Linear.Config(bias=True),
        ),
    ),
    "flux-debug": FluxModel.Config(
        in_channels=64,
        out_channels=64,
        vec_in_dim=768,
        context_in_dim=4096,
        hidden_size=1536,
        mlp_ratio=4.0,
        num_heads=12,
        depth=2,
        depth_single_blocks=2,
        axes_dim=(16, 56, 56),
        theta=10_000,
        qkv_bias=True,
        img_in=Linear.Config(bias=True),
        txt_in=Linear.Config(bias=True),
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
            hidden_dim=1536,
            in_layer=Linear.Config(bias=True),
            out_layer=Linear.Config(bias=True),
        ),
        vector_in_config=MLPEmbedder.Config(
            in_dim=768,
            hidden_dim=1536,
            in_layer=Linear.Config(bias=True),
            out_layer=Linear.Config(bias=True),
        ),
        double_block_config=_make_double_block_config(
            hidden_size=1536,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
        ),
        single_block_config=_make_single_block_config(
            hidden_size=1536,
            num_heads=12,
            mlp_ratio=4.0,
        ),
        final_layer_config=LastLayer.Config(
            hidden_size=1536,
            patch_size=1,
            out_channels=64,
            linear=Linear.Config(bias=True),
            adaln_linear=Linear.Config(bias=True),
        ),
    ),
}


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="flux",
        flavor=flavor,
        model=flux_configs[flavor],
        parallelize_fn=parallelize_flux,
        pipelining_fn=None,
        build_loss_fn=build_mse_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=FluxStateDictAdapter,
    )
