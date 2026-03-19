# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_mse_loss
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.param_init import (
    init_by_regex,
    init_normal,
    init_ones,
    init_xavier_uniform,
    init_zeros,
)
from torchtitan.protocols.model_spec import ModelSpec

from .flux_datasets import FluxDataLoader
from .model.autoencoder import AutoEncoderParams
from .model.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
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


def _flux_param_init():
    """DiT-style param_init for Flux flow matching transformer.

    Most weights use xavier_uniform (DiT convention). Exceptions:
    - Modulation + LastLayer output weights: zero-init for stable training start
    - MLPEmbedder weights (time_in, vector_in): normal(std=0.02)
    - RMSNorm weights (QKNorm children): ones
    """
    return init_by_regex(
        {
            # Modulation layers: zero-init weights for stable training start
            r".*mod(?:ulation)?\.lin\.weight": init_zeros(),
            # LastLayer: zero-init weights for output stability
            r"final_layer\.linear\.weight": init_zeros(),
            r"final_layer\.adaLN_modulation\.1\.weight": init_zeros(),
            # MLPEmbedder (time_in, vector_in): normal init for weights
            r"(?:time_in|vector_in)\.(?:in|out)_layer\.weight": init_normal(std=0.02),
            # RMSNorm weights (QKNorm children)
            r".*norm.*\.weight": init_ones(),
            # Default: xavier_uniform for remaining weights (DiT-style)
            r".*\.weight": init_xavier_uniform(),
            # Default: zeros for all biases
            r".*\.bias": init_zeros(),
        }
    )


flux_configs = {
    "flux-dev": FluxModel.Config(
        param_init=_flux_param_init(),
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
        double_block_config=DoubleStreamBlock.Config(
            hidden_size=3072,
            num_heads=24,
            mlp_ratio=4.0,
            qkv_bias=True,
            img_mlp_in=Linear.Config(bias=True),
            img_mlp_out=Linear.Config(bias=True),
            txt_mlp_in=Linear.Config(bias=True),
            txt_mlp_out=Linear.Config(bias=True),
        ),
        single_block_config=SingleStreamBlock.Config(
            hidden_size=3072,
            num_heads=24,
            mlp_ratio=4.0,
            linear1=Linear.Config(bias=True),
            linear2=Linear.Config(bias=True),
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
        param_init=_flux_param_init(),
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
        double_block_config=DoubleStreamBlock.Config(
            hidden_size=3072,
            num_heads=24,
            mlp_ratio=4.0,
            qkv_bias=True,
            img_mlp_in=Linear.Config(bias=True),
            img_mlp_out=Linear.Config(bias=True),
            txt_mlp_in=Linear.Config(bias=True),
            txt_mlp_out=Linear.Config(bias=True),
        ),
        single_block_config=SingleStreamBlock.Config(
            hidden_size=3072,
            num_heads=24,
            mlp_ratio=4.0,
            linear1=Linear.Config(bias=True),
            linear2=Linear.Config(bias=True),
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
        param_init=_flux_param_init(),
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
        double_block_config=DoubleStreamBlock.Config(
            hidden_size=1536,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            img_mlp_in=Linear.Config(bias=True),
            img_mlp_out=Linear.Config(bias=True),
            txt_mlp_in=Linear.Config(bias=True),
            txt_mlp_out=Linear.Config(bias=True),
        ),
        single_block_config=SingleStreamBlock.Config(
            hidden_size=1536,
            num_heads=12,
            mlp_ratio=4.0,
            linear1=Linear.Config(bias=True),
            linear2=Linear.Config(bias=True),
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
