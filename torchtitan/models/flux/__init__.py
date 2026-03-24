# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import torch.nn as nn

from torchtitan.components.loss import build_mse_loss
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.protocols.module import set_param_init

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


def setup_flux_param_init(model: FluxModel) -> None:
    """DiT-style param_init for Flux flow matching transformer.

    Most weights use xavier_uniform (DiT convention). Exceptions:
    - Modulation + LastLayer output weights: zero-init for stable training start
    - MLPEmbedder weights (time_in, vector_in): normal(std=0.02)
    - RMSNorm weights (QKNorm children): ones
    """
    xavier = nn.init.xavier_uniform_
    zeros = nn.init.zeros_
    normal = partial(nn.init.normal_, std=0.02)
    ones = nn.init.ones_

    set_param_init(model.img_in, {"weight": xavier, "bias": zeros})
    set_param_init(model.txt_in, {"weight": xavier, "bias": zeros})
    for emb in (model.time_in, model.vector_in):
        set_param_init(
            emb.in_layer,  # pyrefly: ignore [bad-argument-type]
            {"weight": normal, "bias": zeros},
        )
        set_param_init(
            emb.out_layer,  # pyrefly: ignore [bad-argument-type]
            {"weight": normal, "bias": zeros},
        )

    for block in model.double_blocks:
        # Modulation: zero-init
        set_param_init(
            block.img_mod.lin,  # pyrefly: ignore [missing-attribute]
            {"weight": zeros, "bias": zeros},
        )
        set_param_init(
            block.txt_mod.lin,  # pyrefly: ignore [missing-attribute]
            {"weight": zeros, "bias": zeros},
        )
        # Attention + MLP: xavier
        for lin in (
            block.img_attn.qkv,  # pyrefly: ignore [missing-attribute]
            block.img_attn.proj,  # pyrefly: ignore [missing-attribute]
            block.txt_attn.qkv,  # pyrefly: ignore [missing-attribute]
            block.txt_attn.proj,  # pyrefly: ignore [missing-attribute]
        ):
            set_param_init(lin, {"weight": xavier, "bias": zeros})
        for mlp in (block.img_mlp, block.txt_mlp):
            set_param_init(
                mlp[0],  # pyrefly: ignore [bad-index, bad-argument-type]
                {"weight": xavier, "bias": zeros},
            )
            set_param_init(
                mlp[2],  # pyrefly: ignore [bad-index, bad-argument-type]
                {"weight": xavier, "bias": zeros},
            )
        # QKNorm: ones
        for attn in (block.img_attn, block.txt_attn):
            set_param_init(
                attn.norm.query_norm,  # pyrefly: ignore [missing-attribute]
                {"weight": ones},
            )
            set_param_init(
                attn.norm.key_norm,  # pyrefly: ignore [missing-attribute]
                {"weight": ones},
            )

    for block in model.single_blocks:
        set_param_init(
            block.modulation.lin,  # pyrefly: ignore [missing-attribute]
            {"weight": zeros, "bias": zeros},
        )
        set_param_init(
            block.linear1,  # pyrefly: ignore [bad-argument-type]
            {"weight": xavier, "bias": zeros},
        )
        set_param_init(
            block.linear2,  # pyrefly: ignore [bad-argument-type]
            {"weight": xavier, "bias": zeros},
        )
        set_param_init(
            block.norm.query_norm,  # pyrefly: ignore [missing-attribute]
            {"weight": ones},
        )
        set_param_init(
            block.norm.key_norm,  # pyrefly: ignore [missing-attribute]
            {"weight": ones},
        )

    # Final layer: zero-init
    set_param_init(
        model.final_layer.linear,  # pyrefly: ignore [bad-argument-type]
        {"weight": zeros, "bias": zeros},
    )
    set_param_init(
        model.final_layer.adaLN_modulation[  # pyrefly: ignore [bad-index, bad-argument-type]
            1
        ],
        {"weight": zeros, "bias": zeros},
    )


flux_configs = {
    "flux-dev": FluxModel.Config(
        param_init_fn=setup_flux_param_init,
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
        param_init_fn=setup_flux_param_init,
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
        param_init_fn=setup_flux_param_init,
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
