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


def _fill_modulation_fields(mod, *, dim: int, double: bool) -> None:
    """Fill expanded fields on a Modulation.Config."""
    mod.dim = dim
    mod.double = double
    multiplier = 6 if double else 3
    mod.lin.in_features = dim
    mod.lin.out_features = multiplier * dim


def _fill_self_attn_fields(attn, *, dim: int) -> None:
    """Fill expanded fields on a flux SelfAttention.Config."""
    attn.dim = dim
    head_dim = dim // attn.num_heads
    attn.qkv.in_features = dim
    attn.qkv.out_features = dim * 3
    attn.proj.in_features = dim
    attn.proj.out_features = dim
    attn.norm.dim = head_dim
    attn.norm.query_norm.normalized_shape = head_dim
    attn.norm.key_norm.normalized_shape = head_dim


def _fill_double_block_fields(cfg) -> None:
    """Fill expanded fields on a DoubleStreamBlock.Config."""
    hs = cfg.hidden_size
    mlp_hidden_dim = int(hs * cfg.mlp_ratio)
    _fill_modulation_fields(cfg.img_mod, dim=hs, double=True)
    _fill_self_attn_fields(cfg.img_attn, dim=hs)
    cfg.img_mlp_in.in_features = hs
    cfg.img_mlp_in.out_features = mlp_hidden_dim
    cfg.img_mlp_out.in_features = mlp_hidden_dim
    cfg.img_mlp_out.out_features = hs
    _fill_modulation_fields(cfg.txt_mod, dim=hs, double=True)
    _fill_self_attn_fields(cfg.txt_attn, dim=hs)
    cfg.txt_mlp_in.in_features = hs
    cfg.txt_mlp_in.out_features = mlp_hidden_dim
    cfg.txt_mlp_out.in_features = mlp_hidden_dim
    cfg.txt_mlp_out.out_features = hs


def _fill_single_block_fields(cfg) -> None:
    """Fill expanded fields on a SingleStreamBlock.Config."""
    hs = cfg.hidden_size
    head_dim = hs // cfg.num_heads
    mlp_hidden_dim = int(hs * cfg.mlp_ratio)
    cfg.linear1.in_features = hs
    cfg.linear1.out_features = hs * 3 + mlp_hidden_dim
    cfg.linear2.in_features = hs + mlp_hidden_dim
    cfg.linear2.out_features = hs
    cfg.norm.dim = head_dim
    cfg.norm.query_norm.normalized_shape = head_dim
    cfg.norm.key_norm.normalized_shape = head_dim
    _fill_modulation_fields(cfg.modulation, dim=hs, double=False)


def _fill_mlp_embedder_fields(cfg) -> None:
    """Fill Linear sub-config fields on a MLPEmbedder.Config."""
    cfg.in_layer.in_features = cfg.in_dim
    cfg.in_layer.out_features = cfg.hidden_dim
    cfg.out_layer.in_features = cfg.hidden_dim
    cfg.out_layer.out_features = cfg.hidden_dim


def _fill_last_layer_fields(cfg) -> None:
    """Fill Linear sub-config fields on a LastLayer.Config."""
    hs = cfg.hidden_size
    cfg.linear.in_features = hs
    cfg.linear.out_features = cfg.patch_size * cfg.patch_size * cfg.out_channels
    cfg.adaln_linear.in_features = hs
    cfg.adaln_linear.out_features = 2 * hs


def expand_layer_configs(config) -> None:
    """Expand block templates into per-block configs via deepcopy.

    Also fills all field(init=False) fields on the config tree.
    Mutates config in place.
    """
    hs = config.hidden_size

    # Top-level Linears
    config.img_in.in_features = config.in_channels
    config.img_in.out_features = hs
    config.txt_in.in_features = config.context_in_dim
    config.txt_in.out_features = hs

    # MLPEmbedders
    _fill_mlp_embedder_fields(config.time_in_config)
    _fill_mlp_embedder_fields(config.vector_in_config)

    # LastLayer
    _fill_last_layer_fields(config.final_layer_config)

    # Expand double and single blocks
    config.double_blocks_expanded = []
    for _ in range(config.depth):
        cfg = deepcopy(config.double_block_config)
        _fill_double_block_fields(cfg)
        config.double_blocks_expanded.append(cfg)

    config.single_blocks_expanded = []
    for _ in range(config.depth_single_blocks):
        cfg = deepcopy(config.single_block_config)
        _fill_single_block_fields(cfg)
        config.single_blocks_expanded.append(cfg)


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


def _flux_dev() -> FluxModel.Config:
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


def _flux_schnell() -> FluxModel.Config:
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


def _flux_debug() -> FluxModel.Config:
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
