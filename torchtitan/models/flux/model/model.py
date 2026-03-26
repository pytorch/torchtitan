# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch
from torch import nn, Tensor
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.models.flux.model.autoencoder import AutoEncoderParams
from torchtitan.models.flux.model.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    Modulation,
    QKNorm,
    SelfAttention,
    SingleStreamBlock,
    timestep_embedding,
)
from torchtitan.protocols import BaseModel
from torchtitan.protocols.module import ModuleList
from torchtitan.tools.logging import logger


class FluxModel(BaseModel):
    """
    Transformer model for flow matching on sequences.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        img_in: Linear.Config
        txt_in: Linear.Config
        in_channels: int = 64
        out_channels: int = 64
        vec_in_dim: int = 768
        context_in_dim: int = 512
        hidden_size: int = 3072
        mlp_ratio: float = 4.0
        num_heads: int = 24
        depth: int = 19
        depth_single_blocks: int = 38
        axes_dim: tuple = (16, 56, 56)
        theta: int = 10_000
        qkv_bias: bool = True
        autoencoder_params: AutoEncoderParams = field(default_factory=AutoEncoderParams)

        # Sub-component configs
        pe_config: EmbedND.Config = field(
            default_factory=lambda: EmbedND.Config(
                dim=128,
                theta=10_000,
                axes_dim=(16, 56, 56),
            )
        )
        time_in_config: MLPEmbedder.Config = field(
            default_factory=lambda: MLPEmbedder.Config(
                in_layer=Linear.Config(bias=True),
                out_layer=Linear.Config(bias=True),
                in_dim=256,
                hidden_dim=3072,
            )
        )
        vector_in_config: MLPEmbedder.Config = field(
            default_factory=lambda: MLPEmbedder.Config(
                in_layer=Linear.Config(bias=True),
                out_layer=Linear.Config(bias=True),
                in_dim=768,
                hidden_dim=3072,
            )
        )
        double_block_config: DoubleStreamBlock.Config = field(
            default_factory=lambda: DoubleStreamBlock.Config(
                img_mlp_in=Linear.Config(bias=True),
                img_mlp_out=Linear.Config(bias=True),
                txt_mlp_in=Linear.Config(bias=True),
                txt_mlp_out=Linear.Config(bias=True),
                img_mod=Modulation.Config(lin=Linear.Config(bias=True)),
                txt_mod=Modulation.Config(lin=Linear.Config(bias=True)),
                img_attn=SelfAttention.Config(
                    qkv=Linear.Config(bias=True),
                    proj=Linear.Config(bias=True),
                    norm=QKNorm.Config(
                        query_norm=RMSNorm.Config(),
                        key_norm=RMSNorm.Config(),
                    ),
                ),
                txt_attn=SelfAttention.Config(
                    qkv=Linear.Config(bias=True),
                    proj=Linear.Config(bias=True),
                    norm=QKNorm.Config(
                        query_norm=RMSNorm.Config(),
                        key_norm=RMSNorm.Config(),
                    ),
                ),
                hidden_size=3072,
                num_heads=24,
                mlp_ratio=4.0,
                qkv_bias=True,
            )
        )
        single_block_config: SingleStreamBlock.Config = field(
            default_factory=lambda: SingleStreamBlock.Config(
                linear1=Linear.Config(bias=True),
                linear2=Linear.Config(bias=True),
                modulation=Modulation.Config(lin=Linear.Config(bias=True)),
                norm=QKNorm.Config(
                    query_norm=RMSNorm.Config(),
                    key_norm=RMSNorm.Config(),
                ),
                hidden_size=3072,
                num_heads=24,
                mlp_ratio=4.0,
            )
        )
        final_layer_config: LastLayer.Config = field(
            default_factory=lambda: LastLayer.Config(
                linear=Linear.Config(bias=True),
                adaln_linear=Linear.Config(bias=True),
                hidden_size=3072,
                patch_size=1,
                out_channels=64,
            )
        )

        # Populated by _expand_layer_configs() in the model registry; one config per block.
        double_blocks_expanded: list = field(default_factory=list)
        single_blocks_expanded: list = field(default_factory=list)

        def update_from_config(self, *, trainer_config, **kwargs) -> None:
            pass

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            # TODO(jianiw): Add the number of flops for the autoencoder
            nparams = sum(p.numel() for p in model.parameters())
            logger.warning(
                "FLUX model haven't implement get_nparams_and_flops() function"
            )
            return nparams, 1

    def __init__(self, config: Config):
        super().__init__()

        self.config = config

        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        if config.hidden_size % config.num_heads != 0:
            raise ValueError(
                f"Hidden size {config.hidden_size} must be divisible by num_heads {config.num_heads}"
            )
        pe_dim = config.hidden_size // config.num_heads
        if sum(config.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {config.axes_dim} but expected positional dim {pe_dim}"
            )
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.pe_embedder = config.pe_config.build()
        self.img_in = config.img_in.build(
            in_features=self.in_channels, out_features=self.hidden_size
        )
        self.time_in = config.time_in_config.build()
        self.vector_in = config.vector_in_config.build()
        self.txt_in = config.txt_in.build(
            in_features=config.context_in_dim, out_features=self.hidden_size
        )

        assert config.double_blocks_expanded, (
            "config.double_blocks_expanded must be populated by _expand_layer_configs() "
            "in the model registry before build()."
        )
        self.double_blocks = ModuleList(
            [cfg.build() for cfg in config.double_blocks_expanded]
        )

        assert config.single_blocks_expanded, (
            "config.single_blocks_expanded must be populated by _expand_layer_configs() "
            "in the model registry before build()."
        )
        self.single_blocks = ModuleList(
            [cfg.build() for cfg in config.single_blocks_expanded]
        )

        self.final_layer = config.final_layer_config.build()

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img
