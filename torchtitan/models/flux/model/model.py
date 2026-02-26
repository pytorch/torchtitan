# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch
from torch import nn, Tensor
from torchtitan.models.flux.model.autoencoder import AutoEncoderParams
from torchtitan.models.flux.model.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)
from torchtitan.protocols import BaseModel


class FluxModel(BaseModel):
    """
    Transformer model for flow matching on sequences.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
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
                in_dim=256,
                hidden_dim=3072,
            )
        )
        vector_in_config: MLPEmbedder.Config = field(
            default_factory=lambda: MLPEmbedder.Config(
                in_dim=768,
                hidden_dim=3072,
            )
        )
        double_block_config: DoubleStreamBlock.Config = field(
            default_factory=lambda: DoubleStreamBlock.Config(
                hidden_size=3072,
                num_heads=24,
                mlp_ratio=4.0,
                qkv_bias=True,
            )
        )
        single_block_config: SingleStreamBlock.Config = field(
            default_factory=lambda: SingleStreamBlock.Config(
                hidden_size=3072,
                num_heads=24,
                mlp_ratio=4.0,
            )
        )
        final_layer_config: LastLayer.Config = field(
            default_factory=lambda: LastLayer.Config(
                hidden_size=3072,
                patch_size=1,
                out_channels=64,
            )
        )

        # Derived sequence lengths, set by update_from_config from trainer
        # config. Used for FLOPs estimation in get_nparams_and_flops.
        seq_len_img: int = field(init=False, repr=False, default=0)
        seq_len_txt: int = field(init=False, repr=False, default=0)

        def update_from_config(self, *, trainer_config, **kwargs) -> None:
            # Compute image token count: autoencoder downscales the image,
            # then pack_latents tiles the latent into 2×2 patches.
            # pyrefly: ignore [missing-attribute]
            img_size = trainer_config.dataloader.img_size
            ae_downscale = 2 ** (len(self.autoencoder_params.ch_mult) - 1)
            LATENT_PATCH_SIZE = 2  # from pack_latents in utils.py
            latent_side = img_size // ae_downscale // LATENT_PATCH_SIZE
            self.seq_len_img = latent_side * latent_side

            # pyrefly: ignore [missing-attribute]
            self.seq_len_txt = trainer_config.encoder.max_t5_encoding_len

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            nparams = sum(p.numel() for p in model.parameters())

            assert self.seq_len_img > 0 and self.seq_len_txt > 0, (
                "update_from_config must be called before get_nparams_and_flops"
            )
            total_seq_len = self.seq_len_img + self.seq_len_txt

            # Base: 6 FLOPs per parameter per token (fwd + bwd for linear
            # layers). This assumes every token passes through every parameter.
            num_flops_per_token = 6 * nparams

            # Correction for DoubleStreamBlocks: img and txt tokens pass
            # through separate but symmetric linear layers, so 6*nparams
            # double-counts one side's parameters. Subtract one side per block.
            #
            # Per-side weight params (bias/norm terms are negligible):
            #   img_attn.qkv:  h * 3h           = 3h²
            #   img_attn.proj: h * h             =  h²
            #   img_mlp:       2 * h * h*r       = 2rh²
            #   img_mod.lin:   h * 6h (double)   = 6h²
            #   Total: h² * (10 + 2r)
            db_h = self.double_block_config.hidden_size
            db_r = self.double_block_config.mlp_ratio
            nparams_db_one_side = int(db_h * db_h * (10 + 2 * db_r))
            num_flops_per_token -= 6 * nparams_db_one_side * self.depth

            # Add non-parameterized self-attention FLOPs (QK^T and attn*V).
            # Per PaLM convention: 6 * hidden_size * seq_len per token per
            # layer (covers 2 matmuls × fwd+bwd × multiply-add).
            sb_h = self.single_block_config.hidden_size
            num_flops_per_token += (
                6 * sb_h * total_seq_len * self.depth_single_blocks
                + 6 * db_h * total_seq_len * self.depth
            )

            return nparams, num_flops_per_token

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
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = config.time_in_config.build()
        self.vector_in = config.vector_in_config.build()
        self.txt_in = nn.Linear(config.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [config.double_block_config.build() for _ in range(config.depth)]
        )

        self.single_blocks = nn.ModuleList(
            [
                config.single_block_config.build()
                for _ in range(config.depth_single_blocks)
            ]
        )

        self.final_layer = config.final_layer_config.build()

    def init_weights(self, *, buffer_device=None, **kwargs):
        # Adapted from DiT weight initialization: https://github.com/facebookresearch/DiT/blob/main/models.py#L189
        # initialize Linear Layers: img_in, txt_in
        nn.init.xavier_uniform_(self.img_in.weight)
        nn.init.constant_(self.img_in.bias, 0)
        nn.init.xavier_uniform_(self.txt_in.weight)
        nn.init.constant_(self.txt_in.bias, 0)

        # Initialize time_in, vector_in (MLPEmbedder)
        self.time_in.init_weights(init_std=0.02)
        self.vector_in.init_weights(init_std=0.02)

        # Initialize transformer blocks:
        for block in self.single_blocks:
            # pyrefly: ignore [not-callable]
            block.init_weights()
        for block in self.double_blocks:
            # pyrefly: ignore [not-callable]
            block.init_weights()

        # Zero-out output layers:
        self.final_layer.init_weights()

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
