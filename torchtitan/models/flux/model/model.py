# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import spmd_types as spmd
import torch
from torch import nn, Tensor
from torchtitan.models.common.nn_modules import Linear
from torchtitan.models.flux.model.autoencoder import AutoEncoder
from torchtitan.models.flux.model.hf_embedder import FluxEmbedder
from torchtitan.protocols import BaseModel
from torchtitan.protocols.module import ModuleList

from torchtitan.models.flux.model.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    local_split_text_image,
    timestep_embedding,
)


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
        autoencoder: AutoEncoder.Config = field(default_factory=AutoEncoder.Config)

        # Text encoder configs, set by the model registry. The trainer can
        # override version and random_init when it builds the encoders.
        clip_encoder: FluxEmbedder.Config
        t5_encoder: FluxEmbedder.Config

        # Sub-component configs (all required — set by the model registry)
        pe_config: EmbedND.Config
        time_in_config: MLPEmbedder.Config
        vector_in_config: MLPEmbedder.Config
        final_layer_config: LastLayer.Config
        double_blocks: list[DoubleStreamBlock.Config]
        single_blocks: list[SingleStreamBlock.Config]

        def update_from_config(self, *, config, **kwargs) -> None:
            from torchtitan.models.flux.sharding import set_flux_sharding_config

            set_flux_sharding_config(self)

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            nparams = sum(p.numel() for p in model.parameters())

            # Base: 6 FLOPs per parameter per token (fwd + bwd for linear
            # layers). This assumes every token passes through every parameter.
            num_flops_per_token = 6 * nparams

            # Correction 1: DoubleStreamBlocks have symmetric img/txt streams;
            # each token only passes through one side. Subtract one side's
            # per-token linear params per block (excluding modulation, which
            # is per-sample and handled separately below).
            #
            # Per-side per-token weight params:
            #   attn.qkv:  h * 3h       = 3h²
            #   attn.proj: h * h         =  h²
            #   mlp:       2 * h * h*r   = 2rh²
            #   Total: h² * (4 + 2r)
            db_h = self.double_blocks[0].hidden_size
            db_r = self.double_blocks[0].mlp_ratio
            nparams_db_one_side_per_token = int(db_h * db_h * (4 + 2 * db_r))
            num_flops_per_token -= 6 * nparams_db_one_side_per_token * self.depth

            # Correction 2: Modulation layers operate on vec (per-sample
            # conditioning from CLIP + timestep), not per-token. The 6*nparams
            # base counts them as per-token; replace with amortized per-token
            # cost (once per sample / seq_len tokens).
            #
            # Per-sample modulation weight params:
            #   DoubleStreamBlock: img_mod(6h²) + txt_mod(6h²) = 12h² per block
            #   SingleStreamBlock: modulation(3h²) per block
            #   LastLayer: adaLN_modulation(2h²)
            sb_h = self.single_blocks[0].hidden_size
            fl_h = self.final_layer_config.hidden_size
            nparams_mod_per_sample = (
                12 * db_h * db_h * self.depth
                + 3 * sb_h * sb_h * self.depth_single_blocks
                + 2 * fl_h * fl_h
            )
            num_flops_per_token -= 6 * nparams_mod_per_sample * (seq_len - 1) // seq_len

            # Add non-parameterized self-attention FLOPs (QK^T and attn*V).
            # Per PaLM convention: 6 * hidden_size * seq_len per token per
            # layer (covers 2 matmuls × fwd+bwd × multiply-add).
            num_flops_per_token += (
                6 * sb_h * seq_len * self.depth_single_blocks
                + 6 * db_h * seq_len * self.depth
            )

            return nparams, num_flops_per_token

    def __init__(self, config: Config):
        super().__init__()

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
        self.img_in = config.img_in.build()
        self.time_in = config.time_in_config.build()
        self.vector_in = config.vector_in_config.build()
        self.txt_in = config.txt_in.build()

        self.double_blocks = ModuleList([cfg.build() for cfg in config.double_blocks])

        self.single_blocks = ModuleList([cfg.build() for cfg in config.single_blocks])

        self.final_layer = config.final_layer_config.build()

    @staticmethod
    @spmd.local_map(
        in_types=(
            spmd.PartitionSpec("dp", "cp", None),
            spmd.PartitionSpec("dp", "cp", None),
        ),
        out_types=spmd.PartitionSpec("dp", "cp", None),
    )
    def local_concat_text_image(text: Tensor, image: Tensor) -> Tensor:
        return torch.cat((text, image), dim=1)

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

        ids = self.local_concat_text_image(txt_ids, img_ids)
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        img = self.local_concat_text_image(txt, img)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        _, img = local_split_text_image(img, txt.shape[1])

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img
