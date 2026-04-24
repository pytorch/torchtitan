# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
from torch import nn

from torchtitan.protocols import BaseModel
from torchtitan.tools.logging import logger

from torchtitan.models.flux2.src.flux2.model import Flux2, Flux2Params


class Flux2Model(Flux2, BaseModel):
    """
    TorchTitan wrapper for the FLUX.2 flow model.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        in_channels: int = 128
        context_in_dim: int = 15360
        hidden_size: int = 6144
        num_heads: int = 48
        depth: int = 8
        depth_single_blocks: int = 48
        axes_dim: tuple[int, int, int, int] = (32, 32, 32, 32)
        theta: int = 2000
        mlp_ratio: float = 3.0
        use_guidance_embed: bool = True

        def update_from_config(self, *, trainer_config, **kwargs) -> None:
            del trainer_config, kwargs
            return None

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            del seq_len
            nparams = sum(p.numel() for p in model.parameters())
            logger.warning(
                "Flux2Model.get_nparams_and_flops() is a placeholder; FLOPs are not computed."
            )
            return nparams, 1

        def build_params(self) -> Flux2Params:
            return Flux2Params(
                in_channels=self.in_channels,
                context_in_dim=self.context_in_dim,
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                depth=self.depth,
                depth_single_blocks=self.depth_single_blocks,
                axes_dim=list(self.axes_dim),
                theta=self.theta,
                mlp_ratio=self.mlp_ratio,
                use_guidance_embed=self.use_guidance_embed,
            )

    def __init__(self, config: Config):
        self.config = config
        params = config.build_params()
        super().__init__(params)

    def init_weights(self, *, buffer_device=None, **kwargs) -> None:
        del buffer_device, kwargs
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if module.elementwise_affine:
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
