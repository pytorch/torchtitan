# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DeepSeek ViT's two-axis rotary positions and padded 3x3 unshuffle aligner.

T: visual patches, D: vision width, H: heads, K: head width, V: language tokens.
"""

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn.functional as F

from torchtitan.distributed.spmd_types import spmd_local_context
from torchtitan.models.base.module import Module, ModuleList
from torchtitan.models.common.feed_forward import DenseFeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.vision_encoder import (
    create_block_diagonal_mask,
    VisionTransformerBlock as BaseVisionTransformerBlock,
)


class VisionTransformerBlock(BaseVisionTransformerBlock):
    """Use the common vision residual block with DeepSeek's gated MLP."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseVisionTransformerBlock.Config):
        mlp: DenseFeedForward.Config  # pyrefly: ignore [bad-override]


def rotary_cache(grid_hw, *, head_dim: int, theta: float, device):
    caches = []
    dim = head_dim // 2
    inv_freq = 1 / theta ** (
        torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim
    )
    for height, width in grid_hw:
        hpos, wpos = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
        angles_TK = (
            torch.stack((hpos, wpos), -1).reshape(-1, 2, 1).float() * inv_freq
        ).flatten(1)
        caches.append(torch.cat((angles_TK.cos(), angles_TK.sin()), -1))
    return torch.cat(caches)


def apply_rotary(q_THK, k_THK, cache_TK):
    cos_T1K, sin_T1K = (part[:, None] for part in cache_TK.chunk(2, -1))

    def rotate(x_THK):
        first, second = x_THK.float().chunk(2, -1)
        return torch.cat(
            (first * cos_T1K - second * sin_T1K, second * cos_T1K + first * sin_T1K), -1
        ).to(x_THK.dtype)

    return rotate(q_THK), rotate(k_THK)


class VisionEncoder(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        num_heads: int
        rope_theta: float
        patch_embed: Linear.Config
        layers: list[VisionTransformerBlock.Config]
        norm: RMSNorm.Config

        def __post_init__(self):
            if self.dim % self.num_heads or self.dim // self.num_heads % 4:
                raise ValueError(
                    "Vision head width must be divisible by four for two-axis RoPE."
                )

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.patch_embed = config.patch_embed.build()
        self.layers = ModuleList([layer.build() for layer in config.layers])
        self.norm = config.norm.build()

    def forward(self, patches_TD, *, grid_hw, rope_cache=None, attention_mask=None):
        if sum(height * width for height, width in grid_hw) != patches_TD.size(0):
            raise ValueError("Vision grid sizes must match the packed patch count.")
        with spmd_local_context("dp"):
            if rope_cache is None:
                rope_cache = rotary_cache(
                    grid_hw,
                    head_dim=self.config.dim // self.config.num_heads,
                    theta=self.config.rope_theta,
                    device=patches_TD.device,
                )
            if attention_mask is None:
                lengths = torch.tensor(
                    [height * width for height, width in grid_hw],
                    device=patches_TD.device,
                )
                attention_mask = create_block_diagonal_mask(
                    lengths, patches_TD.size(0), patches_TD.device
                )
            hidden_TD = self.patch_embed(patches_TD)
            for block in self.layers:
                hidden_TD = block(
                    hidden_TD,
                    rope_cache=rope_cache,
                    rope_apply=apply_rotary,
                    attention_mask=attention_mask,
                )
            output_TD = self.norm(hidden_TD)
        if spmd.is_type_checking():
            spmd.mutate_type(output_TD, "dp", src=spmd.R, dst=spmd.V)
        return output_TD


class Aligner(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        downsample_ratio: int
        w1: Linear.Config
        w2: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.ratio = config.downsample_ratio
        self.w1, self.w2 = config.w1.build(), config.w2.build()

    def forward(self, patches_TD, *, grid_hw):
        outputs, offset = [], 0
        with spmd_local_context("dp"):
            for height, width in grid_hw:
                image_DHW = (
                    patches_TD[offset : offset + height * width]
                    .view(height, width, -1)
                    .permute(2, 0, 1)
                )
                offset += height * width
                image_DHW = F.pad(
                    image_DHW, (0, -width % self.ratio, 0, -height % self.ratio)
                )
                # unfold is channel-major, matching the checkpoint's w1 column order.
                merged_VD = (
                    F.unfold(image_DHW[None], self.ratio, stride=self.ratio)
                    .squeeze(0)
                    .transpose(0, 1)
                )
                outputs.append(self.w2(F.gelu(self.w1(merged_VD))))
            output_VD = torch.cat(outputs)
        if spmd.is_type_checking():
            spmd.mutate_type(output_VD, "dp", src=spmd.R, dst=spmd.V)
        return output_VD
