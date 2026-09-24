# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoonViT3d vision encoder used by Kimi K3.

Shape suffixes:
- M = total merged tokens
- F = merged feature dimension
- O = projected text dimension
"""

from dataclasses import dataclass, field

import torch

from torchtitan.models.common import Linear
from torchtitan.models.common.nn_modules import GELU, RMSNorm
from torchtitan.models.common.vision_encoder import VisionFlopsEstimator, VisionGrid
from torchtitan.models.flops import active_parameter_flops_per_unit
from torchtitan.models.kimi_k2_7.vision_encoder import MoonViTEncoder
from torchtitan.protocols.module import Module


class KimiK3VisionProjector(Module):
    """PatchMergerMLPV2 projector from merged vision features to text width."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        linear_1: Linear.Config
        linear_2: Linear.Config
        post_norm: RMSNorm.Config
        activation: GELU.Config = field(default_factory=GELU.Config)

    def __init__(self, config: Config):
        super().__init__()
        self.linear_1 = config.linear_1.build()
        self.linear_2 = config.linear_2.build()
        self.post_norm = config.post_norm.build()
        self.activation = config.activation.build()

    def forward(self, merged_MF: torch.Tensor) -> torch.Tensor:
        projected_MO = self.linear_2(self.activation(self.linear_1(merged_MF)))
        return self.post_norm(projected_MO)


class KimiK3VisionEncoder(MoonViTEncoder):
    @dataclass(kw_only=True, slots=True)
    class Config(MoonViTEncoder.Config):
        patch_size: int
        in_channels: int
        merge_kernel_size: tuple[int, int]  # pyrefly: ignore [bad-override]
        max_num_frames: int

        final_norm: RMSNorm.Config  # pyrefly: ignore [bad-override]
        projector: KimiK3VisionProjector.Config  # pyrefly: ignore [bad-override]

        def build_vision_flops_estimator(
            self,
            encoder: "KimiK3VisionEncoder",
        ) -> VisionFlopsEstimator:
            input_patch_flops = sum(
                active_parameter_flops_per_unit(module)
                for module in (
                    encoder.patch_embed,
                    encoder.layers,
                    encoder.final_norm,
                )
            )
            output_token_flops = sum(
                active_parameter_flops_per_unit(module)
                for module in (
                    encoder.projector.linear_1,
                    encoder.projector.linear_2,
                    encoder.projector.post_norm,
                )
            )
            attention_pair_flops = (
                self.num_layers * self.block.attn.flops_per_query_key_pair()
            )
            merge_h, merge_w = self.merge_kernel_size

            def estimate(grids: tuple[VisionGrid, ...]) -> int:
                total_flops = 0
                for temporal, grid_h, grid_w in grids:
                    num_input_patches = temporal * grid_h * grid_w
                    num_output_tokens = (grid_h // merge_h) * (grid_w // merge_w)
                    total_flops += (
                        num_input_patches * input_patch_flops
                        + num_output_tokens * output_token_flops
                        + num_input_patches**2 * attention_pair_flops
                    )
                return total_flops

            return estimate
