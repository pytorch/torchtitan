# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoonViT3d vision encoder used by Kimi K3.

Shape suffixes:
- T = total packed patches
- D = vision hidden dimension
- C = number of complex-valued head-dimension pairs
- M = total merged tokens
- F = merged feature dimension
- O = projected text dimension
"""

from dataclasses import dataclass, field

import spmd_types as spmd
import torch

from torchtitan.models.common import Linear
from torchtitan.models.common.nn_modules import GELU, RMSNorm
from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.models.common.vision_encoder import (
    create_block_diagonal_mask,
    VisionTransformerBlock,
)
from torchtitan.models.kimi_k2_7.vision_encoder import (
    _compute_2d_rope_cache,
    _compute_learned_pos_embeds,
    _tpool_patch_merger,
    VisionRotaryEmbedding2D,
)
from torchtitan.protocols.module import Module, ModuleDict


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


class KimiK3VisionEncoder(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        num_layers: int
        patch_size: int
        in_channels: int
        merge_kernel_size: tuple[int, int]
        init_pos_emb_height: int
        init_pos_emb_width: int
        max_num_frames: int
        interpolation_mode: str
        patch_embed_proj: Linear.Config
        rotary_pos_emb: VisionRotaryEmbedding2D.Config
        block: VisionTransformerBlock.Config
        final_norm: RMSNorm.Config
        projector: KimiK3VisionProjector.Config

    def __init__(self, config: Config):
        super().__init__()
        self.merge_kernel_size = config.merge_kernel_size
        self.interpolation_mode = config.interpolation_mode
        self.patch_embed = config.patch_embed_proj.build()
        self.pos_embed = torch.nn.Parameter(
            torch.empty(
                config.init_pos_emb_height,
                config.init_pos_emb_width,
                config.dim,
            )
        )
        self.rotary_pos_emb = config.rotary_pos_emb.build()
        self.register_buffer("_cached_freq_table", None, persistent=False)
        self.layers = ModuleDict(
            {
                str(layer_idx): config.block.build()
                for layer_idx in range(config.num_layers)
            }
        )
        self.final_norm = config.final_norm.build()
        self.projector = config.projector.build()

    def _compute_position_embeddings(
        self, grids: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        max_grid_side = max(max(grid_h, grid_w) for _, grid_h, grid_w in grids)
        if (
            self._cached_freq_table is None
            or self._cached_freq_table.shape[0] < max_grid_side
        ):
            self._cached_freq_table = self.rotary_pos_emb(max_grid_side)
        learned_pos = _compute_learned_pos_embeds(
            self.pos_embed,
            grids,
            self.interpolation_mode,
        )
        rope_cache = _compute_2d_rope_cache(
            self._cached_freq_table,
            grids,
            self.rotary_pos_emb.head_dim,
        )
        return learned_pos, rope_cache

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Encode packed raster-order patches and return packed text features."""
        grids = grid_thw.tolist()

        kernel_h, kernel_w = self.merge_kernel_size
        for _, grid_h, grid_w in grids:
            if grid_h % kernel_h != 0 or grid_w % kernel_w != 0:
                raise ValueError(
                    f"Vision grid {grid_h}x{grid_w} is not divisible by "
                    f"merge kernel {self.merge_kernel_size}."
                )

        segment_lengths = grid_thw.prod(dim=-1)
        total_tokens = pixel_values.shape[0]
        expected_tokens = sum(t * h * w for t, h, w in grids)
        if total_tokens != expected_tokens:
            raise ValueError(
                f"pixel_values contains {total_tokens} patches but grid_thw "
                f"describes {expected_tokens}."
            )

        learned_pos, rope_cache = self._compute_position_embeddings(grids)
        hidden_TD = self.patch_embed(pixel_values) + learned_pos

        with spmd.no_typecheck():
            attention_mask = create_block_diagonal_mask(
                segment_lengths,
                total_tokens,
                hidden_TD.device,
            )
        for block in self.layers.values():
            hidden_TD = block(
                hidden_TD,
                rope_cache=rope_cache,
                rope_apply=ComplexRoPE.apply_rotary_emb,
                attention_mask=attention_mask,
            )
        hidden_TD = self.final_norm(hidden_TD)
        merged_MF = _tpool_patch_merger(hidden_TD, grids, self.merge_kernel_size)
        return self.projector(merged_MF)
