# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Shape suffixes:
#   N = packed patches, D = encoder width, H = heads, K = head dimension,
#   R = rotated head channels, M = merged tokens, E = text model width.

from dataclasses import dataclass, field

import spmd_types as spmd
import torch
import torch_remat as remat

from torchtitan.models.base.module import Module, ModuleDict
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import GELU, LayerNorm
from torchtitan.models.common.rope import CosSinRoPE
from torchtitan.models.common.vision_encoder import (
    create_block_diagonal_mask,
    VisionTransformerBlock,
)

__all__ = ["MiniMaxM3VisionEncoder", "MiniMaxM3VisionProjector"]


def _vision_rope_cache(
    grids: list[list[int]],
    *,
    spatial_merge_size: int,
    head_dim: int,
    theta: float,
    device: torch.device,
) -> torch.Tensor:
    axis_dim = 2 * ((head_dim // 3) // 2)
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, axis_dim, 2, device=device, dtype=torch.float) / axis_dim)
    )
    window = (spatial_merge_size, spatial_merge_size)
    positions = []
    for frames, rows, cols in grids:
        row_ids, col_ids = torch.meshgrid(
            torch.arange(rows, device=device),
            torch.arange(cols, device=device),
            indexing="ij",
        )
        block_shape = (rows // window[0], window[0], cols // window[1], window[1])
        row_ids = row_ids.reshape(block_shape).transpose(1, 2).flatten()
        col_ids = col_ids.reshape(block_shape).transpose(1, 2).flatten()
        frame_ids = torch.arange(frames, device=device).repeat_interleave(rows * cols)
        positions.append(
            torch.stack([frame_ids, row_ids.repeat(frames), col_ids.repeat(frames)], -1)
        )
    freqs_NR = (torch.cat(positions)[..., None].float() * inv_freq).flatten(1)
    if spmd.is_type_checking():
        # Built from each rank's own grids: rank-local on DP, invariant on TP/CP.
        freqs_NR = spmd.mutate_type(
            freqs_NR,
            src={"dp": spmd.R, "tp": spmd.R, "cp": spmd.R},
            dst={"dp": spmd.V, "tp": spmd.I, "cp": spmd.I},
        )
    freqs_NR = torch.cat([freqs_NR, freqs_NR], dim=-1)
    return torch.cat([freqs_NR.cos(), freqs_NR.sin()], dim=-1).unsqueeze(1)


def _apply_vision_rope(
    q_NHK: torch.Tensor, k_NHK: torch.Tensor, rope_cache: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary_dim = rope_cache.shape[-1] // 2
    q_NHR, k_NHR = CosSinRoPE.apply_rotary_emb(
        q_NHK[..., :rotary_dim], k_NHK[..., :rotary_dim], rope_cache
    )
    return (
        torch.cat([q_NHR, q_NHK[..., rotary_dim:]], dim=-1),
        torch.cat([k_NHR, k_NHK[..., rotary_dim:]], dim=-1),
    )


class MiniMaxM3VisionProjector(Module):
    """Per-patch MLP into the text width, then an MLP over each merge window."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        spatial_merge_size: int = 2
        linear_1: Linear.Config
        linear_2: Linear.Config
        merge_linear_1: Linear.Config
        merge_linear_2: Linear.Config
        act_fn: GELU.Config = field(default_factory=GELU.Config)

    def __init__(self, config: Config):
        super().__init__()
        self.merge_unit = config.spatial_merge_size**2
        self.linear_1 = config.linear_1.build()
        self.linear_2 = config.linear_2.build()
        self.merge_linear_1 = config.merge_linear_1.build()
        self.merge_linear_2 = config.merge_linear_2.build()
        self.act_fn = config.act_fn.build()

    def forward(self, x_ND: torch.Tensor) -> torch.Tensor:
        hidden_NE = self.linear_1(x_ND)
        remat.recompute_needs_tensor(hidden_NE)
        hidden_NE = self.linear_2(self.act_fn(hidden_NE))
        remat.recompute_needs_tensor(hidden_NE)
        # The collator orders each merge window's patches contiguously.
        hidden_ME = hidden_NE.reshape(hidden_NE.shape[0] // self.merge_unit, -1)
        hidden_ME = self.merge_linear_1(hidden_ME)
        remat.recompute_needs_tensor(hidden_ME)
        return self.merge_linear_2(self.act_fn(hidden_ME))


class MiniMaxM3VisionEncoder(Module):
    """CLIP-style ViT with 3D RoPE; each image, and each video segment, attends within itself.

    A video is split into segments of at most ``max_temporal_groups_per_segment``
    temporal patch groups.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        num_layers: int
        num_heads: int
        patch_size: int = 14
        temporal_patch_size: int = 2
        in_channels: int = 3
        spatial_merge_size: int = 2
        max_temporal_groups_per_segment: int = 4
        rope_theta: float = 10000.0
        patch_embed_proj: Linear.Config
        pre_norm: LayerNorm.Config
        block: VisionTransformerBlock.Config
        projector: MiniMaxM3VisionProjector.Config

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = config.spatial_merge_size**2
        self.patch_embed = config.patch_embed_proj.build()
        self.pre_norm = config.pre_norm.build()
        self.layers = ModuleDict(
            {str(idx): config.block.build() for idx in range(config.num_layers)}
        )
        self.projector = config.projector.build()

    def _segment_lengths(self, grids: list[list[int]]) -> list[int]:
        max_groups = self.config.max_temporal_groups_per_segment
        lengths = []
        for frames, rows, cols in grids:
            for start in range(0, frames, max_groups):
                lengths.append(min(max_groups, frames - start) * rows * cols)
        return lengths

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Encode packed patches ``(N, patch_dim)`` into merged tokens ``(M, E)``."""
        grids = grid_thw.tolist()
        num_patches = pixel_values.shape[0]
        expected_patches = sum(t * h * w for t, h, w in grids)
        if num_patches != expected_patches:
            raise ValueError(
                f"pixel_values contains {num_patches} patches but grid_thw "
                f"describes {expected_patches}."
            )

        x_ND = self.pre_norm(self.patch_embed(pixel_values))
        rope_cache = _vision_rope_cache(
            grids,
            spatial_merge_size=self.spatial_merge_size,
            head_dim=self.config.dim // self.config.num_heads,
            theta=self.config.rope_theta,
            device=x_ND.device,
        )
        # BlockMask creation and use in FlexInnerAttention are blackboxed from
        # typechecking.
        with spmd.no_typecheck():
            attention_mask = create_block_diagonal_mask(
                torch.tensor(self._segment_lengths(grids), device=x_ND.device),
                num_patches,
                x_ND.device,
            )
        for layer in self.layers.values():
            x_ND = layer(
                x_ND,
                rope_cache=rope_cache,
                rope_apply=_apply_vision_rope,
                attention_mask=attention_mask,
            )
        return self.projector(x_ND)
