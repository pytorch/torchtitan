# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3.5 vision encoder.

Shape suffixes:
- T = packed patch tokens
- D = vision hidden dimension
- Dh = attention head dimension
- M = packed merged tokens
- K = merged feature dimension
- C = two spatial coordinates (row, column)
- P = cached coordinate positions
- F = rotary frequencies per coordinate (head_dim / 4)
- R = combined spatial frequencies (head_dim / 2)
- H = patch merge block rows
- W = patch merge block columns
- S = patches per side of a merge block
"""

from dataclasses import dataclass, field

import spmd_types as spmd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.models.base.model import FlopsEstimator
from torchtitan.models.base.module import Module, ModuleDict
from torchtitan.models.common import Linear

from torchtitan.models.common.flops import active_parameter_flops_per_token
from torchtitan.models.common.nn_modules import GELU, LayerNorm
from torchtitan.models.common.rope import CosSinRoPE
from torchtitan.models.common.vision_encoder import (
    create_block_diagonal_mask,
    get_packed_vision_grids,
    VisionTransformerBlock,
)


def _compute_learned_pos_embeds(
    learned_pos_embed: torch.Tensor,
    grids: list[list[int]],
    num_grid_per_side: int,
    spatial_merge_size: int,
    dim: int,
) -> torch.Tensor:
    """Compute bilinear-interpolated learned position embeddings.

    Reshapes the learnable position embedding table into a 2D grid, interpolates
    it to each image's (h, w) resolution, and permutes to block order matching
    the patch layout from the collator.

    Args:
        learned_pos_embed: (num_position_embeddings, dim) learnable position embeddings
        grids: per-item ``[t, h, w]`` patch counts as host ints.
        num_grid_per_side: Side length of the square position embedding grid
        spatial_merge_size: Number of patches to merge per spatial dimension
        dim: Hidden dimension

    Returns:
        pos_embeds: (total_num_patches, dim) packed position embeddings.
    """
    dtype = learned_pos_embed.dtype
    merge_size = spatial_merge_size

    pos_embeds: dict[int, torch.Tensor] = {}

    # Group images by (h, w) to batch compute position embeddings
    hw_to_indices: dict[tuple[int, int], list[int]] = {}
    for i, (_, h, w) in enumerate(grids):
        key = (h, w)
        if key not in hw_to_indices:
            hw_to_indices[key] = []
        hw_to_indices[key].append(i)

    # Reshape pos_embed to 2D grid for F.interpolate:
    # (num_position_embeddings, dim) → (1, dim, grid_side, grid_side)
    pos_grid = (
        learned_pos_embed.reshape(num_grid_per_side, num_grid_per_side, -1)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
    )

    for (h, w), indices in hw_to_indices.items():
        pos_hw = F.interpolate(
            pos_grid,
            size=[h, w],
            mode="bilinear",
            align_corners=True,
        )

        # (1, dim, h, w) → (h*w, dim)
        pos_hw = pos_hw.squeeze(0).permute(1, 2, 0).reshape(-1, dim).to(dtype)

        # Permute learned pos_hw from raster order to block order
        # to match the patch sequence produced by image_to_patches
        pos_hw_block = (
            pos_hw.view(h // merge_size, merge_size, w // merge_size, merge_size, -1)
            .permute(0, 2, 1, 3, 4)
            .flatten(0, 3)
        )  # (h*w, dim)

        # Apply to all visual items with this (h, w).
        # For videos (t > 1), repeat spatial embeddings per frame;
        # temporal position encoding is handled by MRoPE in the LLM
        for i in indices:
            t = grids[i][0]
            if t > 1:
                pos_embeds[i] = pos_hw_block.repeat(t, 1)
            else:
                pos_embeds[i] = pos_hw_block

    packed_pos_embeds = torch.cat([pos_embeds[i] for i in range(len(grids))], dim=0)
    if spmd.is_type_checking():
        packed_pos_embeds = spmd.mutate_type(
            packed_pos_embeds,
            src={"dp": spmd.R, "tp": spmd.I},
            dst={"dp": spmd.V, "tp": spmd.I},
        )
    return packed_pos_embeds


def _compute_2d_positions(
    grids: list[list[int]], spatial_merge_size: int, device: torch.device
) -> torch.Tensor:
    """Compute 2D RoPE coordinates for vision patches.

    Builds row and col indices in block order (matching the patch layout from
    the collator).

    Args:
        grids: per-item ``[t, h, w]`` patch counts as host ints.
        spatial_merge_size: Number of patches to merge per spatial dimension
        device: Device for the position indices.

    Returns:
        ``(total_num_patches, 2)`` row/column coordinates.
    """
    positions: dict[int, torch.Tensor] = {}
    # Group images by (h, w) to batch compute RoPE positions
    hw_to_indices: dict[tuple[int, int], list[int]] = {}
    for i, (_, h, w) in enumerate(grids):
        hw_to_indices.setdefault((h, w), []).append(i)

    merge_size = spatial_merge_size
    for (h, w), indices in hw_to_indices.items():
        # Compute RoPE position ids in block order (once per unique h, w)
        # A "block" is a merge_size x merge_size group of patches that will
        # be merged into one LLM visual token. RoPE indices must follow
        # block order to match the patch layout from image_to_patches
        merged_h, merged_w = h // merge_size, w // merge_size
        # starting row of each block
        row_base_H = torch.arange(merged_h, device=device) * merge_size
        # starting col of each block
        col_base_W = torch.arange(merged_w, device=device) * merge_size
        intra_row_S = torch.arange(merge_size, device=device)
        intra_col_S = torch.arange(merge_size, device=device)
        # Build row and col indices in block order
        # The 4 dimensions represent: [block_row, block_col, intra_row, intra_col]
        # e.g., for merge_size=2 on a 4x4 patch grid (2x2 blocks):
        #   row_idx = [0,0,1,1, 0,0,1,1, 2,2,3,3, 2,2,3,3]
        #   col_idx = [0,1,0,1, 2,3,2,3, 0,1,0,1, 2,3,2,3]
        row_T = (
            (row_base_H[:, None, None, None] + intra_row_S[None, None, :, None])
            .expand(merged_h, merged_w, merge_size, merge_size)
            .reshape(-1)
        )
        col_T = (
            (col_base_W[None, :, None, None] + intra_col_S[None, None, None, :])
            .expand(merged_h, merged_w, merge_size, merge_size)
            .reshape(-1)
        )
        if spmd.is_type_checking():
            row_T = spmd.mutate_type(row_T, "tp", src=spmd.R, dst=spmd.I)
            row_T = spmd.mutate_type(row_T, "cp", src=spmd.R, dst=spmd.I)
            col_T = spmd.mutate_type(col_T, "tp", src=spmd.R, dst=spmd.I)
            col_T = spmd.mutate_type(col_T, "cp", src=spmd.R, dst=spmd.I)
        spatial_TC = torch.stack((row_T, col_T), dim=-1)
        # Apply to all visual items with this (h, w).
        # For videos (t > 1), repeat spatial positions per frame;
        # temporal position encoding is handled by MRoPE in the LLM
        for i in indices:
            positions[i] = spatial_TC.repeat(grids[i][0], 1)

    positions_TC = torch.cat([positions[i] for i in range(len(grids))], dim=0)
    if spmd.is_type_checking():
        positions_TC = spmd.mutate_type(positions_TC, "dp", src=spmd.R, dst=spmd.V)
    return positions_TC


class VisionRotaryEmbedding(CosSinRoPE):
    """2D Rotary Position Embedding for Vision Transformer.

    Coordinates are zero-based ``(row, column)`` pairs in token order.
    ``max_context_length`` is the exclusive bound on either coordinate.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(CosSinRoPE.Config):
        pass

    def __init__(self, config: Config):
        if config.scaling is not None:
            raise ValueError(
                "VisionRotaryEmbedding does not support frequency scaling."
            )
        super().__init__(config)

    def _precompute_cache(self) -> torch.Tensor:
        """Compute the per-coordinate inverse frequencies.

        Returns:
            Cache of shape ``(dim / 4,)``.
        """
        # Meta construction needs only the table shape and dtype.
        if torch.get_default_device().type == "meta":
            return torch.empty(
                (self.dim // 4,),
                dtype=torch.float32,
                device="meta",
            )
        half_dim = self.dim // 2
        return 1.0 / (
            self.theta ** (torch.arange(0, half_dim, 2, dtype=torch.float32) / half_dim)
        )

    def prepare_positions(self, positions: torch.Tensor) -> torch.Tensor:
        """Compute 2D RoPE cache for vision patches.

        Builds separate frequency sets for each dimension and concatenates them
        into a rope_cache for VisionAttention.

        Args:
            positions: ``(total_num_patches, 2)`` row/column coordinates.

        Returns:
            rope_cache: (total_num_patches, 1, head_dim*2) float32 for
                VisionAttention.
        """
        # 2D RoPE: row and col each get separate frequency sets, concatenated
        # (not interleaved).
        angles_TR = (positions.unsqueeze(-1) * self.cache).flatten(1, 2)
        angles_TDh = torch.cat((angles_TR, angles_TR), dim=-1)
        # Compute cos/sin in float32 for numerical precision
        return torch.cat((angles_TDh.cos(), angles_TDh.sin()), dim=-1).unsqueeze(1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None,
        positions: torch.Tensor,
        *,
        inverse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Rotate with positions prepared once outside the vision block."""
        return self.apply_rotary_emb(query, key, positions, inverse=inverse)


class PatchMerger(Module):
    """Merge spatial patches to reduce sequence length.

    Applies LayerNorm before spatial reshape, then projects through a
    two-layer MLP (fc1 → GELU → fc2).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        spatial_merge_size: int
        merged_hidden_size: int
        norm: LayerNorm.Config
        fc1: Linear.Config
        act_fn: GELU.Config = field(
            default_factory=lambda: GELU.Config(approximate="tanh")
        )
        fc2: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        self.merged_hidden_size = config.merged_hidden_size

        self.norm = config.norm.build()
        self.linear_fc1 = config.fc1.build()
        self.act_fn = config.act_fn.build()
        self.linear_fc2 = config.fc2.build()

    def forward(self, x_TD: torch.Tensor) -> torch.Tensor:
        """Merge spatial patches and project to output dimension.

        Args:
            x_TD: Packed patch features. Each visual item's segment length is
                divisible by ``spatial_merge_size**2``.

        Returns:
            Packed merged patch features.
        """
        x_TD = self.norm(x_TD)
        x_MK = x_TD.view(-1, self.merged_hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x_MK)))


class Qwen35VisionEncoder(Module):
    """Qwen3.5 Vision Encoder with FlexInnerAttention.

    Processes visual items as one packed patch sequence.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        """Configuration for Qwen3.5 Vision Encoder (ViT)."""

        dim: int = 1280
        num_layers: int = 32
        num_heads: int = 16

        patch_size: int = 16
        temporal_patch_size: int = 2
        in_channels: int = 3
        spatial_merge_size: int = 2

        num_position_embeddings: int = 4096

        # Sub-module configs
        patch_embed_proj: Linear.Config
        block: VisionTransformerBlock.Config
        merger: PatchMerger.Config

        def build_flops_estimator(
            self,
            encoder: "Qwen35VisionEncoder",
        ) -> FlopsEstimator:
            input_patch_flops = sum(
                active_parameter_flops_per_token(module)
                for module in (
                    encoder.patch_embed,
                    encoder.layers,
                    encoder.merger.norm,
                )
            )
            output_token_flops = sum(
                active_parameter_flops_per_token(module)
                for module in (
                    encoder.merger.linear_fc1,
                    encoder.merger.linear_fc2,
                )
            )
            attention_pair_flops = (
                self.num_layers * self.block.attn.flops_per_query_key_pair()
            )
            merge_size = self.spatial_merge_size

            def estimate(input_dict) -> int:
                total_flops = 0
                for temporal, grid_h, grid_w in get_packed_vision_grids(input_dict):
                    num_spatial_patches = grid_h * grid_w
                    num_input_patches = temporal * num_spatial_patches
                    num_output_tokens = (
                        temporal * (grid_h // merge_size) * (grid_w // merge_size)
                    )
                    total_flops += (
                        num_input_patches * input_patch_flops
                        + num_output_tokens * output_token_flops
                        + temporal * num_spatial_patches**2 * attention_pair_flops
                    )
                return total_flops

            return estimate

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = config.spatial_merge_size**2

        # Patches are pre-extracted by the collator, so Linear replaces Conv3d (equivalent at full-patch kernel size).
        self.patch_embed = config.patch_embed_proj.build()

        # nn.Parameter (not nn.Embedding) because we interpolate the weight directly
        self.num_position_embeddings = config.num_position_embeddings
        self.pos_embed = nn.Parameter(
            torch.empty(config.num_position_embeddings, config.dim)
        )
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        self.layers = ModuleDict(
            {str(idx): config.block.build() for idx in range(config.num_layers)}
        )

        self.merger = config.merger.build()

    def compute_position_embeddings(
        self, grids: list[list[int]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute packed learned position embeddings and RoPE positions.

        Delegates to two standalone helpers:
        - ``_compute_learned_pos_embeds``: bilinear-interpolated learned embeddings
        - ``_compute_2d_positions``: 2D RoPE coordinates

        Args:
            grids: per-item ``[t, h, w]`` patch counts as host ints.

        Returns:
            learned_pos: ``(total_num_patches, dim)`` learned positions.
            positions: ``(total_num_patches, 2)`` row/column coordinates.
        """
        rope_config = self.config.block.attn.rope
        max_hw = max(max(h, w) for _, h, w in grids)
        if (
            isinstance(rope_config, CosSinRoPE.Config)
            and max_hw > rope_config.max_context_length
        ):
            raise ValueError(
                f"Vision grid side {max_hw} exceeds the RoPE coordinate bound "
                f"{rope_config.max_context_length}; increase block.attn.rope.max_context_length."
            )
        learned_pos_TD = _compute_learned_pos_embeds(
            self.pos_embed,
            grids,
            self.num_grid_per_side,
            self.spatial_merge_size,
            self.config.dim,
        )
        positions_TC = _compute_2d_positions(
            grids, self.spatial_merge_size, self.pos_embed.device
        )
        return learned_pos_TD, positions_TC

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the vision encoder.

        Processes both images and videos. Each visual item has a ``(t, h, w)``
        patch grid, and all valid patches are packed into one sequence.

        Args:
            pixel_values: Packed patches ``(total_num_patches, patch_dim)``.
            grid_thw: Grid dimensions ``(num_vision, 3)`` for
                ``[temporal, height, width]``, measured in patches.

        Returns:
            merged_hidden_states: Packed merged patch features with shape
                ``(total_merged_num_patches, out_hidden_size)``.
        """
        # One host sync for the whole forward: read the (N, 3) grid to CPU ints
        # so every per-item loop below builds shapes without a device sync.
        grids = grid_thw.tolist()  # [[t, h, w], ...]
        segment_lengths = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        )
        total_tokens = pixel_values.shape[0]
        expected_tokens = sum(t * h * w for t, h, w in grids)
        if total_tokens != expected_tokens:
            raise ValueError(
                f"pixel_values contains {total_tokens} patches but grid_thw "
                f"describes {expected_tokens}."
            )

        x = self.patch_embed(pixel_values)
        learned_pos_TD, positions_TC = self.compute_position_embeddings(grids)
        x = x + learned_pos_TD

        # BlockMask creation and use in FlexInnerAttention are blackboxed from
        # typechecking.
        with spmd.no_typecheck():
            attention_mask = create_block_diagonal_mask(
                segment_lengths,
                total_tokens,
                x.device,
            )

        # Every layer is built from the same block config.
        rope_positions = (
            self.layers["0"].attn.rope.prepare_positions(positions_TC)
            if len(self.layers) > 0
            else positions_TC
        )
        for layer in self.layers.values():
            x = layer(
                x,
                positions=rope_positions,
                attention_mask=attention_mask,
            )

        return self.merger(x)
