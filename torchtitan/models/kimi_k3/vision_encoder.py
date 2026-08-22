# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoonViT3d vision encoder used by Kimi K3.

Shape suffixes:
- N = number of visual items
- T = total packed patches
- D = vision hidden dimension
- H = number of attention heads
- K = attention head dimension
- C = number of complex-valued head-dimension pairs
- M = total merged tokens
- F = merged feature dimension
- O = projected text dimension
"""

from dataclasses import dataclass, field

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.models.common import Linear
from torchtitan.models.common.nn_modules import GELU, RMSNorm
from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.models.common.vision_encoder import (
    create_block_diagonal_mask,
    VisionAttention,
    VisionMLP,
)
from torchtitan.protocols.module import Module, ModuleDict


def _get_temporal_pos_embed(
    num_frames: int,
    embed_dim: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Return fixed 1D sinusoidal embeddings for video frame positions."""
    grid = torch.arange(num_frames, dtype=torch.float32, device=device)
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=device) / (
        embed_dim / 2.0
    )
    omega = 1.0 / 10000.0**omega
    angles = torch.outer(grid, omega)
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


def _compute_learned_pos_embeds(
    pos_embed: torch.Tensor,
    grids: list[list[int]],
    interpolation_mode: str,
    max_num_frames: int,
) -> torch.Tensor:
    """Interpolate the learned 2D table and add fixed temporal embeddings."""
    height, width, dim = pos_embed.shape
    pos_grid = pos_embed.permute(2, 0, 1).unsqueeze(0).float()

    cached_spatial: dict[tuple[int, int], torch.Tensor] = {}
    positions = []
    for num_frames, grid_h, grid_w in grids:
        if num_frames > max_num_frames:
            raise ValueError(
                f"Vision grid has {num_frames} frames, exceeding "
                f"max_num_frames={max_num_frames}."
            )
        spatial = cached_spatial.get((grid_h, grid_w))
        if spatial is None:
            if (grid_h, grid_w) == (height, width):
                spatial = pos_embed.flatten(end_dim=1)
            else:
                spatial = (
                    F.interpolate(
                        pos_grid,
                        size=(grid_h, grid_w),
                        mode=interpolation_mode,
                    )
                    .squeeze(0)
                    .permute(1, 2, 0)
                    .reshape(grid_h * grid_w, dim)
                    .to(pos_embed.dtype)
                )
            cached_spatial[(grid_h, grid_w)] = spatial

        if num_frames == 1:
            item_pos = spatial
        else:
            temporal = _get_temporal_pos_embed(num_frames, dim, device=pos_embed.device)
            item_pos = spatial.unsqueeze(0) + temporal.unsqueeze(1).to(spatial.dtype)
            item_pos = item_pos.reshape(num_frames * grid_h * grid_w, dim)
        positions.append(item_pos)

    return torch.cat(positions)


def _compute_2d_rope_cache(
    freq_table: torch.Tensor,
    grids: list[list[int]],
    head_dim: int,
) -> torch.Tensor:
    """Build the real-valued 2D RoPE cache in raster patch order."""
    cached_spatial: dict[tuple[int, int], torch.Tensor] = {}
    item_angles = []
    for num_frames, grid_h, grid_w in grids:
        spatial = cached_spatial.get((grid_h, grid_w))
        if spatial is None:
            flat = torch.arange(grid_h * grid_w, device=freq_table.device)
            x_angles = freq_table[flat % grid_w]
            y_angles = freq_table[flat // grid_w]
            spatial = torch.stack((x_angles, y_angles), dim=-1).reshape(
                grid_h * grid_w, head_dim // 2
            )
            cached_spatial[(grid_h, grid_w)] = spatial
        item_angles.append(spatial.repeat(num_frames, 1))

    angles = torch.cat(item_angles)
    # ComplexRoPE.apply_rotary_emb multiplies in complex64; float() only widens
    # the container, so cos/sin keep whatever precision angles were computed in.
    cos_sin = torch.stack((angles.cos(), angles.sin()), dim=-1).float()
    return torch.view_as_complex(cos_sin).unsqueeze(1)


def _temporal_pool_and_merge(
    hidden_TD: torch.Tensor,
    grids: list[list[int]],
    merge_kernel_size: tuple[int, int],
) -> torch.Tensor:
    """Temporally pool and concatenate neighboring spatial patch features."""
    dim = hidden_TD.shape[-1]
    kernel_h, kernel_w = merge_kernel_size
    merged_dim = kernel_h * kernel_w * dim

    merged_items = []
    offset = 0
    for num_frames, grid_h, grid_w in grids:
        num_patches = num_frames * grid_h * grid_w
        item = hidden_TD[offset : offset + num_patches]
        offset += num_patches
        merged_h = grid_h // kernel_h
        merged_w = grid_w // kernel_w
        item = item.view(
            num_frames,
            merged_h,
            kernel_h,
            merged_w,
            kernel_w,
            dim,
        )
        item = item.permute(0, 1, 3, 2, 4, 5).mean(dim=0)
        merged_items.append(item.reshape(merged_h * merged_w, merged_dim))

    return torch.cat(merged_items)


class VisionRotaryEmbedding2D(Module):
    """Per-axis frequency table for MoonViT's interleaved 2D RoPE."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        head_dim: int
        theta: float = 10000.0

    def __init__(self, config: Config):
        super().__init__()
        if config.head_dim % 4 != 0:
            raise ValueError(
                "Vision 2D RoPE head_dim must be divisible by 4, "
                f"got {config.head_dim}."
            )
        self.head_dim = config.head_dim
        self.theta = config.theta
        self.register_buffer("inv_freq", self._compute_inv_freq(), persistent=False)

    def _compute_inv_freq(self, *, device: torch.device | None = None) -> torch.Tensor:
        return 1.0 / (
            self.theta
            ** (
                torch.arange(
                    0,
                    self.head_dim,
                    4,
                    dtype=torch.float32,
                    device=device,
                )
                / self.head_dim
            )
        )

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        device = buffer_device or self.inv_freq.device
        self.inv_freq = self._compute_inv_freq(device=device)

    def forward(self, seqlen: int) -> torch.Tensor:
        positions = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        return torch.outer(positions, self.inv_freq)


class KimiK3VisionBlock(Module):
    """MoonViT pre-norm attention and MLP block."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        norm1: RMSNorm.Config
        norm2: RMSNorm.Config
        attn: VisionAttention.Config
        mlp: VisionMLP.Config

    def __init__(self, config: Config):
        super().__init__()
        self.norm1 = config.norm1.build()
        self.norm2 = config.norm2.build()
        self.attn = config.attn.build()
        self.mlp = config.mlp.build()

    def forward(
        self,
        x_TD: torch.Tensor,
        *,
        rope_cache: torch.Tensor,
        attention_mask: BlockMask,
    ) -> torch.Tensor:
        x_TD = x_TD + self.attn(
            self.norm1(x_TD),
            rope_cache=rope_cache,
            rope_apply=ComplexRoPE.apply_rotary_emb,
            attention_mask=attention_mask,
        )
        return x_TD + self.mlp(self.norm2(x_TD))


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
        block: KimiK3VisionBlock.Config
        final_norm: RMSNorm.Config
        projector: KimiK3VisionProjector.Config

    def __init__(self, config: Config):
        super().__init__()
        self.merge_kernel_size = config.merge_kernel_size
        self.max_num_frames = config.max_num_frames
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
            self.max_num_frames,
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
                attention_mask=attention_mask,
            )
        hidden_TD = self.final_norm(hidden_TD)
        merged_MF = _temporal_pool_and_merge(hidden_TD, grids, self.merge_kernel_size)
        return self.projector(merged_MF)
