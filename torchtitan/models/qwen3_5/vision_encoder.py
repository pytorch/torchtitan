# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass, field

import spmd_types as spmd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.experimental import local_map
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common import Linear
from torchtitan.models.common.attention import FlexAttention
from torchtitan.models.common.nn_modules import GELU, LayerNorm
from torchtitan.models.common.rope import _maybe_wrap_positions, CosSinRoPE
from torchtitan.protocols.module import Module, ModuleDict

_compiled_create_block_mask = torch.compile(create_block_mask)


def get_vision_block_mask_mod(num_patch: torch.Tensor) -> Callable:
    """Create a mask modifier for block-diagonal attention.

    Each image only attends to its own patches.

    Args:
        num_patch: (num_vision,) actual number of patches per visual item
    """

    def mask_mod(b, h, q_idx, kv_idx):
        valid_q = q_idx < num_patch[b]
        valid_kv = kv_idx < num_patch[b]
        return valid_q & valid_kv

    return mask_mod


def _compute_learned_pos_embeds(
    learned_pos_embed: torch.Tensor,
    grid_thw: torch.Tensor,
    max_num_patch: int,
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
        grid_thw: (num_vision, 3) with patch counts [t, h, w] per visual item
        max_num_patch: Maximum number of patches (for padding)
        num_grid_per_side: Side length of the square position embedding grid
        spatial_merge_size: Number of patches to merge per spatial dimension
        dim: Hidden dimension

    Returns:
        pos_embeds: (num_vision, max_num_patch, dim) interpolated position embeddings
    """
    num_vision = grid_thw.shape[0]
    dtype = learned_pos_embed.dtype
    merge_size = spatial_merge_size

    pos_embeds = learned_pos_embed.new_zeros(num_vision, max_num_patch, dim)
    if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
        pos_embeds = spmd.mutate_type(pos_embeds, "tp", src=spmd.R, dst=spmd.I)

    # Group images by (h, w) to batch compute position embeddings
    hw_to_indices: dict[tuple[int, int], list[int]] = {}
    for i in range(num_vision):
        h, w = int(grid_thw[i, 1].item()), int(grid_thw[i, 2].item())
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
        if isinstance(pos_grid, DTensor):
            pos_hw = local_map(F.interpolate, out_placements=(pos_grid.placements,),)(
                pos_grid,
                size=[h, w],  # pyrefly: ignore [unexpected-keyword]
                mode="bilinear",  # pyrefly: ignore [unexpected-keyword]
                align_corners=True,  # pyrefly: ignore [unexpected-keyword]
            )
        else:
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

        # Apply to all images with this (h, w)
        # For videos (t > 1), repeat spatial embeddings per frame;
        # temporal position encoding is handled by MRoPE in the LLM
        for i in indices:
            t = int(grid_thw[i, 0].item())
            seq_len = t * h * w
            if t > 1:
                pos_embeds[i, :seq_len] = pos_hw_block.repeat(t, 1)
            else:
                pos_embeds[i, :seq_len] = pos_hw_block

    return pos_embeds


def _compute_2d_rope_cache(
    freq_table: torch.Tensor,
    grid_thw: torch.Tensor,
    max_num_patch: int,
    spatial_merge_size: int,
    head_dim: int,
) -> torch.Tensor:
    """Compute 2D RoPE cache for vision patches.

    Builds row and col indices in block order (matching the patch layout from
    the collator), looks up separate frequency sets for each dimension, and
    concatenates them into a rope_cache for VisionAttention.

    Args:
        freq_table: (max_hw, head_dim//4) precomputed RoPE frequencies
        grid_thw: (num_vision, 3) with patch counts [t, h, w] per visual item
        max_num_patch: Maximum number of patches (for padding)
        spatial_merge_size: Number of patches to merge per spatial dimension
        head_dim: Attention head dimension

    Returns:
        rope_cache: (num_vision, max_num_patch, 1, head_dim*2) float32 for
            VisionAttention
    """
    num_vision = grid_thw.shape[0]
    device = grid_thw.device
    merge_size = spatial_merge_size

    rope_embeds = torch.zeros(
        num_vision, max_num_patch, head_dim // 2, device=device, dtype=torch.float32
    )
    if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
        rope_embeds = spmd.mutate_type(rope_embeds, "tp", src=spmd.R, dst=spmd.I)

    # Group images by (h, w) to batch compute RoPE embeddings
    hw_to_indices: dict[tuple[int, int], list[int]] = {}
    for i in range(num_vision):
        h, w = int(grid_thw[i, 1].item()), int(grid_thw[i, 2].item())
        key = (h, w)
        if key not in hw_to_indices:
            hw_to_indices[key] = []
        hw_to_indices[key].append(i)

    for (h, w), indices in hw_to_indices.items():
        # Compute RoPE position ids in block order (once per unique h, w)
        # A "block" is a merge_size x merge_size group of patches that will
        # be merged into one LLM visual token. RoPE indices must follow
        # block order to match the patch layout from image_to_patches
        merged_h, merged_w = h // merge_size, w // merge_size
        row_base = (
            torch.arange(merged_h, device=device) * merge_size
        )  # starting row of each block
        col_base = (
            torch.arange(merged_w, device=device) * merge_size
        )  # starting col of each block
        intra_row = torch.arange(merge_size, device=device)
        intra_col = torch.arange(merge_size, device=device)

        # Build row and col indices in block order
        # The 4 dimensions represent: [block_row, block_col, intra_row, intra_col]
        # e.g., for merge_size=2 on a 4x4 patch grid (2x2 blocks):
        #   row_idx = [0,0,1,1, 0,0,1,1, 2,2,3,3, 2,2,3,3]
        #   col_idx = [0,1,0,1, 2,3,2,3, 0,1,0,1, 2,3,2,3]
        row_idx = (
            (row_base[:, None, None, None] + intra_row[None, None, :, None])
            .expand(merged_h, merged_w, merge_size, merge_size)
            .reshape(-1)
        )
        col_idx = (
            (col_base[None, :, None, None] + intra_col[None, None, None, :])
            .expand(merged_h, merged_w, merge_size, merge_size)
            .reshape(-1)
        )
        if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
            row_idx = spmd.mutate_type(row_idx, "tp", src=spmd.R, dst=spmd.I)
            col_idx = spmd.mutate_type(col_idx, "tp", src=spmd.R, dst=spmd.I)

        # 2D RoPE: row and col each get separate frequency sets, concatenated
        # (not interleaved). freq_table shape: (max_hw, head_dim//4)
        rope_row = freq_table[row_idx]  # (h*w, head_dim//4)
        rope_col = freq_table[col_idx]  # (h*w, head_dim//4)
        rope_2d = torch.cat([rope_row, rope_col], dim=-1)  # (h*w, head_dim//2)

        # Apply to all images with this (h, w)
        # For videos (t > 1), repeat spatial embeddings per frame;
        # temporal position encoding is handled by MRoPE in the LLM
        for i in indices:
            t = int(grid_thw[i, 0].item())
            seq_len = t * h * w
            if t > 1:
                rope_embeds[i, :seq_len] = rope_2d.repeat(t, 1)
            else:
                rope_embeds[i, :seq_len] = rope_2d

    # Compute cos/sin in float32 for numerical precision
    rope_embeds = torch.cat((rope_embeds, rope_embeds), dim=-1)  # (N, L, head_dim)
    rope_cache = torch.cat([rope_embeds.cos(), rope_embeds.sin()], dim=-1).unsqueeze(
        2
    )  # (N, L, 1, head_dim*2)

    return rope_cache


class VisionRotaryEmbedding(Module):
    """2D Rotary Position Embedding for Vision Transformer."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        theta: float = 10000.0

    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.theta = config.theta
        inv_freq = 1.0 / (
            config.theta
            ** (torch.arange(0, config.dim, 2, dtype=torch.float) / config.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        """Re-compute inv_freq on the target device after to_empty()."""
        device = buffer_device or self.inv_freq.device
        self.inv_freq = 1.0 / (
            self.theta
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.float, device=device)
                / self.dim
            )
        )

    def forward(self, seqlen: int) -> torch.Tensor:
        """Compute rotary frequency table for positions up to seqlen."""
        seq = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        seq = _maybe_wrap_positions(seq, self.inv_freq)
        if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
            seq = spmd.mutate_type(seq, "tp", src=spmd.R, dst=spmd.I)
        return torch.outer(seq, self.inv_freq)  # pyrefly: ignore


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Merge spatial patches and project to output dimension.

        Args:
            x: (batch, seq_len, hidden_size) where seq_len is divisible by spatial_merge_size^2

        Returns:
            (batch, seq_len // spatial_merge_size^2, out_hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        x = self.norm(x)
        x = x.view(
            batch_size,
            seq_len // (self.spatial_merge_size**2),
            self.merged_hidden_size,
        )
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


class VisionAttention(Module):
    """Multi-head attention with FlexAttention for efficient batched processing."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        num_heads: int
        wq: Linear.Config
        wk: Linear.Config
        wv: Linear.Config
        proj: Linear.Config
        inner_attention: Module.Config = field(default_factory=FlexAttention.Config)

    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads

        self.wq = config.wq.build()
        self.wk = config.wk.build()
        self.wv = config.wv.build()
        self.proj = config.proj.build()
        self.flex_attention = config.inner_attention.build()

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_cache: torch.Tensor,
        attention_mask: BlockMask,
    ) -> torch.Tensor:
        bs, seqlen, _ = x.shape

        xq = self.wq(x).view(bs, seqlen, -1, self.head_dim)
        xk = self.wk(x).view(bs, seqlen, -1, self.head_dim)
        xv = self.wv(x).view(bs, seqlen, -1, self.head_dim)

        xq, xk = CosSinRoPE.apply_rotary_emb(xq, xk, rope_cache)

        output = self.flex_attention(xq, xk, xv, attention_masks=attention_mask)
        output = output.reshape(bs, seqlen, -1)
        return self.proj(output)


class VisionMLP(Module):
    """Feed-forward network with GELU activation."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        fc1: Linear.Config
        fc2: Linear.Config
        act_fn: GELU.Config = field(
            default_factory=lambda: GELU.Config(approximate="tanh")
        )

    def __init__(self, config: Config):
        super().__init__()
        self.linear_fc1 = config.fc1.build()
        self.linear_fc2 = config.fc2.build()
        self.act_fn = config.act_fn.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class VisionTransformerBlock(Module):
    """Single transformer block for vision encoder."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        norm1: LayerNorm.Config
        norm2: LayerNorm.Config
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
        x: torch.Tensor,
        *,
        rope_cache: torch.Tensor,
        attention_mask: BlockMask,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x), rope_cache=rope_cache, attention_mask=attention_mask
        )
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen35VisionEncoder(Module):
    """Qwen3.5 Vision Encoder with FlexAttention.

    Uses padded batches (N, L, D) format for efficient processing.
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
        rotary_pos_emb: VisionRotaryEmbedding.Config
        merger: PatchMerger.Config

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

        self.rotary_pos_emb = config.rotary_pos_emb.build()
        self._cached_freq_table: torch.Tensor | None = None

        self.layers = ModuleDict(
            {str(idx): config.block.build() for idx in range(config.num_layers)}
        )

        self.merger = config.merger.build()

    def compute_position_embeddings(
        self, grid_thw: torch.Tensor, max_num_patch: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute position embeddings for padded batch.

        Delegates to two standalone helpers:
        - ``_compute_learned_pos_embeds``: bilinear-interpolated learned embeddings
        - ``_compute_2d_rope_cache``: 2D RoPE cache

        Args:
            grid_thw: (num_vision, 3) with patch counts [t, h, w] per visual item
            max_num_patch: Maximum number of patches (for padding)

        Returns:
            learned_pos: (num_vision, max_num_patch, dim) learnable position embeddings
            rope_cache: (num_vision, max_num_patch, 1, head_dim*2) RoPE cache for
                VisionAttention
        """
        head_dim = self.config.dim // self.config.num_heads

        # Get RoPE freq table, reusing cache when possible
        max_hw = int(grid_thw[:, 1:].max().item())
        if self._cached_freq_table is None or self._cached_freq_table.shape[0] < max_hw:
            self._cached_freq_table = self.rotary_pos_emb(max_hw)

        learned_pos = _compute_learned_pos_embeds(
            self.pos_embed,
            grid_thw,
            max_num_patch,
            self.num_grid_per_side,
            self.spatial_merge_size,
            self.config.dim,
        )

        if isinstance(self._cached_freq_table, DTensor):
            rope_cache = local_map(
                _compute_2d_rope_cache,
                out_placements=(self._cached_freq_table.placements,),
            )(
                self._cached_freq_table,
                grid_thw,  # pyrefly: ignore [bad-argument-count]
                max_num_patch,
                self.spatial_merge_size,
                head_dim,
            )
        else:
            rope_cache = _compute_2d_rope_cache(
                self._cached_freq_table,
                grid_thw,
                max_num_patch,
                self.spatial_merge_size,
                head_dim,
            )

        return learned_pos, rope_cache

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the vision encoder.

        Processes both images and videos — each visual item is a batch of
        padded patches with a (t, h, w) grid.

        Args:
            pixel_values: Padded patches (num_vision, max_num_patch, patch_dim)
            grid_thw: Grid dimensions (num_vision, 3) for [temporal, height, width] measured in patches

        Returns:
            merged_hidden_states: (num_vision, max_merged_num_patch, out_hidden_size)
        """
        num_vision, max_num_patch, _ = pixel_values.shape

        num_patch = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).to(torch.long)

        x = self.patch_embed(pixel_values)  # (num_vision, max_num_patch, dim)
        learned_pos, rope_cache = self.compute_position_embeddings(
            grid_thw, max_num_patch
        )
        x = x + learned_pos

        mask_mod = get_vision_block_mask_mod(num_patch)
        with spmd.no_typecheck():
            attention_mask = _compiled_create_block_mask(
                mask_mod,
                num_vision,
                None,
                max_num_patch,
                max_num_patch,
                device=x.device,
            )

        for layer in self.layers.values():
            x = layer(x, rope_cache=rope_cache, attention_mask=attention_mask)

        return self.merger(x)
