# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoonViT3d vision encoder used by Kimi K3.

Vision attention is an eager PyTorch loop over each visual item, which
preserves the block-diagonal attention semantics of the HuggingFace
implementation without requiring FlashAttention or a device-specific kernel.

Shape suffixes:
- N = number of visual items
- P = maximum patches per item (padded)
- D = vision hidden dimension
- H = number of attention heads
- K = attention head dimension
- M = maximum merged tokens per item (padded)
"""

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from torchtitan.models.common import Linear
from torchtitan.models.common.nn_modules import GELU, RMSNorm
from torchtitan.models.common.vision_encoder import VisionMLP
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


def _pad_sequence(x: torch.Tensor, target_length: int) -> torch.Tensor:
    """Pad the leading sequence dimension without modifying ``x`` in place."""
    padding_length = target_length - x.shape[0]
    if padding_length < 0:
        raise ValueError(
            f"Cannot pad a sequence of length {x.shape[0]} to {target_length}."
        )
    if padding_length == 0:
        return x
    padding = x.new_zeros(padding_length, *x.shape[1:])
    return torch.cat((x, padding), dim=0)


def _compute_learned_pos_embeds(
    pos_embed: torch.Tensor,
    grids: list[list[int]],
    max_num_patches: int,
    interpolation_mode: str,
    max_num_frames: int,
) -> torch.Tensor:
    """Interpolate the learned 2D table and add fixed temporal embeddings."""
    height, width, dim = pos_embed.shape
    pos_grid = pos_embed.permute(2, 0, 1).unsqueeze(0).float()

    cached_spatial: dict[tuple[int, int], torch.Tensor] = {}
    padded_positions = []
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
        padded_positions.append(_pad_sequence(item_pos, max_num_patches))

    return torch.stack(padded_positions)


def _compute_2d_rope_cache(
    freq_table: torch.Tensor,
    grids: list[list[int]],
    max_num_patches: int,
    head_dim: int,
) -> torch.Tensor:
    """Build the real-valued 2D RoPE cache in raster patch order."""
    cached_spatial: dict[tuple[int, int], torch.Tensor] = {}
    padded_angles = []
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
        item_angles = spatial.repeat(num_frames, 1)
        padded_angles.append(_pad_sequence(item_angles, max_num_patches))

    angles = torch.stack(padded_angles)
    return torch.stack((angles.cos(), angles.sin()), dim=-1).unsqueeze(2)


def _apply_2d_rope(
    q_NPHK: torch.Tensor,
    k_NPHK: torch.Tensor,
    rope_cache_NP1C2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply 2D RoPE using the real form of complex multiplication."""

    cos_NP1C = rope_cache_NP1C2[..., 0]
    sin_NP1C = rope_cache_NP1C2[..., 1]

    def rotate(x_NPHK: torch.Tensor) -> torch.Tensor:
        x_NPHC2 = x_NPHK.float().reshape(*x_NPHK.shape[:-1], -1, 2)
        real_NPHC = x_NPHC2[..., 0] * cos_NP1C - x_NPHC2[..., 1] * sin_NP1C
        imag_NPHC = x_NPHC2[..., 0] * sin_NP1C + x_NPHC2[..., 1] * cos_NP1C
        return torch.stack((real_NPHC, imag_NPHC), dim=-1).flatten(-2)

    return rotate(q_NPHK).to(q_NPHK.dtype), rotate(k_NPHK).to(k_NPHK.dtype)


def _temporal_pool_and_merge(
    hidden_NPD: torch.Tensor,
    grids: list[list[int]],
    merge_kernel_size: tuple[int, int],
) -> torch.Tensor:
    """Temporally pool and concatenate neighboring spatial patch features."""
    _, _, dim = hidden_NPD.shape
    kernel_h, kernel_w = merge_kernel_size
    merged_dim = kernel_h * kernel_w * dim
    max_merged = max(
        (grid_h // kernel_h) * (grid_w // kernel_w) for _, grid_h, grid_w in grids
    )

    padded_items = []
    for item_idx, (num_frames, grid_h, grid_w) in enumerate(grids):
        merged_h = grid_h // kernel_h
        merged_w = grid_w // kernel_w
        item = hidden_NPD[item_idx, : num_frames * grid_h * grid_w].view(
            num_frames,
            merged_h,
            kernel_h,
            merged_w,
            kernel_w,
            dim,
        )
        item = item.permute(0, 1, 3, 2, 4, 5).mean(dim=0)
        item = item.reshape(merged_h * merged_w, merged_dim)
        padded_items.append(_pad_sequence(item, max_merged))

    return torch.stack(padded_items)


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


class KimiK3VisionAttention(Module):
    """Eager, block-diagonal MoonViT attention reference."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        qkv_dim: int
        num_heads: int
        wq: Linear.Config
        wk: Linear.Config
        wv: Linear.Config
        proj: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        if config.qkv_dim % config.num_heads != 0:
            raise ValueError(
                f"qkv_dim ({config.qkv_dim}) must be divisible by "
                f"num_heads ({config.num_heads})."
            )
        self.num_heads = config.num_heads
        self.head_dim = config.qkv_dim // config.num_heads
        self.scale = self.head_dim**-0.5
        self.wq = config.wq.build()
        self.wk = config.wk.build()
        self.wv = config.wv.build()
        self.proj = config.proj.build()

    def forward(
        self,
        x_NPD: torch.Tensor,
        *,
        rope_cache: torch.Tensor,
        num_patches: list[int],
    ) -> torch.Tensor:
        num_items, max_num_patches, _ = x_NPD.shape
        q_NPHK = self.wq(x_NPD).view(
            num_items, max_num_patches, self.num_heads, self.head_dim
        )
        k_NPHK = self.wk(x_NPD).view(
            num_items, max_num_patches, self.num_heads, self.head_dim
        )
        v_NPHK = self.wv(x_NPD).view(
            num_items, max_num_patches, self.num_heads, self.head_dim
        )
        q_NPHK, k_NPHK = _apply_2d_rope(q_NPHK, k_NPHK, rope_cache)

        padded_outputs = []
        for item_idx, item_length in enumerate(num_patches):
            q_HPK = q_NPHK[item_idx, :item_length].transpose(0, 1)
            k_HPK = k_NPHK[item_idx, :item_length].transpose(0, 1)
            v_HPK = v_NPHK[item_idx, :item_length].transpose(0, 1)
            scores_HPP = torch.matmul(q_HPK, k_HPK.transpose(-2, -1))
            scores_HPP = scores_HPP * self.scale
            probs_HPP = torch.softmax(scores_HPP, dim=-1, dtype=torch.float32).to(
                q_HPK.dtype
            )
            output_PHK = torch.matmul(probs_HPP, v_HPK).transpose(0, 1)
            padded_outputs.append(_pad_sequence(output_PHK, max_num_patches))

        output_NPHK = torch.stack(padded_outputs)
        return self.proj(output_NPHK.flatten(start_dim=-2))


class KimiK3VisionBlock(Module):
    """MoonViT pre-norm attention and MLP block."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        norm1: RMSNorm.Config
        norm2: RMSNorm.Config
        attn: KimiK3VisionAttention.Config
        mlp: VisionMLP.Config

    def __init__(self, config: Config):
        super().__init__()
        self.norm1 = config.norm1.build()
        self.norm2 = config.norm2.build()
        self.attn = config.attn.build()
        self.mlp = config.mlp.build()

    def forward(
        self,
        x_NPD: torch.Tensor,
        *,
        rope_cache: torch.Tensor,
        num_patches: list[int],
    ) -> torch.Tensor:
        x_NPD = x_NPD + self.attn(
            self.norm1(x_NPD),
            rope_cache=rope_cache,
            num_patches=num_patches,
        )
        return x_NPD + self.mlp(self.norm2(x_NPD))


class KimiK3VisionProjector(Module):
    """PatchMergerMLPV2 projector from merged vision features to text width."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        merged_dim: int
        linear_1: Linear.Config
        linear_2: Linear.Config
        post_norm: RMSNorm.Config
        activation: GELU.Config = field(default_factory=GELU.Config)

    def __init__(self, config: Config):
        super().__init__()
        self.merged_dim = config.merged_dim
        self.linear_1 = config.linear_1.build()
        self.linear_2 = config.linear_2.build()
        self.post_norm = config.post_norm.build()
        self.activation = config.activation.build()

    def forward(self, merged_NMK: torch.Tensor) -> torch.Tensor:
        if merged_NMK.shape[-1] != self.merged_dim:
            raise ValueError(
                f"Expected merged vision dim {self.merged_dim}, got "
                f"{merged_NMK.shape[-1]}."
            )
        projected = self.linear_2(self.activation(self.linear_1(merged_NMK)))
        return self.post_norm(projected)


class KimiK3VisionEncoder(Module):
    """Device-neutral MoonViT3d encoder and PatchMergerMLPV2 projector."""

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
        self.dim = config.dim
        self.patch_size = config.patch_size
        self.in_channels = config.in_channels
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
        self._cached_freq_table: torch.Tensor | None = None
        self.layers = ModuleDict(
            {
                str(layer_idx): config.block.build()
                for layer_idx in range(config.num_layers)
            }
        )
        self.final_norm = config.final_norm.build()
        self.projector = config.projector.build()

    def _compute_position_embeddings(
        self, grids: list[list[int]], max_num_patches: int
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
            max_num_patches,
            self.interpolation_mode,
            self.max_num_frames,
        )
        rope_cache = _compute_2d_rope_cache(
            self._cached_freq_table,
            grids,
            max_num_patches,
            self.rotary_pos_emb.head_dim,
        )
        return learned_pos, rope_cache

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Encode padded raster-order patches and return padded text features."""
        if grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
            raise ValueError(f"grid_thw must have shape (N, 3), got {grid_thw.shape}.")
        num_items, max_num_patches, _ = pixel_values.shape
        grids = grid_thw.tolist()
        if len(grids) != num_items:
            raise ValueError(
                f"pixel_values contains {num_items} items but grid_thw "
                f"contains {len(grids)}."
            )

        kernel_h, kernel_w = self.merge_kernel_size
        num_patches = []
        for num_frames, grid_h, grid_w in grids:
            if grid_h % kernel_h != 0 or grid_w % kernel_w != 0:
                raise ValueError(
                    f"Vision grid {grid_h}x{grid_w} is not divisible by "
                    f"merge kernel {self.merge_kernel_size}."
                )
            item_num_patches = num_frames * grid_h * grid_w
            if item_num_patches > max_num_patches:
                raise ValueError(
                    f"Vision grid requires {item_num_patches} patches, but "
                    f"pixel_values only provides {max_num_patches}."
                )
            num_patches.append(item_num_patches)

        learned_pos, rope_cache = self._compute_position_embeddings(
            grids, max_num_patches
        )
        hidden_NPD = self.patch_embed(pixel_values) + learned_pos
        for block in self.layers.values():
            hidden_NPD = block(
                hidden_NPD,
                rope_cache=rope_cache,
                num_patches=num_patches,
            )
        hidden_NPD = self.final_norm(hidden_NPD)
        merged_NMK = _temporal_pool_and_merge(hidden_NPD, grids, self.merge_kernel_size)
        return self.projector(merged_NMK)
