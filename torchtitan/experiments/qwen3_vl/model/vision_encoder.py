# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Qwen3-VL Vision Encoder implementation.

This module implements the Vision Transformer (ViT) encoder used in Qwen3-VL,
using FlexAttention with padded batches for efficient processing.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from torchtitan.models.attention import FlexAttentionWrapper

from .args import Qwen3VLVisionEncoderArgs


# Compiled block mask creation
_compiled_create_block_mask = torch.compile(create_block_mask)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision_batched(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embeddings to query and key tensors.

    Args:
        q: (batch, max_num_patch, heads, head_dim)
        k: (batch, max_num_patch, heads, head_dim)
        cos: (batch, max_num_patch, head_dim)
        sin: (batch, max_num_patch, head_dim)
    """
    # Expand cos/sin for heads dimension, keep in original dtype
    cos = cos.unsqueeze(2)  # (batch, max_num_patch, 1, head_dim)
    sin = sin.unsqueeze(2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def get_vision_block_mask_mod(num_patch: torch.Tensor, max_num_patch: int):
    """Create a mask modifier for block-diagonal attention.

    Each image only attends to its own patches.

    Args:
        num_patch: (num_images,) actual number of patches per image
        max_num_patch: Maximum number of patches (padded length)
    """
    def mask_mod(b, h, q_idx, kv_idx):
        # Check if both indices are within the valid range for this image
        valid_q = q_idx < num_patch[b]
        valid_kv = kv_idx < num_patch[b]
        return valid_q & valid_kv

    return mask_mod


class PatchEmbed(nn.Module):
    """Patch Embedding using Linear projection.

    Since patches are already extracted by the collator, we use Linear instead of Conv3d.
    This is mathematically equivalent when Conv3d kernel_size equals input size:
    - Conv3d: (B, C, T, H, W) with kernel=C*(T,H,W) and dim kernels → (B, dim, 1, 1, 1)
    - Linear: (B, C*T*H*W) → (B, dim)
    Same weighted sum, but Linear uses efficient batched matrix multiplication.
    """

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1280,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # patch_dim matches the flattened patch from collator: (pt * ph * pw * c)
        self.patch_dim = in_channels * temporal_patch_size * patch_size * patch_size
        self.proj = nn.Linear(self.patch_dim, embed_dim, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, max_num_patch, patch_dim)
        Returns:
            (batch, max_num_patch, embed_dim)
        """
        return self.proj(hidden_states.to(dtype=self.proj.weight.dtype))

    def init_weights(self):
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)


class VisionRotaryEmbedding(nn.Module):
    """2D Rotary Position Embedding for Vision Transformer."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def init_weights(self, device: torch.device | None = None):
        """Re-compute inv_freq on the target device after to_empty()."""
        device = device or self.inv_freq.device
        self.inv_freq = (
            1.0
            / (
                self.theta
                ** (torch.arange(0, self.dim, 2, dtype=torch.float, device=device) / self.dim)
            )
        )

    def forward(self, seqlen: int) -> torch.Tensor:
        """Compute rotary embeddings for a sequence."""
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class PatchMerger(nn.Module):
    """Merge spatial patches to reduce sequence length.

    Args:
        hidden_size: Hidden dimension of input features.
        out_hidden_size: Output hidden dimension after merging.
        spatial_merge_size: Number of patches to merge per spatial dimension.
        use_postshuffle_norm: If True, apply LayerNorm after spatial reshape
            (norm dim = hidden_size * spatial_merge_size^2). If False, apply
            before reshape (norm dim = hidden_size). DeepStack mergers use
            postshuffle norm; the main merger uses pre-shuffle norm.
    """

    def __init__(
        self,
        hidden_size: int,
        out_hidden_size: int,
        spatial_merge_size: int = 2,
        use_postshuffle_norm: bool = False,
    ):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.merged_hidden_size = hidden_size * (spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm

        norm_dim = self.merged_hidden_size if use_postshuffle_norm else hidden_size
        self.norm = nn.LayerNorm(norm_dim, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.merged_hidden_size, self.merged_hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.merged_hidden_size, out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size) where seq_len is divisible by spatial_merge_size^2
        Returns:
            (batch, seq_len // spatial_merge_size^2, out_hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        if self.use_postshuffle_norm:
            # Reshape first, then norm
            x = x.view(batch_size, seq_len // (self.spatial_merge_size ** 2), self.merged_hidden_size)
            x = self.norm(x)
        else:
            # Norm first, then reshape
            x = self.norm(x)
            x = x.view(batch_size, seq_len // (self.spatial_merge_size ** 2), self.merged_hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x

    def init_weights(self):
        self.norm.reset_parameters()
        nn.init.trunc_normal_(self.linear_fc1.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.linear_fc2.weight, mean=0.0, std=0.02)
        if self.linear_fc1.bias is not None:
            nn.init.zeros_(self.linear_fc1.bias)
        if self.linear_fc2.bias is not None:
            nn.init.zeros_(self.linear_fc2.bias)


class VisionAttention(nn.Module):
    """Multi-head attention with FlexAttention for efficient batched processing."""

    def __init__(self, args: Qwen3VLVisionEncoderArgs):
        super().__init__()
        self.dim = args.dim
        self.num_heads = args.n_heads
        self.head_dim = self.dim // self.num_heads

        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.flex_attention = FlexAttentionWrapper()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: BlockMask,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (num_images, max_num_patch, dim)
            position_embeddings: (cos, sin) each of shape (num_images, max_num_patch, head_dim)
            attention_mask: BlockMask for attention

        Returns:
            (num_images, max_num_patch, dim)
        """
        num_images, max_num_patch, _ = hidden_states.shape

        # QKV projection
        qkv = self.qkv(hidden_states).reshape(num_images, max_num_patch, 3, -1, self.head_dim)
        q, k, v = qkv.permute(2, 0, 1, 3, 4).unbind(0)  # Each: (num_images, max_num_patch, heads, head_dim)

        # Apply rotary embeddings
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision_batched(q, k, cos, sin)

        # Reshape for attention: (num_images, heads, max_num_patch, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # FlexAttention
        attn_output = self.flex_attention(q, k, v, block_mask=attention_mask)

        # Reshape back: (num_images, max_num_patch, dim)
        attn_output = attn_output.transpose(1, 2).reshape(num_images, max_num_patch, -1)
        return self.proj(attn_output)

    def init_weights(self):
        nn.init.trunc_normal_(self.qkv.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=0.02)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)


class VisionMLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, args: Qwen3VLVisionEncoderArgs):
        super().__init__()
        self.linear_fc1 = nn.Linear(args.dim, args.ffn_dim, bias=True)
        self.linear_fc2 = nn.Linear(args.ffn_dim, args.dim, bias=True)
        self.act_fn = nn.GELU()

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))

    def init_weights(self):
        nn.init.trunc_normal_(self.linear_fc1.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.linear_fc2.weight, mean=0.0, std=0.02)
        if self.linear_fc1.bias is not None:
            nn.init.zeros_(self.linear_fc1.bias)
        if self.linear_fc2.bias is not None:
            nn.init.zeros_(self.linear_fc2.bias)


class VisionTransformerBlock(nn.Module):
    """Single transformer block for vision encoder."""

    def __init__(self, args: Qwen3VLVisionEncoderArgs):
        super().__init__()
        self.norm1 = nn.LayerNorm(args.dim, eps=args.layer_norm_eps)
        self.norm2 = nn.LayerNorm(args.dim, eps=args.layer_norm_eps)
        self.attn = VisionAttention(args)
        self.mlp = VisionMLP(args)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: BlockMask,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states

    def init_weights(self):
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.attn.init_weights()
        self.mlp.init_weights()


class Qwen3VLVisionEncoder(nn.Module):
    """
    Qwen3-VL Vision Encoder with FlexAttention.

    Uses padded batches (N, L, D) format for efficient processing.
    """

    def __init__(self, args: Qwen3VLVisionEncoderArgs):
        super().__init__()
        self.args = args
        self.spatial_merge_size = args.spatial_merge_size
        self.patch_size = args.patch_size
        self.spatial_merge_unit = args.spatial_merge_size ** 2

        # Patch embedding
        self.patch_embed = PatchEmbed(
            patch_size=args.patch_size,
            temporal_patch_size=args.temporal_patch_size,
            in_channels=args.in_channels,
            embed_dim=args.dim,
        )

        # Position embeddings (learnable, with bilinear interpolation)
        self.num_position_embeddings = args.num_position_embeddings
        self.pos_embed = nn.Embedding(args.num_position_embeddings, args.dim)
        self.num_grid_per_side = int(args.num_position_embeddings ** 0.5)

        # Rotary embeddings for 2D positions
        head_dim = args.dim // args.n_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2, theta=args.rope_theta)

        # Transformer layers
        self.layers = nn.ModuleDict(
            {str(idx): VisionTransformerBlock(args) for idx in range(args.n_layers)}
        )

        # Main patch merger usd in the last layer
        self.merger = PatchMerger(
            hidden_size=args.dim,
            out_hidden_size=args.out_hidden_size,
            spatial_merge_size=args.spatial_merge_size,
        )

        # DeepStack mergers for intermediate layers
        # DeepStack mergers use postshuffle norm (norm after spatial reshape)
        self.deepstack_visual_indicies = args.deepstack_visual_indicies
        self.deepstack_merger_list = nn.ModuleList([
            PatchMerger(
                hidden_size=args.dim,
                out_hidden_size=args.out_hidden_size,
                spatial_merge_size=args.spatial_merge_size,
                use_postshuffle_norm=True,
            )
            for _ in range(len(args.deepstack_visual_indicies))
        ])

    def init_weights(self):
        self.patch_embed.init_weights()
        nn.init.trunc_normal_(self.pos_embed.weight, mean=0.0, std=0.02)
        self.rotary_pos_emb.init_weights()
        self.merger.init_weights()
        for merger in self.deepstack_merger_list:
            merger.init_weights()
        for block in self.layers.values():
            block.init_weights()

    def compute_position_embeddings(
        self,
        grid_thw: torch.Tensor,
        max_num_patch: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute position embeddings for padded batch.

        Args:
            grid_thw: (num_images, 3) with [t, h, w] per image
            max_num_patch: Maximum number of patches (for padding)

        Returns:
            pos_embeds: (num_images, max_num_patch, dim) learnable position embeddings
            rope_cos: (num_images, max_num_patch, head_dim) RoPE cosines
            rope_sin: (num_images, max_num_patch, head_dim) RoPE sines
        """
        num_images = grid_thw.shape[0]
        device = grid_thw.device
        dtype = self.pos_embed.weight.dtype
        merge_size = self.spatial_merge_size
        head_dim = self.args.dim // self.args.n_heads

        # Compute max height/width for RoPE freq table
        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)

        # Get pos_embed weights for direct indexing (avoid nn.Embedding forward overhead)
        # Convert to local tensor if DTensor (from FSDP/TP wrapping) to allow
        # regular tensor indexing operations.
        pos_embed_weight = self.pos_embed.weight
        if hasattr(pos_embed_weight, "to_local"):
            pos_embed_weight = pos_embed_weight.to_local()

        # Pre-allocate output tensors
        pos_embeds = torch.zeros(num_images, max_num_patch, self.args.dim, device=device, dtype=dtype)
        rope_embeds = torch.zeros(num_images, max_num_patch, head_dim // 2, device=device, dtype=dtype)

        # Group images by (h, w) to batch compute position embeddings
        hw_to_indices: dict[tuple[int, int], list[int]] = {}
        for i in range(num_images):
            h, w = int(grid_thw[i, 1].item()), int(grid_thw[i, 2].item())
            key = (h, w)
            if key not in hw_to_indices:
                hw_to_indices[key] = []
            hw_to_indices[key].append(i)

        for (h, w), indices in hw_to_indices.items():
            # Compute bilinear interpolated position embeddings once per unique (h, w)
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h, device=device)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w, device=device)

            h_floor, w_floor = h_idxs.long(), w_idxs.long()
            h_ceil = (h_floor + 1).clamp(max=self.num_grid_per_side - 1)
            w_ceil = (w_floor + 1).clamp(max=self.num_grid_per_side - 1)

            dh, dw = h_idxs - h_floor.float(), w_idxs - w_floor.float()

            # Compute indices for 4 corners
            # [idx_00 idx_01
            #  idx_10 idx_11]
            # Map from 2d indices to 1d indices used in pos_embed_weight
            base_h = h_floor * self.num_grid_per_side
            base_h_ceil = h_ceil * self.num_grid_per_side
            idx_00 = (base_h[:, None] + w_floor[None, :]).flatten()
            idx_01 = (base_h[:, None] + w_ceil[None, :]).flatten()
            idx_10 = (base_h_ceil[:, None] + w_floor[None, :]).flatten()
            idx_11 = (base_h_ceil[:, None] + w_ceil[None, :]).flatten()

            # Bilinear weights
            w_00 = ((1 - dh)[:, None] * (1 - dw)[None, :]).flatten()
            w_01 = ((1 - dh)[:, None] * dw[None, :]).flatten()
            w_10 = (dh[:, None] * (1 - dw)[None, :]).flatten()
            w_11 = (dh[:, None] * dw[None, :]).flatten()

            # Accumulate bilinear interpolation (avoids materializing all 4 corners)
            pos_hw  = w_00[:, None] * pos_embed_weight[idx_00]
            pos_hw += w_01[:, None] * pos_embed_weight[idx_01]
            pos_hw += w_10[:, None] * pos_embed_weight[idx_10]
            pos_hw += w_11[:, None] * pos_embed_weight[idx_11]

            # Compute RoPE position ids in block order (once per unique h, w)
            merged_h, merged_w = h // merge_size, w // merge_size

            # Create block order indices using einops-style reshaping
            row_base = torch.arange(merged_h, device=device) * merge_size # starting row of each block
            col_base = torch.arange(merged_w, device=device) * merge_size # starting col of each block
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            # Build row and col indices in block order
            # The 4 dimensions represent: [block_row, block_col, intra_row, intra_col]
            row_idx = (row_base[:, None, None, None] + intra_row[None, None, :, None]).expand(
                merged_h, merged_w, merge_size, merge_size
            ).reshape(-1)
            col_idx = (col_base[None, :, None, None] + intra_col[None, None, None, :]).expand(
                merged_h, merged_w, merge_size, merge_size
            ).reshape(-1)

            # Get RoPE embeddings for this (h, w) - freq_table is (max_hw, head_dim//4)
            rope_row = freq_table[row_idx]  # (h*w, head_dim//4)
            rope_col = freq_table[col_idx]  # (h*w, head_dim//4)
            rope_2d = torch.cat([rope_row, rope_col], dim=-1)  # (h*w, head_dim//2)

            # Permute learned pos_hw to block order
            # Before permute: (block_row, intra_row, block_col, intra_col, dim)
            # After permute:  (block_row, block_col, intra_row, intra_col, dim)
            pos_hw_block = (
                pos_hw.view(h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 2, 1, 3, 4)
                .flatten(0, 3)
            )  # (h*w, dim)

            # Apply to all images with this (h, w)
            for i in indices:
                t = int(grid_thw[i, 0].item())
                seq_len = t * h * w

                if t > 1:
                    pos_embeds[i, :seq_len] = pos_hw_block.repeat(t, 1)
                    rope_embeds[i, :seq_len] = rope_2d.repeat(t, 1)
                else:
                    pos_embeds[i, :seq_len] = pos_hw_block
                    rope_embeds[i, :seq_len] = rope_2d

        # Compute cos/sin from RoPE embeddings
        rope_embeds = torch.cat((rope_embeds, rope_embeds), dim=-1)  # (N, L, head_dim)
        rope_cos = rope_embeds.cos()
        rope_sin = rope_embeds.sin()

        return pos_embeds, rope_cos, rope_sin

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass of the vision encoder.

        Args:
            pixel_values: Padded patches (num_images, max_num_patch, patch_dim)
            grid_thw: Grid dimensions (num_images, 3) for [temporal, height, width] measured in patches

        Returns:
            merged_hidden_states: (num_images, max_merged_num_patch, out_hidden_size)
            deepstack_features: List of features from intermediate layers
        """
        num_images, max_num_patch, _ = pixel_values.shape

        # Compute actual number of patches per image
        num_patch = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).to(torch.long)

        # Patch embedding -> [num_images, max_num_patch, dim]
        hidden_states = self.patch_embed(pixel_values)

        # Compute position embeddings (learned + RoPE)
        pos_embeds, rope_cos, rope_sin = self.compute_position_embeddings(grid_thw, max_num_patch)
        hidden_states = hidden_states + pos_embeds

        # Create attention mask for block-diagonal attention
        mask_mod = get_vision_block_mask_mod(num_patch, max_num_patch)
        attention_mask = _compiled_create_block_mask(
            mask_mod, num_images, None, max_num_patch, max_num_patch
        )

        # Apply transformer layers with DeepStack extraction
        position_embeddings = (rope_cos, rope_sin)
        deepstack_feature_lists = []

        for layer_num, blk in self.layers.items():
            hidden_states = blk(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )
            if int(layer_num) in self.deepstack_visual_indicies:
                idx = self.deepstack_visual_indicies.index(int(layer_num))
                deepstack_feature = self.deepstack_merger_list[idx](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        # Final merge
        merged_hidden_states = self.merger(hidden_states)

        return merged_hidden_states, deepstack_feature_lists
